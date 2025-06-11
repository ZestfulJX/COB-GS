import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

import argparse

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description='extract mask')
parser.add_argument('--resolution', type=int, default=-1)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--text', type=str, default='')
parser.add_argument('--frame_idx', type=int, default=0)
args = parser.parse_args()

# init sam image predictor and video predictor model
sam2_checkpoint = "./submodules/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino model
device = "cuda" if torch.cuda.is_available() else "cpu"
grounding_model = load_model(
    model_config_path="./submodules/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    model_checkpoint_path="./submodules/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
    device=device
)
# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot

def downsample_images(images_folder, output_path, resolution):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(images_folder):
        if filename.endswith(('.png', '.jpg', '.JPG', '.jpeg', '.bmp')):
            img_path = os.path.join(images_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                new_width = img.shape[1] // resolution
                new_height = img.shape[0] // resolution

                downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                output_img_path = os.path.join(output_path, filename)
                cv2.imwrite(output_img_path, downsampled_img)
            else:
                print(f"None: {img_path}")

dataset = args.dataset
output_path = args.output
scene = args.scene
text = args.text
resolution = args.resolution

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
if dataset == '3DOVS':
    video_dir = os.path.join("/data_nvme/zjx/3DOVS", scene, 'images')
elif dataset == 'llff':
    video_dir = os.path.join("/data_nvme/zjx/llff_mask", scene, 'images')
elif dataset == 'lerf':
    video_dir = os.path.join("/data_nvme/zjx/lerf_mask", scene, 'images_train')
elif dataset == 'in2n':
    video_dir = os.path.join("/data_nvme/zjx/in2n", scene, 'images')
elif dataset == '360':
    video_dir = os.path.join("/data_nvme/zjx/360_v2", scene, 'images')
elif dataset == 'tnt':
    video_dir = os.path.join("/data_nvme/zjx/tandt", scene, 'images')
else:
    assert False, "Dataset not handled!"
# scan all the JPEG frame names in this directory
if resolution != -1:
    print(os.path.exists(video_dir+f"_{resolution}"))
    if not os.path.exists(video_dir+f"_{resolution}"):
        print("create dataset....")
        downsample_images(video_dir, video_dir+f"_{resolution}", resolution)
    video_dir = video_dir+f"_{resolution}"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
]
frame_names.sort(key=lambda p: os.path.splitext(p)[0])

num_frames = len(frame_names)

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)

ann_frame_idx = args.frame_idx  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(video_dir, frame_names[ann_frame_idx])

image_source, image = load_image(img_path)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=0.3,
    text_threshold=0.45
)
# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(image_source)

# process the detection results
# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
confidences = confidences.numpy().tolist()

class_names = labels

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

max_confidence = np.max(confidences)

threshold = max_confidence * 0.30

high_confidence_indices = np.where(np.abs(np.array(confidences) - max_confidence) <= threshold)[0]

labels = [labels[i] for i in high_confidence_indices]
input_boxes = [input_boxes[i] for i in high_confidence_indices]

OBJECTS = labels

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
if ann_frame_idx == 0:
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=ann_frame_idx,
                                                                                          reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
elif ann_frame_idx == num_frames - 1:
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=ann_frame_idx,
                                                                                          reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
else:
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=ann_frame_idx,
                                                                                          reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                          start_frame_idx=ann_frame_idx,
                                                                                          reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

valid_segments_dix = {}
for frame_idx, segments in video_segments.items():
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    xyxy = sv.mask_to_xyxy(masks)[0]
    if xyxy[0] == 0 and xyxy[1] == 0 and xyxy[2] == 0 and xyxy[3] == 0:
        valid_segments_dix[frame_idx] = 0
    else:
        valid_segments_dix[frame_idx] = 1

global_idx = 0
while global_idx < len(frame_names):
    if global_idx in valid_segments_dix and valid_segments_dix[global_idx] == 1:
        global_idx += 1
        continue
    else:
        img_path = os.path.join(video_dir, frame_names[global_idx])
        print("empty mask " + img_path)
        image_source, image = load_image(img_path)
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text,
            box_threshold=max(0.6, max_confidence * 0.85),
            text_threshold=max(0.65, max_confidence * 0.85)
        )
        if boxes.numel() == 0:
            global_idx += 1
            continue
        else:
            image_predictor.set_image(image_source)

            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            assert PROMPT_TYPE_FOR_VIDEO in ["point", "box",
                                             "mask"], "SAM 2 video predictor only support point/box/mask prompt"

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if PROMPT_TYPE_FOR_VIDEO == "point":
                # sample the positive points from mask for each objects
                all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

                for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                    labels = np.ones((points.shape[0]), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )
            # Using box prompt
            elif PROMPT_TYPE_FOR_VIDEO == "box":
                for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        box=box,
                    )
            # Using mask prompt is a more straightforward way
            elif PROMPT_TYPE_FOR_VIDEO == "mask":
                for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                    labels = np.ones((1), dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=global_idx,
                        obj_id=object_id,
                        mask=mask
                    )
            else:
                raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
            max_zero_index = global_idx

            for idx in range(global_idx + 1, len(valid_segments_dix)):
                if idx in valid_segments_dix and valid_segments_dix[idx] == 0:
                    max_zero_index = idx
                else:
                    break
            print(f"Last frame with empty mask: {max_zero_index}")
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=max_zero_index-global_idx, start_frame_idx=global_idx):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                masks = list(video_segments[out_frame_idx].values())
                masks = np.concatenate(masks, axis=0)
                xyxy = sv.mask_to_xyxy(masks)[0]
                if xyxy[0] == 0 and xyxy[1] == 0 and xyxy[2] == 0 and xyxy[3] == 0:
                    valid_segments_dix[out_frame_idx] = 0
                else:
                    valid_segments_dix[out_frame_idx] = 1

    global_idx += 1
"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = "."
save_dir = os.path.join(save_dir, output_path,  scene, "masks", text)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    masks = np.max(masks, axis=0)
    cv2.imwrite(os.path.join(save_dir, frame_names[frame_idx]), (masks * 255).astype(np.uint8))
    # # Visualization
    # detections = sv.Detections(
    #     xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
    #     mask=masks, # (n, h, w)
    #     class_id=np.array(object_ids, dtype=np.int32),
    # )
    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    # label_annotator = sv.LabelAnnotator()
    # annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    # mask_annotator = sv.MaskAnnotator()
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(save_dir, frame_names[frame_idx]), annotated_frame)