# #!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# llff
#dataset_path="/data_nvme/zjx/nerf_llff_data"                             # mask_signals_threshold = 0.5; N4views = 22(10+10+2)
#scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "trex")

# 360_v2
#dataset_path="/data_nvme/zjx/360_v2"                                     # mask_signals_threshold = 0.8; N4views = 14(6+6+2)
#scenes=("garden")
#scenes=("kitchen")

# tnt
dataset_path="/data_nvme/zjx/tandt"                                       # mask_signals_threshold = 0.8; N4views = 14(6+6+2)
scenes=("truck")

output_path="output"


declare -A texts
texts["fern"]="fern"
texts["flower"]="flower"
texts["fortress"]="fortress"
texts["horns"]="horns_center;horns_left"
texts["leaves"]="leaves"
texts["orchids"]="orchids"
texts["trex"]="trex"

texts["garden"]="The bonsai"
texts["kitchen"]="Lego"
texts["truck"]="The truck"

for scene in "${scenes[@]}"; do

    echo "Processing scene: $scene"

    # 1. Optimize 3DGS scenes, any 3DGS scenes using the original pipeline are allowed.

    # llff
#    python train.py -s "$dataset_path/$scene" -m "$output_path/$scene" --eval -r 4
#    python render.py -m "$output_path/$scene" -r 4

     # 360_v2/tandt
    python train.py -s "$dataset_path/$scene" -m "$output_path/$scene"  --images "images_4"
    python render.py -m "$output_path/$scene" --images "images_4"

    IFS=';' read -r -a text_array <<< "${texts[$scene]}"
    for text in "${text_array[@]}"; do
        text=$(echo "$text" | xargs)

        # 2. Prepare mask

        # 1) Dataset NVOS dataset provides point-based prompts, and we provide masks obtained from points. It is fair to perform scene segmentation based on consistent 2D masks, not just consistent points.

        # 2) Extract the mask corresponding to the text. We provide a stable sequence masks extraction method based on Grounded-SAM-2.

    #    python submodules/Grounded-SAM-2/grounded_sam2_stable_tracking.py --dataset "360" --output "$output_path" --scene "$scene" --text "$text" --resolution 4 --frame_idx 0
       python submodules/Grounded-SAM-2/grounded_sam2_stable_tracking.py --dataset "tnt" --output "$output_path" --scene "$scene" --text "$text" --resolution 4 --frame_idx 0

        # 3. 3DGS segmentation

        # llff
        # python train.py -s "$dataset_path/$scene" -m "$output_path/${scene}" --eval --start_checkpoint "$output_path/$scene/chkpnt30000.pth" --include_mask --finetune_mask --text "$text" -r 4 --N4views 22 --mask_signals_threshold 0.5

        # python render.py -m "$output_path/${scene}" --include_mask --finetune_mask --skip_train --text "$text" -r 4 --N4views 22 -w

        # 360/tandt
        python train.py -s "$dataset_path/$scene" -m "$output_path/${scene}" --start_checkpoint "$output_path/$scene/chkpnt30000.pth" --include_mask --finetune_mask --text "$text" --images "images_4" --N4views 14 --mask_signals_threshold 0.8

        python render.py -m "$output_path/${scene}" --include_mask --finetune_mask --text "$text" --images "images_4" --N4views 14 -w
    done
done