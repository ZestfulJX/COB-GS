#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args):
    if not args.include_mask:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]

            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]

            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(str(args.N4views) + "x"), "mask_renders",
                                   args.text)
        image_path = os.path.join(model_path, name, "ours_{}".format(str(args.N4views) + "x"), "image_renders",
                                  args.text)
        depth_path = os.path.join(model_path, name, "ours_{}".format(str(args.N4views) + "x"), "depth_renders",
                                  args.text)

        makedirs(render_path, exist_ok=True)
        makedirs(image_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            renders = render(view, gaussians, pipeline, background, args, use_trained_exp=train_test_exp)
            depth = renders["depth"]
            mask = renders["mask"]
            render_image = renders["render"]
            mask[mask <= 0.5] = 0
            mask[mask != 0] = 1
            mask = mask[0, :, :]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            torchvision.utils.save_image(depth, os.path.join(depth_path, 'depth_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(render_image, os.path.join(image_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(mask, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        if not args.include_mask:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        else:
            scene = Scene(dataset, gaussians, shuffle=False)
            checkpoint = os.path.join(dataset.mask_path,
                                      'chkpnt' + str(len(scene.getTrainCameras()) * args.N4views) + '.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            gaussians.segment()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--N4views", type=int, default=20)
    parser.add_argument("--include_mask", action="store_true")
    parser.add_argument("--finetune_mask", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print(args.text)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)