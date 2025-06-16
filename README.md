# COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting

Accepted by CVPR 2025

### [Webpage](https://cob-gs.github.io/) | [Paper](https://arxiv.org/pdf/2503.19443) | [arXiv](https://arxiv.org/abs/2503.19443)

This repository contains the official authors implementation associated with the paper "COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting". We further introduce how to complete 3DGS segmentation with only images and text prompts.

## Environment Setup
To prepare the environment, 

1. Clone this repository. 
	```
	git clone https://github.com/ZestfulJX/COB-GS.git
	```
2. Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies. 
   	```
	conda env create --file environment.yml
    conda activate cobgs
	```
	Please notice, that the ```diff-gaussian-rasterization``` module contained in this repository has integrated the mask training branch to implement ```Boundary-Adaptive Gaussian Splitting```.

3. Install [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2).
   
   We provide a stable sequence masks extraction method based on Grounded-SAM-2 in ```./submodules/Grounded-SAM-2-utils```.
	```
	cd submodules
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    cd checkpoints
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    cd ..
    cd gdino_checkpoints
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    cd ..

    pip install -e .
    pip install --no-build-isolation -e grounding_dino

    cd ../..
    cp ./submodules/Grounded-SAM-2-utils/grounded_sam2_tracking_demo.py ./submodules/Grounded-SAM-2
	```
    

## Run COB-GS

We provide ```process.sh``` to easily implement the complete segmentation process, which only requires the image sequence of the scene and the text prompts of the segmented parts.

1. Train 3DGS
  ```
    python train.py -s "dataset/tandt/truck" -m "output/truck"  --images "images_4"
  ```
2. Extract masks based on text prompt
  ```
    python submodules/Grounded-SAM-2/grounded_sam2_stable_tracking.py --dataset "tnt" --output "output" --scene "truck" --text "The truck" --resolution 4
  ```
3. Run 3DGS segmentation
   
  ```
    python train.py -s "dataset/tandt/truck" -m "output/truck" --start_checkpoint "output/truck/chkpnt30000.pth" --include_mask --finetune_mask --text "The truck" --images "images_4" --N4views 14 --mask_signals_threshold 0.8
  ```
  - ```--include_mask```: Add mask to the render.
  - ```--finetune_mask```: Split the boundary Gaussian using mask gradient. Using only ```include_mask``` does not change the structure of the scene.
  - ```--N4views```: ```L``` images, additionally optimize ```L*N4views``` epochs.
  - ```--mask_signals_threshold```: Threshold of relative distance. 

Noting the need for fair comparison, we provide [masks](https://drive.google.com/drive/folders/1mMwj1510hb0PMEnxjUpzIDe2N3EL2PUF?usp=sharing) obtained on the [NVOS dataset](https://jason718.github.io/nvos/) based on points prompts. Under our project, just put them under the ```./output``` folder and skip ```Extract masks based on text prompt```. Finally different scenes are evaluated in ```eval/eval_NVOS.py```

### TODO List
- [ ]  Update efficient multi-object segmentation.
- [ ]  Update efficient texture optimizations.
- [ ]  Provide demo and more visualizations.

## Citation
*If you find this project helpful for your research, please consider citing the report and giving a ⭐.*

*Any questions are welcome for discussion.*
```
@inproceedings{zhang2025cobgs,
  title     = {COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting},
  author    = {Zhang, Jiaxin and Jiang, Junjun and Chen, Youyu and Jiang, Kui and Liu, Xianming},
  booktitle = {CVPR},
  year      = {2025}
}
```
