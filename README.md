# EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data

This repository provides the PyTorch implementation of the paper [EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data]().
This branch contains complete source code and model weights for the **object detection** experiments in the paper. 

<p align="center">
  <img src="resources/eurnet.png"/> 
</p>

EurNet employs the Gated Relational Message Passing (GRMP) layer as its basic component, as graphically shown above.
EurNet can be applied to various domains (e.g., images, protein structures and knowledge graphs) for efficient multi-relational modeling at scale.
Here are the links to other applied domains/tasks of this project:
- [EurNet for Image Classification (TorchDrug implementation)](https://github.com/hirl-team/EurNet-Image/tree/main)
- [EurNet for Image Classification (PyTorch Geometric implementation)](https://github.com/hirl-team/EurNet-Image/tree/pyg-cls)
- [EurNet for Semantic Segmentation]()
- EurNet for Protein Structure Modeling (*Working, will release soon*)
- EurNet for Knowledge Graph Reasoning (*Working, will release soon*)

## Roadmap
- [2022/11/xx] The initial release! We release all source code and model weights of EurNet for image classification (both TorchDrug and PyTorch Geometric implementations), object detection and semantic segmentation. 

## TODO
- [ ] Release code and model weights of EurNet for protein structure modeling.
- [ ] Release code and model weights of EurNet for knowledge graph reasoning. 

## Benchmark and Model Zoo

### Pre-training on ImageNet-1K

Following [FocalNet](https://arxiv.org/pdf/2203.11926), we fine-tune the models pre-trained on ImageNet-1K classification.
The pre-trained checkpoints for EurNet-T/S/B can be downloaded with the following links:
- [Pre-trained EurNet-T]()
- [Pre-trained EurNet-S]()
- [Pre-trained EurNet-B]()

### Fine-tuning on COCO

The experiments are conducted with [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf).

|    Model     |   Schedule    | #Params. (M) | FLOPs (G) | box mAP | mask mAP |                                                        Config                                                         |   Ckpt   |   Log   |
|:------------:|:-----------:|:------------:|:------:|:-------:|:--------:|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|
|   EurNet-T   | 1x   | 49.8 | 271 |  46.1   |  41.6     |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_tiny_1x_coco.py)     | [ckpt]() | [log]() |
|   EurNet-T   |  3x   | 49.8 | 271 |  47.8    |   42.9    |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_tiny_3x_coco.py)   | [ckpt]() | [log]() |
|   EurNet-S   | 1x   | 72.8 | 364 |  48.4   |  43.2     |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_small_1x_coco.py)     | [ckpt]() | [log]() |
|   EurNet-S   |  3x   | 72.8 | 364 |  49.4    |   44.0    |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_small_3x_coco.py)   | [ckpt]() | [log]() |
|   EurNet-B   | 1x   | 112.1 | 506 |  49.3   |  43.9     |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_base_1x_coco.py)     | [ckpt]() | [log]() |
|   EurNet-B   |  3x   | 112.1 | 506 |  50.1    |   44.5    |    [config](https://github.com/hirl-team/EurNet-Image/blob/det/detection/configs//mask_rcnn/mask_rcnn_eurnet_base_3x_coco.py)   | [ckpt]() | [log]() |


## Installation

This repository is officially tested with the following environments:
- Linux
- Python 3.6+
- PyTorch 1.10.0
- CUDA 11.3

The environment could be prepared in the following steps:
1. Create a virtual environment with conda:
```
conda create -n eurnet_det python=3.7.3 -y
conda activate eurnet_det
```
2. Install PyTorch with the [official instructions](https://pytorch.org/). For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Install `mmcv` and `mmdetection` for object detection:
```
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.25.0
```

## Usage

### Prepare Dataset

#### COCO

COCO dataset is used for object detection and instance segmentation experiments. 
Download train/val2017 splits from the [official website](https://cocodataset.org/#download). We recommend 
symlink the dataset folder to `./detection/datasets/coco/`. The folder structure is expected to be:
```
detection/
  datasets/
    coco/
      annotations/
        instances_train2017.json
        instances_val2017.json
        ...
      train2017/
      val2017/
```

## Launch Experiments

We follow `mmdetection` to use python based configuration file system. The config could be modified by command line arguments.

To run an experiment:
```
python3 ./detection/launch.py -c [config file] --output_dir [output directory] [config options]
```
The config options are in "key=value" format. For example, `optimizer.type=AdamW` modifies the sub key `type` in `optimizer` with value `AdamW`.

A full example of training and evaluating EurNet-T with Mask-RCNN on COCO for 1x schedule (12 epochs starting from an ImageNet-1K pre-trained checkpoint):
```
python3 ./detection/launch.py -c configs/mask_rcnn/mask_rcnn_eurnet_tiny_1x_coco.py \
--output_dir ./experiments/coco/maskrcnn-1x/EurNet-T/ --pretrained [path of imagenet pretrained model]
```

**Note that**, the root of relative dir is `./detection`. All the training configuration files are in `./detection/configs/`. 
To reproduce the COCO object detection results, please **follow the corresponding config file** and **remember to load the ImageNet-1K pre-trained checkpoint**. 

## License

This repository is released under the MIT license as in the [LICENSE](LICENSE) file.

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@article{xu2022eurnet,
  title={EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data},
  author={Xu, Minghao and Guo, Yuanfan and Xu, Yi and Tang, Jian and Chen, Xinlei and Tian, Yuandong},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

## Acknowledgements

The development of this project is guided by the following open-source projects. 
We would like to thank the authors for releasing the source code.
- [FocalNet](https://github.com/microsoft/FocalNet)
- [Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)
