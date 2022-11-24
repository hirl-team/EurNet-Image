# EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data

This repository provides the PyTorch implementation of the paper [EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data](https://arxiv.org/pdf/2211.12941.pdf).
This branch is the **PyTorch Geometric** implementation of EurNet for image classification.
This branch provides source code and model weights for EurNet-T/S/B trained from scratch on ImageNet-1K.

<p align="center">
  <img src="resources/eurnet.png"/> 
</p>

EurNet employs the Gated Relational Message Passing (GRMP) layer as its basic component, as graphically shown above.
EurNet can be applied to various domains (e.g., images, protein structures and knowledge graphs) for efficient multi-relational modeling at scale.
Here are the links to other applied domains/tasks of this project:
- [EurNet for Image Classification (TorchDrug implementation)](https://github.com/hirl-team/EurNet-Image/tree/main)
- [EurNet for Object Detection](https://github.com/hirl-team/EurNet-Image/tree/det)
- [EurNet for Semantic Segmentation](https://github.com/hirl-team/EurNet-Image/tree/seg)
- EurNet for Protein Structure Modeling (*Working, will release soon*)
- EurNet for Knowledge Graph Reasoning (*Working, will release soon*)

## Roadmap
- [2022/11/24] The initial release! We release all source code and model weights of EurNet for image classification (both TorchDrug and PyTorch Geometric implementations), object detection and semantic segmentation. 

## TODO
- [ ] Release code and model weights of EurNet for protein structure modeling.
- [ ] Release code and model weights of EurNet for knowledge graph reasoning. 

## Benchmark and Model Zoo

|        Model         |   Training    | #Params. (M) | FLOPs (G) | IN-1K Top-1 (%) |                                                         Config                                                         |   Ckpt   |   Log   |
|:--------------------:|:-------------:|:------------:|:------:|:---------------:|:----------------------------------------------------------------------------------------------------------------------:|:--------:|:-------:|
| EurNet-T (TorchDrug) |  1K-scratch   | 29 | 4.6 |      82.3       |    [config](https://github.com/hirl-team/EurNet-Image/blob/main/configs/classification/eurnet_tiny_1k_300eps.yaml)     | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/td_eurnet_tiny_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/td_eurnet_tiny_1k_300eps_log.txt) |
|    EurNet-T (PyG)    |  1K-scratch   | 29 | 4.6 |      82.3       |    [config](https://github.com/hirl-team/EurNet-Image/blob/pyg-cls/configs/classification/eurnet_tiny_1k_300eps.yaml)     | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/pyg_eurnet_tiny_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/pyg_eurnet_tiny_1k_300eps_log.txt) |
| EurNet-S (TorchDrug) |  1K-scratch   | 50 | 8.8 |      83.6       |    [config](https://github.com/hirl-team/EurNet-Image/blob/main/configs/classification/eurnet_small_1k_300eps.yaml)    | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/td_eurnet_small_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/td_eurnet_small_1k_300eps_log.txt) |
|    EurNet-S (PyG)    |  1K-scratch   | 50 | 8.8 |      83.6       |    [config](https://github.com/hirl-team/EurNet-Image/blob/pyg-cls/configs/classification/eurnet_small_1k_300eps.yaml)    | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/pyg_eurnet_small_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/pyg_eurnet_small_1k_300eps_log.txt) |
| EurNet-B (TorchDrug) |  1K-scratch   | 89 | 15.6 |      84.1       |    [config](https://github.com/hirl-team/EurNet-Image/blob/main/configs/classification/eurnet_base_1k_300eps.yaml)     | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/td_eurnet_base_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/td_eurnet_base_1k_300eps_log.txt) |
|    EurNet-B (PyG)    |  1K-scratch   | 89 | 15.6 |      84.0       |    [config](https://github.com/hirl-team/EurNet-Image/blob/pyg-cls/configs/classification/eurnet_base_1k_300eps.yaml)     | [ckpt](https://eurnet.s3.us-east-2.amazonaws.com/checkpoints/pyg_eurnet_base_1k_300eps_last_epoch.pth) | [log](https://eurnet.s3.us-east-2.amazonaws.com/logs/pyg_eurnet_base_1k_300eps_log.txt) |

## Installation

This repository is officially tested with the following environments:
- Linux
- Python 3.6+
- PyTorch 1.10.0
- CUDA 11.3

The environment could be prepared in the following steps:
1. Create a virtual environment with conda:
```
conda create -n eurnet_pyg python=3.7.3 -y
conda activate eurnet_pyg
```
2. Install PyTorch with the [official instructions](https://pytorch.org/). For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Prepare Dataset

#### ImageNet-1K

We support ImageNet [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/) for training from scratch or fine-tuning. 

We recommend symlink the dataset folder to `./datasets/ImageNet1K`. The folder structure would be:
```
datasets/
  ImageNet1K/
    ILSVRC/
      Annotations/
      Data/
        train/
        val/
      ImageSets/
      meta/
```
After downloading and unzip the dataset, go to path `./datasets/ImageNet1K/ILSVRC/Data/val/` and move images to labeled sub-folders with [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Launch Experiments

We provide an easy yaml based configuration file system. The config could be modified by 
command line arguments.

To run an experiment:
```
python3 launch.py --launch ./tools/train.py -c [config file] [config options]
```
The config options are in "key=value" format. For example, `ouput_dir=your_path` and `batch_size=64`. 
Sub module is seperated by `.`. For example, `optimizer.name=AdamW` modifies the sub key `name` in 
`optimizer` with value `AdamW`.

A full example of training EurNet-T on ImageNet1K for 300 epochs:
```
python3 launch.py --launch ./tools/train.py -c configs/classification/eurnet_tiny_1k_300eps.yaml \
output_dir=./experiments/imagenet1k/EurNet-T-PyG/
```

**Note that**, all the training configuration files are in `./configs/classification/`. To reproduce the ImageNet classification results, please **follow the corresponding config file**. 

## Multinode training

`launch.py` would automatically find a free port to launch single node experiments. 
If multiple nodes training is necessary, the number of nodes `--nn`, node rank `--nr`, master port `--port` and master address `-ma` should be set. 

Take training EurNet-T on ImageNet-1K with two nodes as an example, use the following commands at node1 and node2, respectively.
```
# use this command at node 1
python3 launch.py --nn 2 --nr 0 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/classification/eurnet_tiny_1k_300eps.yaml

# use this command at node 2
python3 launch.py --nn 2 --nr 1 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/classification/eurnet_tiny_1k_300eps.yaml
```

## License

This repository is released under the MIT license as in the [LICENSE](LICENSE) file.

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@article{xu2022eurnet,
  title={EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data},
  author={Xu, Minghao and Guo, Yuanfan and Xu, Yi and Tang, Jian and Chen, Xinlei and Tian, Yuandong},
  journal={arXiv preprint arXiv:2211.12941},
  year={2022}
}
```

## Acknowledgements

The development of this project is guided by the following open-source projects. 
We would like to thank the authors for releasing the source code.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [FocalNet](https://github.com/microsoft/FocalNet)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)
