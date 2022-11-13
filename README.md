# EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data

This repository provides the PyTorch implementation of the paper EurNet: Efficient Multi-Range Relational Modeling of Spatial Multi-Relational Data.

## Installation
This repository is officially tested with the following environments:
- Linux
- Python 3.6+
- PyTorch 1.10.0
- CUDA 11.3

The environment could be prepared in the following steps:
1. Create a virtual environment with conda:
```
conda create -n eurnet python=3.7.3 -y
conda activate eurnet
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

#### ImageNet-22K

We also support [ImageNet22K](http://www.image-net.org/) for pre-training. We recommend symlink the dataset folder to `./datasets/imagenet22k/`. Move all images to the labeled sub-folders in the data path. Then, download the train-val split files ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt)) and move to the data path.
The folder structure should be:
```bash
datasets/
  imagenet22k/
    ILSVRC2011fall_whole_map_train.txt
    ILSVRC2011fall_whole_map_val.txt
    n00004475/
    n00005787
    ...
```

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
output_dir=./experiments/imagenet1k/EurNet-T/
```

**Note that**, all the training configuration files are in `./configs/classification/`. To reproduce the ImageNet classification results, please follow the corresponding config file. 

## Multinode training
`launch.py` would automatically find a free port to launch single node experiments. If multiple nodes training is necessary, the number of nodes `--nn`, node rank `--nr`, master port `--port` and master address `-ma` should be set. 

Take training EurNet-T on ImageNet-1K with two nodes as an example, use the following commands at node1 and node2, respectively.
```
# use this command at node 1
python3 launch.py --nn 2 --nr 0 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/classification/eurnet_tiny_1k_300eps.yaml

# use this command at node 2
python3 launch.py --nn 2 --nr 1 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/classification/eurnet_tiny_1k_300eps.yaml
```
