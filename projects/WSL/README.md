# Parallel Detection-and-Segmentation Learning for Weakly Supervised Instance Segmentation

By [Yunhang Shen](), [Liujuan Cao](), [Zhiwei Chen](), [Baochang Zhang](), [Chi Su](), [Yongjian Wu](), [Feiyue Huang](), [Rongrong Ji]().

ICCV 2021 Paper.

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).


## License

PDSL is released under the [Apache 2.0 license](LICENSE).


## Installation

Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install PDSL project:
```
cd projects/WSL
pip3 install -r requirements.txt
git submodule update --init --recursive
python3 -m pip install -e .
```


## Dataset Preparation

#### PASCAL VOC 2012:
Please follow [this](https://github.com/shenyunhang/PDSL/blob/PDSL/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.

Also download SBD data:
```
cd datasets/
wget --no-check-certificate http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar xvzf benchmark.tgz
ln -s benchmark_RELEASE/dataset/ SBD
```

Convert VOC 2012 and SBD to our format:
```
python3 projects/WSL/tools/convert_voc2012_and_sbd_instance.py
```

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-Pascal-Segmentation_trainvaltest_2012-boxes.tgz
tar xvzf MCG-SBD-trainval-boxes.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py voc_2012_train_instance datasets/proposals/MCG-Pascal-Segmentation_trainvaltest_2012-boxes datasets/proposals/mcg_voc_2012_train_instance_d2.pkl
python3 projects/WSL/tools/proposal_convert.py voc_2012_val_instance datasets/proposals/MCG-Pascal-Segmentation_trainvaltest_2012-boxes datasets/proposals/mcg_voc_2012_val_instance_d2.pkl
python3 projects/WSL/tools/proposal_convert.py sbd_9118_instance datasets/proposals/MCG-SBD-trainval-boxes datasets/proposals/mcg_sbd_9118_instance_d2.pkl
```

#### COCO:
Please follow [this](https://github.com/shenyunhang/PDSL/blob/PDSL/datasets/README.md#expected-dataset-structure-for-coco-instancekeypoint-detection) to creating symlinks for MS COCO.

Download
```
wget https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
```

Please follow [this](https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/data/README.md#coco-minival-annotations) to download `minival` and `valminusminival` annotations.

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-COCO-train2014-boxes.tgz
tar xvzf MCG-COCO-val2014-boxes.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py coco_2014_train datasets/proposals/MCG-COCO-train2014-boxes datasets/proposals/mcg_coco_2014_train_d2.pkl
python3 projects/WSL/tools/proposal_convert.py coco_2014_valminusminival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/mcg_coco_2014_valminusminival_d2.pkl
python3 projects/WSL/tools/proposal_convert.py coco_2014_minival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/mcg_coco_2014_minival_d2.pkl
```


## Model Preparation

Download models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models $PDSL
```

Then we have the following directory structure:
```
PDSL
|_ models
|  |_ DRN-WSOD
|     |_ resnet18_ws_model_120.pkl
|     |_ resnet150_ws_model_120.pkl
|     |_ resnet101_ws_model_120.pkl
|_ ...
```


## Quick Start: Using PDSL

### PASCAL VOC

ResNet18-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-InstanceSegmentation/pdsl_WSR_18_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_18_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```

ResNet50-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-InstanceSegmentation/pdsl_WSR_50_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_50_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```

ResNet101-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-InstanceSegmentation/pdsl_WSR_101_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_101_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```


### MS COCO

ResNet18-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/COCO-InstanceSegmentation/pdsl_WSR_18_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_18_DC5_coco_`date +'%Y-%m-%d_%H-%M-%S'`
```

ResNet50-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/COCO-InstanceSegmentation/pdsl_WSR_50_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_50_DC5_coco_`date +'%Y-%m-%d_%H-%M-%S'`
```

ResNet101-WS
```
python3.9 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/COCO-InstanceSegmentation/pdsl_WSR_101_DC5_1x.yaml OUTPUT_DIR output/pdsl_WSR_101_DC5_coco_`date +'%Y-%m-%d_%H-%M-%S'`
```

### Results
We proivde [two log files](https://1drv.ms/u/s!Am1oWgo9554dhuRVV1X-o7qwepuK3A?e=ONv0Kl).

The old one (pdsl_WSR_18_DC5_VOC_SBD_2020-10-05_18-42-57) is trained on 4 GPUs with detectron:0.2

The new one (pdsl_WSR_18_DC5_voc2012_sbd_2021-10-10_01-02-28) is trained on 8 GPUs with detectron:0.5


## Citing PDSL

If you find PDSL useful in your research, please consider citing:

```
@InProceedings{PDSL_2021_CVPR,
	author = {Shen, Yunhang and Cao, Liujuan and Chen, Zhiwei and Zhang, Baochang and Su, Chi and Wu, Yongjian and Huang, Feiyue and Ji, Rongrong},
	title = {Parallel Detection-and-Segmentation Learning for Weakly Supervised Instance Segmentation},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	year = {2021},
	pages = {8198-8208}
}   
```
