# -*- coding: utf-8 -*-

print("################################################################")
print(
    "# The function of this python code is the same as that of matlab code in https://github.com/shenyunhang/cocoapi/blob/master/MatlabAPI/get_ImageNetCls_GT.m"
)
print("################################################################")

import json
import os
import xml.etree.ElementTree as ET
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

from detectron2.utils.file_io import PathManager

root_dir = "datasets/ILSVRC/"
anno_dir = "datasets/ILSVRC/Annotations/CLS-LOC"
data_dir = "datasets/ILSVRC/Data/CLS-LOC"
imageset_dir = "datasets/ILSVRC/ImageSets/CLS-LOC"
meta_file = "datasets/ILSVRC/devkit/data/meta_clsloc.mat"


folders = []
categories = [
    # {"supercategory": "none", "id": 1, "name": "aeroplane"},
    # {"supercategory": "none", "id": 2, "name": "bicycle"},
    # {"supercategory": "none", "id": 3, "name": "bird"},
    # {"supercategory": "none", "id": 4, "name": "boat"},
    # {"supercategory": "none", "id": 5, "name": "bottle"},
    # {"supercategory": "none", "id": 6, "name": "bus"},
    # {"supercategory": "none", "id": 7, "name": "car"},
    # {"supercategory": "none", "id": 8, "name": "cat"},
    # {"supercategory": "none", "id": 9, "name": "chair"},
    # {"supercategory": "none", "id": 10, "name": "cow"},
    # {"supercategory": "none", "id": 11, "name": "diningtable"},
    # {"supercategory": "none", "id": 12, "name": "dog"},
    # {"supercategory": "none", "id": 13, "name": "horse"},
    # {"supercategory": "none", "id": 14, "name": "motorbike"},
    # {"supercategory": "none", "id": 15, "name": "person"},
    # {"supercategory": "none", "id": 16, "name": "pottedplant"},
    # {"supercategory": "none", "id": 17, "name": "sheep"},
    # {"supercategory": "none", "id": 18, "name": "sofa"},
    # {"supercategory": "none", "id": 19, "name": "train"},
    # {"supercategory": "none", "id": 20, "name": "tvmonitor"},
]


def read_txt_train(path, split="train_cls"):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        lines = f.readlines()

    pathes = []
    idxs = []
    folder_names = []

    for line in lines:
        line = line.strip()
        path, idx = line.split()
        folder_name = path.split("/")[-1].split("_")[0]

        pathes.append(path)
        idxs.append(idx)
        folder_names.append(folder_name)

    return pathes, idxs, folder_names


def read_txt_val(path, split="val"):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        lines = f.readlines()

    pathes = []
    idxs = []

    for line in lines:
        line = line.strip()
        path, idx = line.split()

        pathes.append(path)
        idxs.append(idx)

    return pathes, idxs


def generate_categories_list():
    meta = sio.loadmat(meta_file)
    meta = meta["synsets"][0]

    for m in meta:
        ids = int(m[0][0][0])
        folder = m[1][0]
        name = str(m[2][0])
        # print(m, ids, folder, name)
        category = {"supercategory": "none", "id": ids, "name": name}
        categories.append(category)
        folders.append(folder)


def generate_image_anno(anno_file, jpeg_file, idx, folder_name, images_info, annotations):

    if not os.path.isfile(anno_file):
        with Image.open(jpeg_file) as img:
            width, height = img.size
        info = {"file_name": jpeg_file, "id": idx, "height": height, "width": width}

        images_info.append(info)

        bbox = [0, 0, width, height]

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = int(w * h)

        anno = {
            "area": area,
            "image_id": idx,
            "bbox": [int(bbox[0]), int(bbox[1]), int(w), int(h)],
            "iscrowd": 0,
            "category_id": folders.index(folder_name) + 1,
            "id": len(annotations) + 1,
            "difficult": 0,
            "truncated": 0,
            "ignore": 0,
        }

        annotations.append(anno)

        return images_info, annotations

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    info = {
        "file_name": jpeg_file,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
        "id": idx,
    }

    images_info.append(info)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        # We include "difficult" samples in training.
        # Based on limited experiments, they don't hurt accuracy.
        difficult = int(obj.find("difficult").text)
        truncated = int(obj.find("truncated").text)
        if difficult == 1:
            continue
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        bbox[0] -= 1.0
        bbox[1] -= 1.0

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = int(w * h)

        anno = {
            "area": area,
            "image_id": idx,
            "bbox": [int(bbox[0]), int(bbox[1]), int(w), int(h)],
            "iscrowd": 0,
            "category_id": folders.index(cls) + 1,
            "id": len(annotations) + 1,
            "difficult": difficult,
            "truncated": truncated,
            "ignore": difficult,
        }

        annotations.append(anno)

    return images_info, annotations


def save_json(ann, path, name):
    os.system("mkdir -p {}".format(path))
    instance_path = os.path.join(path, name)
    with open(instance_path, "w") as f:
        json.dump(ann, f)


def convert_to_json(pathes, idxs, folder_names, split, json_name):
    images_info = []
    annotations = []
    for i in tqdm(range(len(idxs))):
        img_path = os.path.join(data_dir, split, pathes[i] + ".JPEG")
        anno_path = os.path.join(anno_dir, split, pathes[i] + ".xml")
        assert os.path.isfile(img_path), img_path

        if not folder_names:
            folder_name = None
            assert os.path.isfile(anno_path), anno_path
        else:
            folder_name = folder_names[i]

        images_info, annotations = generate_image_anno(
            anno_path, img_path, idxs[i], folder_name, images_info, annotations
        )

    ilsvrc_instance = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories,
    }
    # print(ilsvrc_instance)
    save_json(ilsvrc_instance, root_dir, json_name)


def convert_ilsvrc():
    generate_categories_list()
    # print(folders)
    # print(categories)

    pathes, idxs, folder_names = read_txt_train(imageset_dir, "train_cls")
    convert_to_json(pathes, idxs, folder_names, "train", "ilsvrc_train.json")

    pathes, idxs = read_txt_val(imageset_dir, "val")
    convert_to_json(pathes, idxs, None, "val", "ilsvrc_val.json")


if __name__ == "__main__":
    convert_ilsvrc()
