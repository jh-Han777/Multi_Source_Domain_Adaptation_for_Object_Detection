# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np
from datasets.cityscape import cityscape
from datasets.cityscapes_car import cityscapes_car
from datasets.clipart import clipart
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.rpc import rpc
from datasets.rpc_fake import rpc_fake
from datasets.sim10k_coco import sim10k
from datasets.vg import vg
from datasets.water import water

from datasets.bdd100k_daytime import bdd100k_daytime
from datasets.bdd100k_night import bdd100k_night
from datasets.bdd100k_dd import bdd100k_dd

from datasets.cityscapes_ms_car import cityscapes_ms_car
from datasets.bdd100k_daytime_car import bdd100k_daytime_car
from datasets.KITTI_car import KITTI_car
# from datasets.bdd100k_daytime_car import BDD100k_daytime_car
# from datasets.KITTI_car import KITTI_car
# from datasets.cityscapes_KDA import cityscapes_KDA

__sets = {}


# Set up voc_<year>_<split>
for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc(split, year)

for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_water(split, year)

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)

for year in ["2007"]:
    for split in ["train", "test"]:
        name = "water_{}".format(split)
        __sets[name] = lambda split=split: water(split, year)

for year in ["2007"]:
    for split in ["val", "test"]:
        name = "rpc_{}".format(split)
        __sets[name] = lambda split=split: rpc(split, year)

for year in ["2007"]:
    for split in [
        "train",
    ]:
        name = "rpc_fake_{}".format(split)
        __sets[name] = lambda split=split: rpc_fake(split, year)

for year in ["2007", "2012"]:
    for split in ["train_s", "train_t", "train_all", "test_s", "test_t", "test_all"]:
        name = "cityscape_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape(split, year)

# Set up coco_2014_<split>
for year in ["2014"]:
    for split in ["train", "val", "minival", "valminusminival", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2014_cap_<split>
for year in ["2014"]:
    for split in ["train", "val", "capval", "valminuscapval", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2015_<split>
for year in ["2015"]:
    for split in ["test", "test-dev"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up sim10k coco style and cityscapes coco style
for year in ["2019"]:
    for split in ["train", "val"]:
        name = "sim10k_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: sim10k(split, year)
# Set up sim10k coco style and cityscapes coco style
for year in ["2019"]:
    for split in ["train", "val"]:
        name = "cityscapes_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscapes_car(split, year)

for split in ["train","val"]:
    name = "bdd100k_daytime_{}".format(split)
    __sets[name] = lambda split = split : bdd100k_daytime(split)

for split in ["train","val"]:
    name = "bdd100k_night_{}".format(split)
    __sets[name] = lambda split = split : bdd100k_night(split)

for split in ["train","val"]:
    name = "bdd100k_dd_{}".format(split)
    __sets[name] = lambda split = split : bdd100k_dd(split)

for split in ["train","val"]:
    name = "KITTI_car_{}".format(split)
    __sets[name] = lambda split=split: KITTI_car(split)

for split in ["train","val","test"]:
    name = "bdd100k_daytime_car_{}".format(split)
    __sets[name] = lambda split=split : bdd100k_daytime_car(split)

for split in ["train", "val"]:
    name = "cityscapes_ms_car_{}".format(split)
    __sets[name] = lambda split=split: cityscapes_ms_car(split)

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in [
    "150-50-20",
    "150-50-50",
    "500-150-80",
    "750-250-150",
    "1750-700-450",
    "1600-400-20",
]:
    for split in [
        "minitrain",
        "smalltrain",
        "train",
        "minival",
        "smallval",
        "val",
        "test",
    ]:
        name = "vg_{}_{}".format(version, split)
        __sets[name] = lambda split=split, version=version: vg(version, split)

# set up image net.
for split in ["train", "val", "val1", "val2", "test"]:
    name = "imagenet_{}".format(split)
    devkit_path = "data/imagenet/ILSVRC/devkit"
    data_path = "data/imagenet/ILSVRC"
    __sets[
        name
    ] = lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(
        split, devkit_path, data_path
    )


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
