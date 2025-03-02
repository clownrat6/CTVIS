# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

from .ytvis import load_ytvis_json, _get_ytvis_2019_instances_meta, register_ytvis_instances


# ==== Predefined splits for HQYTVIS ===========
_PREDEFINED_SPLITS_HQYTVIS = {
    "hqytvis_train": ("ytvis_2019/train/JPEGImages",
                      "hqytvis/ytvis_hq-train.json"),
    "hqytvis_val":   ("ytvis_2019/train/JPEGImages",
                      "hqytvis/ytvis_hq-val.json"),
    "hqytvis_test":  ("ytvis_2019/train/JPEGImages",
                      "hqytvis/ytvis_hq-test.json"),
}


def register_all_hqytvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_HQYTVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./media`.
_root = os.getenv("DETECTRON2_DATASETS", "./datasets")
register_all_hqytvis(_root)
