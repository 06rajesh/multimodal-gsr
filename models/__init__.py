# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

from .mgsrtr import build
from .gsrtr import build as build_gsrtr
from .types import ModelType

def build_model(args):
    if args.model_type == ModelType.GSRTR:
        return build_gsrtr(args)
    return build(args)
