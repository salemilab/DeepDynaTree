#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
__init__.py.py: 
"""
import os.path as osp
import json
from dl.config import get_arguments
import logging
from pathlib import Path


def get_logger(args):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s(%(lineno)d): %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def get_feat_dict():
    # load the feature dictionary
    script_dir = osp.realpath(__file__)
    js_f = osp.normpath(osp.join(script_dir, "../../aly/feat_dict.json"))
    with open(js_f, 'r') as infile:
        feat_dict = json.load(infile)
    return feat_dict


args = get_arguments()
logger = get_logger(args)
feat_dict = get_feat_dict()