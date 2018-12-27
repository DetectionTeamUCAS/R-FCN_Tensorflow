# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf


# ------------------------------------------------
VERSION = 'RFCN_20180708_pascal'
NET_NAME = 'resnet_v1_101'  # 'resnet_v1_101, resnet_v1_50, MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/' + VERSION

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 2.0
RPN_CLASSIFICATION_LOSS_WEIGHT = 2.0

RFCN_LOCATION_LOSS_WEIGHT = 2.0
RFCN_CLASSIFICATION_LOSS_WEIGHT = 2.0
RPN_SIGMA = 3.0
RFCN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001  # 0.001  # 0.0003
DECAY_STEP = [40000, 80000]  # 50000, 70000
MAX_ITERATION = 120000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000
CLASS_NUM = 20

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.0000 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None

# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 300  # 300


# -------------------------------------------RFCN config
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard

RFCN_NMS_IOU_THRESHOLD = 0.3  # 0.6
RFCN_NMS_MAX_BOXES_PER_CLASS = 100
RFCN_IOU_POSITIVE_THRESHOLD = 0.5
RFCN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
RFCN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
RFCN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False



