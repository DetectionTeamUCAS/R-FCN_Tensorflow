# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer
# from libs.lib_kernel.lib_psroi_pooling import psroi_pooling_op, psroi_pooling_op_grad
# from libs.lib_kernel.lib_psalign_pooling import psalign_pooling_op, psalign_pooling_op_grad
from libs.lib_kernel import psroi_warp


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sorry, we only support resnet or mobilenet_v2')

    def postprocess_rfcn(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_rfcn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.RFCN_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.RFCN_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def psroi_pooling(self, feature_maps, rois, crop_size=[9, 9], num_spatial_bins=[3, 3]):
        '''

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('PSROI_Pooling'):
            # [psroipooled_cls_rois, _] = psroi_pooling_op.psroi_pool(
            #     rfcn_cls, rois, group_size=group_size, spatial_scale=spatial_scale)
            # [psroipooled_loc_rois, _] = psroi_pooling_op.psroi_pool(
            #     rfcn_bbox, rois, group_size=group_size, spatial_scale=spatial_scale)

             # return psroipooled_cls_rois, psroipooled_loc_rois
            result = psroi_warp.psroi_warp(feature_maps=feature_maps,
                                           boxes=rois,
                                           crop_size=crop_size,
                                           num_spatial_bins=num_spatial_bins)

            return result

    def build_rfcn_head(self, rfcn_base, rois, img_shape, bin_nums=[3, 3], crop_size=[9, 9]):

        with tf.variable_scope('RFCN'):
            # 5. PSROI Pooling
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            conv_new_1 = slim.conv2d(rfcn_base, 256, [3, 3], trainable=self.is_training,
                                     weights_initializer=initializer, activation_fn=tf.nn.relu, scope='conv_new1')

            rfcn_cls = slim.conv2d(conv_new_1, bin_nums[0]*bin_nums[1]*(cfgs.CLASS_NUM+1), [1, 1],
                                   trainable=self.is_training, weights_initializer=initializer,
                                   activation_fn=None, scope='rfcn_cls')
            rfcn_bbox = slim.conv2d(conv_new_1, bin_nums[0]*bin_nums[1]*4, [1, 1], trainable=self.is_training,
                                    weights_initializer=initializer,
                                    activation_fn=None, scope='rfcn_bbox')

            # deal the boxes
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cls_score = self.psroi_pooling(feature_maps=rfcn_cls, rois=normalized_rois, crop_size=crop_size,
                                           num_spatial_bins=bin_nums)

            bbox_pred = self.psroi_pooling(feature_maps=rfcn_bbox, rois=normalized_rois, crop_size=crop_size,
                                           num_spatial_bins=bin_nums)
            cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])

            bbox_pred = tf.reshape(bbox_pred, [-1, 4])

            # psroipooled_cls_rois, psroipooled_loc_rois = self.psroi_pooling(rfcn_cls, rfcn_bbox, rois, group_size=7,
            #                                                                 spatial_scale=1.0/cfgs.ANCHOR_STRIDE[0])
            # cls_score = tf.reduce_mean(psroipooled_cls_rois, axis=[1, 2])
            # bbox_pred = tf.reduce_mean(psroipooled_loc_rois, axis=[1, 2])
            bbox_pred = tf.tile(bbox_pred, [1, (cfgs.CLASS_NUM+1)])

            # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])
            bbox_pred = tf.reshape(bbox_pred, [-1, (cfgs.CLASS_NUM+1)*4])

        return bbox_pred, cls_score

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)


        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                               boxes=neg_roi)
        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, cls_score, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('RFCN_loss'):
                if not cfgs.RFCN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.RFCN_SIGMA)

                    # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    # labels = tf.reshape(labels, [-1])
                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score,
                        labels=labels))  # beacause already sample before
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                               label=labels,
                                                               bbox_targets=bbox_targets,
                                                               bbox_pred=bbox_pred,
                                                               num_ohem_samples=128,
                                                               num_classes=cfgs.CLASS_NUM + 1)
                cls_loss = cls_loss * cfgs.RFCN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss = bbox_loss * cfgs.RFCN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'rfcn_cls_loss': cls_loss,
                'rfcn_loc_loss': bbox_loss
            }
        return loss_dict

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):
        
        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1.0 build base network
        rpn_base = self.build_base_network(input_img_batch)

        # 1.1 build the head_base network
        if self.base_network_name.startswith('resnet_v1'):
            rfcn_base = resnet.restnet_head(rpn_base, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            rfcn_base = mobilenet_v2.mobilenetv2_head(rpn_base, is_training=self.is_training)

        else:
            raise ValueError('Sorry, we only support resnet or mobilenet_v2')

        # 2. build rpn head
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

            rpn_conv3x3 = slim.conv2d(rpn_base, 128, [3, 3],
                                      trainable=self.is_training,
                                      weights_initializer=cfgs.INITIALIZER,
                                      activation_fn=tf.nn.relu,
                                      scope='rpn_conv/3x3')
            rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*2, [1, 1],
                                        stride=1,
                                        trainable=self.is_training,
                                        weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1],
                                       stride=1,
                                       trainable=self.is_training,
                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None,
                                       scope='rpn_bbox_pred')
            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        # 3. generate_anchors
        featuremap_height, featuremap_width = tf.shape(rpn_base)[1], tf.shape(rpn_base)[2]
        featuremap_height = tf.cast(featuremap_height, tf.float32)
        featuremap_width = tf.cast(featuremap_width, tf.float32)

        anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                            anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=cfgs.ANCHOR_STRIDE,
                                            name="make_anchors_forRPN")

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_RPN'):
            rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                         rpn_cls_prob=rpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=anchors,
                                                         is_training=self.is_training)
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                        boxes=rois,
                                                                        scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                                boxes=score_gre_05_rois,
                                                                                scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                rpn_labels, rpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [gtboxes_batch, img_shape, anchors],
                        [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels)

            # --------------------------------------add smry----------------------------------------------------------

            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope('sample_RFCN_minibatch'):
                    rois, labels, bbox_targets = \
                    tf.py_func(proposal_target_layer,
                               [rois, gtboxes_batch],
                               [tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 4*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels)

        # -------------------------------------------------------------------------------------------------------------#
        #                                                          RFCN                                                #
        # -------------------------------------------------------------------------------------------------------------#
        # 5. build rfcn head
        bbox_pred, cls_score = self.build_rfcn_head(rfcn_base=rfcn_base,
                                                    rois=rois,
                                                    img_shape=img_shape,
                                                    bin_nums=[3, 3],
                                                    crop_size=[9, 9])  # crop_size is the total size
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        cls_prob = slim.softmax(cls_score, 'cls_prob')

        # ----------------------------------------------add smry-------------------------------------------------------
        if self.is_training:
            cls_category = tf.argmax(cls_prob, axis=1)
            rfcn_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))
            tf.summary.scalar('ACC/rfcn_acc', rfcn_acc)

        #  6. postprocess_rfcn
        if not self.is_training:
            return self.postprocess_rfcn(rois=rois, bbox_ppred=bbox_pred, scores=cls_prob, img_shape=img_shape)
        else:
            '''
            when trian. We need build Loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred=bbox_pred,
                                        bbox_targets=bbox_targets,
                                        cls_score=cls_score,
                                        labels=labels)

            final_bbox, final_scores, final_category = self.postprocess_rfcn(rois=rois,
                                                                             bbox_ppred=bbox_pred,
                                                                             scores=cls_prob,
                                                                             img_shape=img_shape)
            return final_bbox, final_scores, final_category, loss_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)

        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            def name_in_ckpt_rpn(var):
                return var.op.name

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith(self.base_network_name):
                    var_name_in_ckpt = name_in_ckpt_rpn(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    continue

            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















