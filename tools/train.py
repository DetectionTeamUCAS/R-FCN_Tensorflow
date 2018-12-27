# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np
import time
sys.path.append("../")
from libs.configs import cfgs
from libs.networks import build_whole_network
from data.io.read_tfrecord import next_batch
from libs.box_utils import show_box_in_tensor
from help_utils import tools


os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def train():
    rfcn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                is_training=True)

    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                       is_training=True)
        gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    biases_regularizer = tf.no_regularizer
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)

    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
        final_bbox, final_scores, final_category, loss_dict = rfcn.build_whole_detection_network(
            input_img_batch=img_batch,
            gtboxes_batch=gtboxes_and_label)

    # ----------------------------------------------------------------------------------------------------build loss
    weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
    # weight_decay_loss = tf.add_n(tf.losses.get_regularization_losses())
    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    rfcn_cls_loss = loss_dict['rfcn_cls_loss']
    rfcn_loc_loss = loss_dict['rfcn_loc_loss']
    rfcn_total_loss = rfcn_cls_loss + rfcn_loc_loss

    total_loss = rpn_total_loss + rfcn_total_loss + weight_decay_loss

    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('RFCN_LOSS/rfcn_cls_loss', rfcn_cls_loss)
    tf.summary.scalar('RFCN_LOSS/rfcn_location_loss', rfcn_loc_loss)
    tf.summary.scalar('RFCN_LOSS/rfcn_total_loss', rfcn_total_loss)

    tf.summary.scalar('LOSS/total_loss', total_loss)
    tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)

    gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_batch,
                                                                   boxes=gtboxes_and_label[:, :-1],
                                                                   labels=gtboxes_and_label[:, -1])
    if cfgs.ADD_BOX_IN_TENSORBOARD:
        detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                                                     boxes=final_bbox,
                                                                                     labels=final_category,
                                                                                     scores=final_scores)
        tf.summary.image('Compare/final_detection', detections_in_img)
    tf.summary.image('Compare/gtboxes', gtboxes_in_img)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

    # ---------------------------------------------------------------------------------------------compute gradients
    gradients = rfcn.get_gradients(optimizer, total_loss)

    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = rfcn.enlarge_gradients_for_bias(gradients)

    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients'):
            gradients = slim.learning.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)

    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = rfcn.get_restorer()
    saver = tf.train.Saver(max_to_keep=30)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        # tools.mkdir(summary_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        for step in range(cfgs.MAX_ITERATION):
            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                _, global_stepnp = sess.run([train_op, global_step])

            else:
                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                    start = time.time()

                    _, global_stepnp, img_name, rpnLocLoss, rpnClsLoss, rpnTotalLoss, \
                    rfcnLocLoss, rfcnClsLoss, rfcnTotalLoss, totalLoss = \
                        sess.run(
                            [train_op, global_step, img_name_batch, rpn_location_loss, rpn_cls_loss, rpn_total_loss,
                             rfcn_loc_loss, rfcn_cls_loss, rfcn_total_loss, total_loss])

                    end = time.time()
                    print(""" {}: step{}    image_name:{} |\t
                              rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t rpn_total_loss:{} |
                              rfcn_loc_loss:{} |\t rfcn_cla_loss:{} |\t rfcn_total_loss:{} |
                              total_loss:{} |\t per_cost_time:{}s""" \
                          .format(training_time, global_stepnp, str(img_name[0]), rpnLocLoss, rpnClsLoss,
                                  rpnTotalLoss, rfcnLocLoss, rfcnClsLoss, rfcnTotalLoss, totalLoss,
                                  (end - start)))
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                        summary_writer.add_summary(summary_str, global_stepnp)
                        summary_writer.flush()

            if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):

                save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_ckpt = os.path.join(save_dir, 'voc_' + str(global_stepnp) + 'model.ckpt')
                saver.save(sess, save_ckpt)
                print(' weights had been saved')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    train()

#
















