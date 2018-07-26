#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:10:39 2018

@author: nzhang
"""

import tensorflow as tf
import os

SAVE_PATH = os.path.join("/Users/nzhang/Rebate/runs/1532276166", "checkpoints")
checkpoint_file = tf.train.latest_checkpoint(SAVE_PATH)

path = os.path.join(os.getcwd(),'output')
builder = tf.saved_model.builder.SavedModelBuilder(path)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        
        
        input_x = graph.get_tensor_by_name("input_x:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

        # Tensors we want to evaluate
        predictions = graph.get_tensor_by_name("output/predictions:0")
        
               
        classification_inputs = tf.saved_model.utils.build_tensor_info(input_x)
        classification_dropout_keep_prob = tf.saved_model.utils.build_tensor_info(dropout_keep_prob)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(predictions)
         
        classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  'inputs':
                      classification_inputs,
                  'droput':
                  classification_dropout_keep_prob
              },     
              outputs={
                  'Predictions':
                      classification_outputs_classes
              },
              method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
               
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            classification_signature})

        builder.save()