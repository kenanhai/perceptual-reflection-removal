from __future__ import division
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from discriminator import build_discriminator
import scipy.stats as st
import argparse

os.environ['CUDA_VISIBLE_DEVICES']=str(0)
#input=tf.placeholder(tf.float32,shape=[None,None,None,3])#input

with tf.Session() as sess:
    saver=tf.train.import_meta_graph("./pre-trained/model.ckpt.meta")
    saver.restore(sess, "./pre-trained/model.ckpt")
    img=cv2.imread("./test_images/CEILNet/qingnan-new2-1-input.jpg")
    input_image=np.expand_dims(np.float32(img), axis=0)/255.0
    #print(input_image)
    network = tf.get_default_graph().get_tensor_by_name("g_conv_last/BiasAdd:0")
    print(network)
    transmission_layer, reflection_layer=tf.split(network, num_or_size_splits=2, axis=3)
    output_image_t, output_image_r=sess.run([transmission_layer, reflection_layer],feed_dict={"Placeholder:0":input_image})
    print(output_image_r)
    print(output_image_t.shape)
    print(output_image_r.shape)
    output_image_t=np.minimum(np.maximum(output_image_t,0.0),1.0)*255.0
    output_image_r=np.minimum(np.maximum(output_image_r,0.0),1.0)*255.0
    cv2.imwrite("t_output.png",np.uint8(output_image_t[0,:,:,0:3])) # output transmission layier
    cv2.imwrite("r_output.png",np.uint8(output_image_r[0,:,:,0:3])) # output reflection layer
