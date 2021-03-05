from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import cv2
import h5py
import argparse
import tensorflow as tf
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-model_path', help='path to model', default='learning/results/test_h5/model_saved', type=str)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='1', type=str)
args = parser.parse_args()

# allow growth is possible using an env var in tf2.0
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# load frozen model
loaded_model = tf.keras.models.load_model(args.model_path)

# load sample image and ones as c-space point
img = np.expand_dims(cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE).reshape(-1),axis=0) / 255.0
#c_point = np.expand_dims(np.ones((5)), axis=0)
z_point = np.expand_dims(np.ones((8)), axis=0)
inputs = np.concatenate([z_point, img], axis=-1)

# print output
y = loaded_model(tf.convert_to_tensor(inputs, dtype=tf.float32))
print(y[0].numpy())