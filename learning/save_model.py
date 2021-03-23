from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from models import VAEModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-model_dir', help='path to model file', default='learning/test_v2_nonoise', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# create model, loss and optimizer
model = VAEModel(n_z=args.n_z, freeze=True)
model.load_weights(os.path.join(args.model_dir, "vaemodel20.ckpt"))

# inference test
y = model(tf.random.normal([1,10209]))

# save model
tf.keras.models.save_model(model, os.path.join(args.model_dir, 'model_saved'))