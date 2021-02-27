from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import cv2
import h5py
import argparse
import tensorflow as tf
from models import VAEModel
from load import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', '-data_file', help='path h5 data file', default='learning/data10k/data.h5', type=str)
parser.add_argument('--model_path', '-model_path', help='path to model', default='learning/test_h5/vaemodel100.ckpt', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=1, type=int)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to train on', default=None, type=int)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='1', type=str)
args = parser.parse_args()

# tf function to test
@tf.function
def predict(images, c_points):
    predictions, _, _, _ = model(tf.concat([c_points, images], axis=-1))
    return predictions

# allow growth is possible using an env var in tf2.0
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# get train and test datasets
sample_img, sample_c = load_data(data_file=args.data_file, num_imgs=args.num_imgs, batch_size=args.batch_size, test_sample=True)

# create model, loss and optimizer
model = VAEModel(n_z=args.n_z)
model.load_weights(args.model_path)

# evaluate prediction
prediction = predict(tf.convert_to_tensor(np.ones((10201))), tf.convert_to_tensor(np.ones((10201))))
print(f"input c point: {sample_c[0]}\n predicted c point: {prediction.numpy()[0]}")
print(f"Dist between the point and the predicted {np.linalg.norm(sample_c[0]-prediction.numpy()[0])}")
