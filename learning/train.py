from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from models import BaseModel
from load import DataManagement

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data_dir', help='path h5 data dir', default='build/data10k_v2', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='learning/test_basemodel', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--num_imgs', '-num_imgs', help='number of images to train on', default=1000000, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=20, type=int)
parser.add_argument('--cp_interval', '-cp_interval', help='interval for checkpoint saving', default=2, type=int)
parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
args = parser.parse_args()

@tf.function
def compute_loss(y, y_pred):

    # copute reconstruction loss
    recon_loss = tf.reduce_mean(tf.keras.losses.MSE(y, y_pred))

    return recon_loss

# tf function to train
@tf.function
def train(images, c_points):
    with tf.GradientTape() as tape:
        # add noise
        #c_points = c_points + tf.random.normal(shape=c_points.shape, mean=0.0, stddev=0.1)

        # inference
        predictions = model(images) #images.shape=[,10201]

        # compute loss
        loss = compute_loss(c_points, predictions)

    # compute step
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_rec_loss(loss)

# tf function to test
@tf.function
def test(images, c_points):
    # inference
    predictions = model(images) #images.shape=[,10201]

    # compute loss
    loss = compute_loss(c_points, predictions)

    test_rec_loss(loss)

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# get train and test datasets
date_mng = DataManagement(data_dir=args.data_dir, num_imgs=args.num_imgs, batch_size=args.batch_size)

# create model, loss and optimizer
model = BaseModel()
optimizer = tf.keras.optimizers.Adam()

# define metrics
train_rec_loss = tf.keras.metrics.Mean(name='train_rec_loss')
test_rec_loss = tf.keras.metrics.Mean(name='test_rec_loss')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# train
print('Start training...')
for epoch in range(args.epochs):

    for images, c_points in date_mng.train_gen:
        train(images, c_points)

    for test_images, test_c_points in date_mng.test_gen:
        test(test_images, test_c_points)
    
    with metrics_writer.as_default():
        tf.summary.scalar('Train reconstruction loss', train_rec_loss.result(), step=epoch)
        tf.summary.scalar('Test reconstruction loss', test_rec_loss.result(), step=epoch)

    print('Epoch {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_rec_loss.result(), test_rec_loss.result()))
    
    # save model
    if (epoch+1) % args.cp_interval == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "basemodel{}.ckpt".format(epoch+1)))
        #model.save(os.path.join(args.output_dir, 'model_saved'))