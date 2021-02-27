import os
import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

def prepare_data(data_dir, scale=50, data_len=100):

    imgs, c_points = [], []
    for i in range(1,data_len+1):

        files_list = {'obstacles': os.path.join(data_dir, f'test_planar_{i}_obstacles.csv'),
                      'inspection_points': os.path.join(data_dir, f'test_planar_{i}_inspection_points.csv'),
                      'configurations': os.path.join(data_dir, f'test_planar_{i}_conf'),
                      'vertex': os.path.join(data_dir, f'test_planar_{i}_vertex'),
                      'results':os.path.join(data_dir, f'test_search_{i}_result')}

        # check if test files exist
        broken_files = False
        for file_path in files_list.values():
            if not os.path.isfile(file_path):
                broken_files = True
        
        if broken_files:
            continue

        # construct image with obstacles
        img = np.zeros((101,101))
        obstacles = (pd.read_csv(files_list['obstacles']) * scale).round(0).astype(int)
        obstacles.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)

        # load inspection points csv
        inspection_points = (pd.read_csv(files_list['inspection_points']) * scale).round(0).astype(int)

        # load configuration space csv
        cspace_df = pd.read_csv(files_list['configurations'], delimiter=' ', header=None).drop(columns=[0,6])

        # create a dictionary with inspection points indices for each vertex index
        inspection_points_per_vertex = {}
        with open(files_list['vertex'],'r') as f:
            for line in f:
                line = line.split(' ')[:-1]
                inspection_points_per_vertex[int(line[0])] = [int(x) for x in line[3:]]

        # drop already seen points from inspection points
        with open(files_list['results'],'r') as f:
            for line in f:
                vertices_idxs = [int(x) for x in line.split(' ')[1:-1]]
                for j in range(1,len(vertices_idxs)):
                    img_copy = img.copy()

                    # get union of inspected vertices indices
                    visited_idxs = vertices_idxs[:j]
                    visited_inspection_points = [inspection_points_per_vertex[x] for x in vertices_idxs]
                    visited_inspection_points = list(set().union(*visited_inspection_points))

                    # draw unseen inspection points on the image
                    inspection_points.drop(index=visited_inspection_points).apply(lambda x: cv2.rectangle(img_copy, (x['x'], x['y']), (x['x'], x['y']), 100, -1), axis=1)

                    imgs.append(np.expand_dims(img_copy, axis=0))
                    c_points.append(np.expand_dims(cspace_df.iloc[j].to_numpy(), axis=0))
        
        if i%10 == 0:
            print(f'Created images: {i}')

    # normilize data
    cs_np = np.concatenate(c_points) / np.pi
    imgs_np = np.concatenate(imgs).reshape([len(cs_np),-1]) / 255.0

    # shuffle images and c-space points in the same order
    # p = np.random.permutation(imgs_np.shape[0])
    # imgs_np = imgs_np[p]
    # cs_np = cs_np[p]

    # write h5 file to the same location
    with h5py.File(os.path.join(data_dir,"data.h5"), "w") as f:
        f.create_dataset("images", data=imgs_np, dtype=imgs_np.dtype)
        f.create_dataset("cpoints", data=cs_np, dtype=cs_np.dtype)


def load_data(data_file, num_imgs=None, batch_size=32, test_sample=False):

    # load h5 file
    dataset_dict = h5py.File(data_file, 'r')

    # get dataset as numpy
    imgs_np = np.asarray(dataset_dict['images'], dtype=np.float32)
    cs_np = np.asarray(dataset_dict['cpoints'], dtype=np.float32)

    # trim data if asked to
    if num_imgs is not None and imgs_np.shape[0] > num_imgs:
        imgs_np = imgs_np[:num_imgs,:]
        cs_np = cs_np[:num_imgs,:]

    if test_sample:
        return imgs_np[-1:,:], cs_np[-1:,:]

    # decide test split
    test_split = int(imgs_np.shape[0] * 0.3) # TODO: Note that the obstacles are already shuffled, but the trajectories are not!

    # add noise
    #imgs_np[:-test_split,:] = imgs_np[:-test_split,:] + np.random.normal(loc=imgs_np[:-test_split,:].mean(), scale=imgs_np[:-test_split,:].std()/10, size=imgs_np[:-test_split,:].shape)

    # convert to tf format dataset and prepare batches
    train_ds = tf.data.Dataset.from_tensor_slices((imgs_np[:-test_split,:], cs_np[:-test_split,:])).shuffle(len(imgs_np[:-test_split,:])).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((imgs_np[-test_split:,:], cs_np[-test_split:,:])).shuffle(len(imgs_np[-test_split:,:])).batch(batch_size)
    return train_ds, test_ds


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

    data_dir = 'learning/data10k'
    prepare_data(data_dir, scale=50, data_len=20000)
