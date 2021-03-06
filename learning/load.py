import os
import glob
import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from shapely.geometry import Polygon 
import visilibity as vis

def prepare_data(data_dir, dest_name, start_idx=0, scale=50, end_idx=100000, limit_samples=1e6, eps=1e-7):
    max_wall = 200
    fov = np.pi / 2
    imgs, c_points = [], []
    counter_samples = 0
    i = start_idx
    while (i < end_idx + 1 and limit_samples > 0 and counter_samples < limit_samples):
        i += 1
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
        #obstacles.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)
        obstacles_columns = obstacles.columns.to_list()
        obstacles['p1'] = obstacles[obstacles_columns].astype(float).apply(lambda x: vis.Point(x['x'],x['y']),axis=1)
        obstacles['p2'] = obstacles[obstacles_columns].astype(float).apply(lambda x: vis.Point(x['x']+x['width'],x['y']),axis=1)
        obstacles['p3'] = obstacles[obstacles_columns].astype(float).apply(lambda x: vis.Point(x['x']+x['width'],x['y']+x['length']),axis=1)
        obstacles['p4'] = obstacles[obstacles_columns].astype(float).apply(lambda x: vis.Point(x['x'],x['y']+x['length']),axis=1)
        obstacles['polygon'] = obstacles.apply(lambda x: vis.Polygon([x['p1'], x['p2'], x['p3'], x['p4']]) ,axis=1)

        # define bounded environment with obstacles
        p1 = vis.Point(0,0)
        p2 = vis.Point(101,0)
        p3 = vis.Point(101,101)
        p4 = vis.Point(0,101)
        walls_poly = vis.Polygon([p1, p2, p3, p4])
        vis_env = vis.Environment([walls_poly] + obstacles['polygon'].to_list())

        # load inspection points csv
        inspection_points = (pd.read_csv(files_list['inspection_points']) * scale).round(0).astype(int)

        # load configuration space csv
        cspace_df = pd.read_csv(files_list['configurations'], delimiter=' ', header=None).drop(columns=[0,6])

        # drop already seen points from inspection points
        with open(files_list['results'],'r') as f:
            for line in f:
                vertices_idxs = [int(x) for x in line.split(' ')[1:-1]]
                img_copy = img.copy()
                for j in range(0,len(vertices_idxs)-1):
                    

                    if (np.random.uniform() < 0.0):
                        c_point = np.random.uniform(low=-np.pi, high=np.pi, size=5)
                        if j > 0:
                            j -= 1
                    else:
                        c_point = cspace_df.iloc[j].to_numpy()

                    # compute end-point for vertex
                    links_val = np.rint(compute_links(c_point) * scale).astype(int)
                    ee_val = links_val[-1]

                    # get orientation of end effector
                    ee_orientation = c_point.sum()

                    # set visibility triangle
                    x1 = ee_val[0] + max_wall * np.cos(ee_orientation + 0.5 * fov)
                    y1 = ee_val[1] + max_wall * np.sin(ee_orientation + 0.5 * fov)
                    x2 = ee_val[0] + max_wall * np.cos(ee_orientation - 0.5 * fov)
                    y2 = ee_val[1] + max_wall * np.sin(ee_orientation - 0.5 * fov)
                    vis_tri = Polygon([tuple(ee_val), (x1, y1), (x2,y2)])

                    # define observer
                    if is_in_bounds(ee_val):
                        observer = vis.Point(float(ee_val[0]), float(ee_val[1]))
                        observer.snap_to_boundary_of(vis_env, eps)
                        observer.snap_to_vertices_of(vis_env, eps)
                        isovist = vis.Visibility_Polygon(observer, vis_env, eps)

                        # get environment in points
                        point_x , point_y  = save_print(isovist)
                        if len(point_x ) == 0 or len(point_y) == 0:
                            continue
                        point_x.append(isovist[0].x())
                        point_y.append(isovist[0].y())
                        poly = Polygon([(x,y) for (x,y) in zip(point_x, point_y)])
                        visilbe_poly = poly.intersection(vis_tri)
                        if type(visilbe_poly) == Polygon and len(list(visilbe_poly.exterior.coords)) > 0:
                            visilbe_poly_pts = np.array(list(visilbe_poly.exterior.coords)).reshape((-1,1,2)).astype(int)

                            # draw visilbe polygon of the observer
                            cv2.fillPoly(img_copy, [visilbe_poly_pts], 150)

                            # draw obstacles and inspection points
                            obstacles.apply(lambda x: cv2.rectangle(img_copy, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)
                            inspection_points.apply(lambda x: cv2.rectangle(img_copy, (x['x'], x['y']), (x['x'], x['y']), 100, -1), axis=1)

                            # add sample (x,y)
                            imgs.append(np.expand_dims(img_copy, axis=0))
                            c_points.append(np.expand_dims(cspace_df.iloc[j+1].to_numpy(), axis=0))
                print(f'processed images: {counter_samples}')
                counter_samples += 1


        if i%100 == 0:
            print(f'Processed solutions: {i}')

    # normilize data
    cs_np = np.concatenate(c_points) / np.pi
    imgs_np = np.concatenate(imgs).reshape([len(cs_np),-1]) / 255.0

    # write h5 file to the same location
    with h5py.File(os.path.join(data_dir, dest_name), "w") as f:
        f.create_dataset("images", data=imgs_np, dtype=imgs_np.dtype)
        f.create_dataset("cpoints", data=cs_np, dtype=cs_np.dtype)


def compute_links(angles):

    origin = [1.0, 0.0]
    link_lengths = [0.2, 0.1, 0.2, 0.3, 0.1]
    link_positions = [origin]

    end_point = origin.copy()
    for i in range(len(angles)):
        end_point[0] += link_lengths[i] * np.cos(angles[i])
        end_point[1] += link_lengths[i] * np.sin(angles[i])
        link_positions.append(end_point.copy())

    return np.array(link_positions)

def is_in_bounds(ee_val, limit=101):
    if (ee_val[0] > 0 and ee_val[0] < limit and 
        ee_val[1] > 0 and ee_val[1] < limit):
        return True
    return False

def save_print(polygon):

    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()
        
        end_pos_x.append(x)
        end_pos_y.append(y)
                        
    return end_pos_x, end_pos_y 


class DataManagement(object):

    def __init__(self, data_dir, num_files=7, num_imgs=None, batch_size=32, test_sample=False, scale=50, noise_thresh=50):
        super(DataManagement, self).__init__()

        self.noise_thresh = noise_thresh

        # load h5 files
        imgs_np_list, cs_np_list = [], []
        for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "data_10k_v2_[!noise]*.h5"))):

            # get h5 file as numpy
            dataset_dict = h5py.File(data_file, 'r')
            imgs_np_list.append(np.asarray(dataset_dict['images'], dtype=np.float32))
            cs_np_list.append(np.asarray(dataset_dict['cpoints'], dtype=np.float32))

            if i == num_files - 1:
                break

        # load inspection points csv
        self.inspection_points = pd.read_csv(os.path.join(data_dir, f'test_planar_1_inspection_points.csv'))
        self.inspection_points = tf.convert_to_tensor(self.inspection_points.to_numpy())

        # concat all h5 files
        imgs_np = np.concatenate(imgs_np_list, axis=0)
        cs_np = np.concatenate(cs_np_list, axis=0)

        # free memory
        imgs_np_list = None
        cs_np_list = None

        # trim data if asked to
        if num_imgs is not None and imgs_np.shape[0] > num_imgs:
            imgs_np = imgs_np[:num_imgs,:]
            cs_np = cs_np[:num_imgs,:]

        if test_sample:
            return imgs_np[-1:,:], cs_np[-1:,:]

        # shuffle images and labels in the same order
        p = np.random.permutation(imgs_np.shape[0])
        imgs_np = imgs_np[p]
        cs_np = cs_np[p]

        # decide test split
        test_split = int(imgs_np.shape[0] * 0.3) 

        # split train and test
        self.imgs_np_train = imgs_np[:-test_split,:]
        self.cs_np_train = cs_np[:-test_split,:]
        self.imgs_np_test = imgs_np[-test_split:,:]
        self.cs_np_test = cs_np[-test_split:,:]

        print("Samples for train: {}".format(imgs_np.shape[0] - test_split))
        print("Samples for test: {}".format(test_split))

        # generator output types
        output_types = (tf.float32, tf.float32)

        # create generators
        self.train_gen = tf.data.Dataset.from_generator(self.train_generator, output_types=output_types).batch(batch_size)
        self.test_gen = tf.data.Dataset.from_generator(self.test_generator, output_types=output_types).batch(batch_size)


    def train_generator(self):

        for sample_idx in range(len(self.imgs_np_train)):

            img = self.imgs_np_train[sample_idx,:]
            c_point = self.cs_np_train[sample_idx,:]

            # add noise
            #white_noise = np.rint(np.clip(np.random.uniform(low=0.0, high=1.0, size=img.shape), self.noise_thresh, 1.0)).astype(int)
            white_noise = np.clip(np.random.randint(low=0, high=self.noise_thresh, size=img.shape) - (self.noise_thresh - 2), 0, 1)

            yield np.clip(img + white_noise, 0, 1), c_point
            #yield img, c_point
        
    def test_generator(self):

        for sample_idx in range(len(self.imgs_np_test)):

            img = self.imgs_np_test[sample_idx,:]
            c_point = self.cs_np_test[sample_idx,:]

            yield img, c_point


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

    data_dir = 'build/data10k_v2'
    for i in range(2, 10):
        prepare_data(data_dir, dest_name=f'data_10k_v2_{i}.h5', start_idx=1000*i, scale=50, end_idx=1000*(i+1), limit_samples=1e6)
