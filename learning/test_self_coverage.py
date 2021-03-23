from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import pandas as pd
import h5py
import cv2
import argparse
import tensorflow as tf
from models import VAEModel, BaseModel
import visilibity as vis
from shapely.geometry import Polygon 
from load import compute_links, save_print, is_in_bounds
import matplotlib.pyplot as plt

def load_plain_images(data_dir, trial_idx, scale=50):

    files_list = {'obstacles': os.path.join(data_dir, f'test_planar_{trial_idx}_obstacles.csv'),
                'inspection_points': os.path.join(data_dir, f'test_planar_{trial_idx}_inspection_points.csv'),
                'configurations': os.path.join(data_dir, f'test_planar_{trial_idx}_conf'),
                'vertex': os.path.join(data_dir, f'test_planar_{trial_idx}_vertex'),
                'results':os.path.join(data_dir, f'test_search_{trial_idx}_result')}

    # check if test files exist
    for file_path in files_list.values():
        if not os.path.isfile(file_path):
            print(f'{file_path} is missing...')
            return None, None, None, None, None

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

    # draw obstacles and inspection points
    obstacles.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)
    inspection_points.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']), (x['x'], x['y']), 100, -1), axis=1)

    # get initial starting point
    cspace_df = pd.read_csv(files_list['configurations'], delimiter=' ', header=None).drop(columns=[0,6])
    start_cpoint = cspace_df.iloc[0].values

    return img, vis_env, obstacles, inspection_points, start_cpoint


def paint_cpoint_on_img(img, vis_env, obstacles, inspection_points, c_point, scale=50, eps=1e-7, max_wall=200, fov=np.pi/2):

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
            print('No points for visibility polygon!')
            return None

        point_x.append(isovist[0].x())
        point_y.append(isovist[0].y())
        poly = Polygon([(x,y) for (x,y) in zip(point_x, point_y)])
        visilbe_poly = poly.intersection(vis_tri)
        if type(visilbe_poly) == Polygon and len(list(visilbe_poly.exterior.coords)) > 0:
            visilbe_poly_pts = np.array(list(visilbe_poly.exterior.coords)).reshape((-1,1,2)).astype(int)

            # draw visilbe polygon of the observer
            cv2.fillPoly(img, [visilbe_poly_pts], 150)

            # draw obstacles and inspection points
            obstacles.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)
            inspection_points.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']), (x['x'], x['y']), 100, -1), axis=1)

    return img


def compute_pixel_coverage(img):

    # get number of pixels per value
    vals, counts = np.unique(img, return_counts=True)
    counts_dict = {int(key):val for key,val in zip(vals, counts)}

    # return percentage of pixels coverage
    if 150 in counts_dict:
        return counts_dict[150] / (counts_dict[0] + counts_dict[150])
    else:
        return 0.0


def plot_coverage_means(coverage_df, xs, trials):

    coverage_means = coverage_df.mean().values
    coverage_stds = coverage_df.std().values
    stes = 1.96 * coverage_stds / np.sqrt(trials)

    # plor results
    plt.plot(xs, coverage_means)
    plt.fill_between(xs, coverage_means + stes, coverage_means - stes, alpha=0.5)
    plt.legend(['Coverage'])
    plt.xlabel('Step')
    plt.ylabel('Coverage')
    plt.title('Coverage per step')
    plt.grid(True)
    plt.savefig("coverage_steps.png")
    plt.clf()


@tf.function
def predict(inputs):
    return model(inputs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='path h5 data dir', default='build/data10k', type=str)
    parser.add_argument('--model_dir', '-model_dir', help='path to model file', default='learning/test_v2_nonoise', type=str)
    parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=8, type=int)
    parser.add_argument('--trials', '-trials', help='number of trials to count', default=300, type=int)
    parser.add_argument('--steps', '-steps', help='number of steps for prediction', default=10, type=int)
    parser.add_argument('--gpu', '-gpu', help='gpu number to train on', default='0', type=str)
    args = parser.parse_args()
 
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    stored_experiments = [16, 256, 192, 232, 258]

    # create model, loss and optimizer
    model = VAEModel(n_z=args.n_z, freeze=True)
    model.load_weights(os.path.join(args.model_dir, "vaemodel20.ckpt"))

    # store pixel coverage in dataframe
    pixcov_df = pd.DataFrame(columns=range(10))

    for j in range(0, args.trials):

        # get starting image with starting point
        img, vis_env, obstacles, inspection_points, start_cpoint = load_plain_images(data_dir=args.data_dir, trial_idx=j)
        if img is not None:
            img_painted = paint_cpoint_on_img(img.copy(), vis_env, obstacles, inspection_points, start_cpoint)
            if len(pixcov_df) in stored_experiments:
                cv2.imwrite(f'tests_{j}_0.png', img_painted)

            # store pixel coverage
            coverage_per_trial = {0:compute_pixel_coverage(img_painted)}

            # repeat sessions
            for i in range(1, args.steps):

                # prepare input
                img_tf = tf.convert_to_tensor(np.expand_dims(img_painted.reshape(-1), axis=0) / 255.0, dtype=tf.float32)
                z_tf = tf.random.normal(shape=[1,8], dtype=tf.float32)
                input_tf = tf.concat([z_tf,img_tf], axis=1)

                # predict next c-point
                predictions = predict(input_tf)
                pred_cpoint = predictions.numpy()[0] * np.pi

                img_painted = paint_cpoint_on_img(img_painted.copy(), vis_env, obstacles, inspection_points, pred_cpoint)
                coverage_per_trial[i] = compute_pixel_coverage(img_painted)

                if len(pixcov_df) in stored_experiments:
                    cv2.imwrite(f'tests_{j}_{i}.png', img_painted)

            pixcov_df = pixcov_df.append(coverage_per_trial, ignore_index=True)

        if j % 10 == 0:
            print(f'Processed {j}/{args.trials} images...')

    plot_coverage_means(coverage_df=pixcov_df, xs=list(range(args.steps)), trials=len(pixcov_df))

