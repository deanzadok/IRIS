import os
import glob
import cv2
import h5py
import sys
import numpy as np
import pandas as pd
from shapely.geometry import Polygon 
import visilibity as vis
from load import compute_links, is_in_bounds, save_print


def save_images_for_path(data_dir, trial_idx, p_zb, scale=50, eps=1e-7):

    max_wall = 200
    fov = np.pi / 2

    files_list = {'obstacles': os.path.join(data_dir, f'test_p_zb_{trial_idx}_{p_zb}_obstacles.csv'),
                'inspection_points': os.path.join(data_dir, f'test_p_zb_{trial_idx}_{p_zb}_inspection_points.csv'),
                'configurations': os.path.join(data_dir, f'test_p_zb_{trial_idx}_{p_zb}_conf'),
                'vertex': os.path.join(data_dir, f'test_p_zb_{trial_idx}_{p_zb}_vertex'),
                'results':os.path.join(data_dir, f'test_search_p_zb_{trial_idx}_{p_zb}_result')}

    # check if test files exist
    broken_files = False
    for file_path in files_list.values():
        if not os.path.isfile(file_path):
            broken_files = True
    
    if broken_files:
        print('Broken file')
        sys.exit()

    # construct image with obstacles
    img = np.zeros((101,101))
    obstacles = (pd.read_csv(files_list['obstacles']) * scale).round(0).astype(int)
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
        line = f.readlines()[-2]
        vertices_idxs = [int(x) for x in line.split(' ')[1:-1]]
        img_copy = img.copy()
        for j in range(0,len(vertices_idxs)-1):

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
                    cv2.imwrite(f'sample_path_{trial_idx}_{p_zb}_{j}.png', img_copy)

if __name__ == '__main__':

    data_dir = 'build/test_zb'
    save_images_for_path(data_dir=data_dir, trial_idx=155, p_zb=0.7, scale=50, eps=1e-7)