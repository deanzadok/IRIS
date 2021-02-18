import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

scale = 50

for i in range(1,101):

    img = np.zeros((101,101))
    obstacles = (pd.read_csv(f'learning/data/test_planar_{i}_obstacles.csv') * scale).round(0).astype(int)
    obstacles.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x']+x['width'], x['y']+x['length']), 255, -1), axis=1)

    inspection_points = (pd.read_csv(f'learning/data/test_planar_{i}_inspection_points.csv') * scale).round(0).astype(int)
    inspection_points.apply(lambda x: cv2.rectangle(img, (x['x'], x['y']),  (x['x'], x['y']), 100, -1), axis=1)

    cv2.imwrite('test.png', img)
    pass
