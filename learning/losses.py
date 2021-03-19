import numpy as np
import tensorflow as tf

class CoverageLoss(tf.keras.losses.Loss):

    def __init__(self, batch_size, dim, inspection_points):
        super(CoverageLoss, self).__init__()

        self.batch_size = batch_size
        self.dim = dim
        self.inspection_points = tf.cast(inspection_points, dtype=tf.float32)
        self.fov = np.pi/2

        self.link_lengths = tf.constant([[0.2, 0.1, 0.2, 0.3, 0.1]])
        #self.link_lengths = tf.tile(self.link_lengths, tf.constant([self.batch_size,1], tf.int32))

        self.origin = tf.constant([[1.0, 0.0]]) 
        #self.origin = tf.tile(self.origin, tf.constant([self.batch_size,1], tf.int32))


    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        #y_true = tf.cast(y_true, y_pred.dtype)

        last_link, ee = self.compute_links(y_pred)

        
        visible_points = self.get_visible_points(last_link, ee)


        return tf.reduce_mean(tf.reduce_sum(visible_points, axis=1))
        #return tf.reduce_mean(visible_points)

    def compute_links(self, angles):

        # tile link lengths and origin with batch size
        link_lengths = tf.tile(self.link_lengths, tf.constant([angles.shape[0],1], tf.int32))
        origin = tf.tile(self.origin, tf.constant([angles.shape[0],1], tf.int32))

        # compute position of last link 
        last_link_x = tf.add(tf.reduce_sum(tf.multiply(link_lengths[:,:-1], tf.math.cos(angles[:,:-1] * np.pi)), axis=1, keepdims=True), origin[:,:1])
        last_link_y = tf.add(tf.reduce_sum(tf.multiply(link_lengths[:,:-1], tf.math.sin(angles[:,:-1] * np.pi)), axis=1, keepdims=True), origin[:,1:])
        last_link = tf.concat([last_link_x, last_link_y], axis=1)

        # compute position of end-effector 
        ee_x = tf.add(tf.reduce_sum(tf.multiply(link_lengths, tf.math.cos(angles * np.pi)), axis=1, keepdims=True), origin[:,:1])
        ee_y = tf.add(tf.reduce_sum(tf.multiply(link_lengths, tf.math.sin(angles * np.pi)), axis=1, keepdims=True), origin[:,1:])
        ee = tf.concat([ee_x, ee_y], axis=1)

        # for i in range(self.dim):
        #     end_point[:,0] = end_point[:,0] + self.link_lengths[:,i] * tf.math.cos(angles[:,i])
        #     end_point[:,1] = end_point[:,1] + self.link_lengths[:,i] * tf.math.sin(angles[:,i])

        #     link_positions.append(tf.identity(end_point))

        return last_link, ee

    def get_visible_points(self, last_link, ee):

        last_link_to_ee = tf.subtract(ee, last_link)
        last_link_to_ee_normalized = tf.linalg.normalize(last_link_to_ee, axis=1)[0]

        visible_points_for_ee = tf.zeros([ee.shape[0], self.inspection_points.shape[0]], tf.float32)
        #visible_points_num = tf.zeros([ee.shape[0]], dtype=tf.float32)


        for i in range(self.inspection_points.shape[0]):

            inspected_point = tf.tile(self.inspection_points[i:i+1,:], tf.constant([ee.shape[0],1], tf.int32))
            ee_to_point = tf.subtract(inspected_point, ee)
            ee_to_point_normalized = tf.linalg.normalize(ee_to_point, axis=1)[0]

            angle_vec = tf.reduce_sum(tf.multiply(last_link_to_ee_normalized, ee_to_point_normalized), axis=1)
            angle = tf.math.acos(angle_vec)

            angle_clipped = tf.clip_by_value(angle, 0.5 * self.fov, np.pi) - 0.5 * self.fov
            angle_sign = (-1) * (tf.math.sign(angle_clipped) - 1) 
            #visible_points_num = visible_points_num + angle_sign

            # get currently inspected points and pad them
            #one_bb_to_rule_them_all = tf.one_hot(tf.squeeze(tf.where(angle < 0.5 * self.fov), axis=1), ee.shape[0], dtype=tf.int32)
            #bb_reduced = tf.expand_dims(tf.reduce_sum(one_bb_to_rule_them_all, axis=0), axis=1)
            #bb_padded = tf.pad(bb_reduced, tf.constant([[0,0],[i, self.inspection_points.shape[0] - i - 1]]))

            # add them to the full one hot vector of visible points
            #visible_points_for_ee = visible_points_for_ee + bb_padded

            #one_bb_to_rule_them_all = tf.reduce_sum(tf.where(angle < 1 * self.fov, tf.ones_like(angle), tf.zeros_like(angle)))

            # for j in range(ee.shape[0]):
            #     if angle[j] < 1 * self.fov:
            #         counter = counter + tf.ones_like(counter)


            bb_reduced = tf.expand_dims(angle_sign, axis=1)
            bb_padded = tf.pad(bb_reduced, tf.constant([[0,0],[i, self.inspection_points.shape[0] - i - 1]]))
            visible_points_for_ee = visible_points_for_ee + bb_padded

            """
            if angle < 0.5 * self.fov:

                is_visible = True
                if is_visible:
            """


        return visible_points_for_ee