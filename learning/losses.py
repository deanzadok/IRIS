import tensorflow as tf

class CoverageLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute l2-norm according to the colors channels
        norm = tf.norm(y_true - y_pred, ord=2, axis=3)

        # sum the norm and compute the mean average over the batch
        return tf.reduce_mean(tf.reduce_sum(norm, axis=(1, 2)))
