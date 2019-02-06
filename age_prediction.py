import tensorflow as tf
from dataset import Dataset
import datetime

IMAGE_SIZE = 128
NUM_EPOCHS = 100
MINIBATCH_SIZE = 16
LOG_DIR = "logs/simple_dense_{}".format(datetime.datetime.now())

class AgeRegressor:
    def __init__(self, size, color_channels, num_classes):
        self.size = size
        self.color_channels = color_channels
        self.num_classes = num_classes
        self.define_graph()

    def define_graph(self):
        self.training = tf.placeholder(tf.bool)
        self.images = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.color_channels])
        
        conv_1 = tf.layers.conv2d(self.images, 128, [5, 5], padding='same', activation=None)
        conv_1 = tf.layers.batch_normalization(conv_1, training=self.training)
        conv_1 = tf.nn.leaky_relu(conv_1)

        pool_1 = tf.layers.max_pooling2d(conv_1, [2, 2], 2)

        conv_2 = tf.layers.conv2d(pool_1, 128, [5, 5], padding='same', activation=None)
        conv_2 = tf.layers.batch_normalization(conv_2, training=self.training)
        conv_2 = tf.nn.leaky_relu(conv_2)

        pool_2 = tf.layers.max_pooling2d(conv_2, [2, 2], 2)

        conv_3 = tf.layers.conv2d(pool_2, 128, [5, 5], padding='same', activation=None)
        conv_3 = tf.layers.batch_normalization(conv_3, training=self.training)
        conv_3 = tf.nn.leaky_relu(conv_3)

        pool_3 = tf.layers.max_pooling2d(conv_3, [2, 2], 2)

        conv_4 = tf.layers.conv2d(pool_3, 128, [5, 5], padding='same', activation=None)
        conv_4 = tf.layers.batch_normalization(conv_4, training=self.training)
        conv_4 = tf.nn.leaky_relu(conv_4)

        pool_4 = tf.layers.max_pooling2d(conv_4, [2, 2], 2)

        conv_5 = tf.layers.conv2d(pool_4, 128, [5, 5], padding='same', activation=None)
        conv_5 = tf.layers.batch_normalization(conv_5, training=self.training)
        conv_5 = tf.nn.leaky_relu(conv_5)

        pool_5 = tf.layers.max_pooling2d(conv_5, [2, 2], 2)

        conv_6 = tf.layers.conv2d(pool_5, self.num_classes, [5, 5], padding='same', activation=None)
        conv_6 = tf.layers.batch_normalization(conv_6, training=self.training)
        conv_6 = tf.nn.leaky_relu(conv_6)

        pool_6 = tf.layers.max_pooling2d(conv_6, [2, 2], 2)

        #self.predicted_ages = tf.layers.flatten(pool_6)

        pool_6_flat = tf.layers.flatten(pool_6)

        dense_1 = tf.layers.dense(pool_6_flat, 100, activation=tf.nn.leaky_relu)
        #dropout = tf.layers.dropout(dense_1, rate=0.5, training=self.training)

        self.predicted_ages = tf.layers.dense(dense_1, self.num_classes, activation=None)
        self.expected_ages = tf.placeholder(tf.float32, [None, self.num_classes])

        tf.summary.image("predicted_ages", tf.reshape(tf.nn.softmax(self.predicted_ages), shape=[1, -1, self.num_classes, 1]))
        tf.summary.image("expected_ages", tf.reshape(self.expected_ages, shape=[1, -1, self.num_classes, 1]))

        self.age_difference = tf.reduce_mean(tf.cast(tf.abs(tf.argmax(self.predicted_ages, axis=1) - tf.argmax(self.expected_ages, axis=1)), tf.float32))
        tf.summary.scalar("age_difference", self.age_difference)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predicted_ages, labels=self.expected_ages)) # + tf.reduce_mean(tf.abs(self.predicted_ages))
        tf.summary.scalar("cost", self.cost)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.minimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        self.merged_summary = tf.summary.merge_all()
    
    def train(self, sess, input_images, ages):
        _, summary = sess.run([self.minimize, self.merged_summary], feed_dict={self.images: input_images, self.expected_ages: ages, self.training: True})
        return summary

    def test(self, sess, test_images, ages):
        summary = sess.run(self.merged_summary, feed_dict={self.images: test_images, self.expected_ages: ages, self.training: False})
        return summary

dataset = Dataset("./cleaned_data", IMAGE_SIZE, MINIBATCH_SIZE, test_size=100, validate_size=0)
age_regressor = AgeRegressor(IMAGE_SIZE, 1, dataset.num_classes)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initializers.global_variables())
    summary_writer_train = tf.summary.FileWriter(LOG_DIR + "_train", tf.get_default_graph())
    summary_writer_test = tf.summary.FileWriter(LOG_DIR + "_test", tf.get_default_graph())

    epoch = 0
    images_counter = 0

    while epoch < NUM_EPOCHS:
        
        while epoch == dataset.epoch:
            images, ages = dataset.get_next_batch()
            summary = age_regressor.train(sess, images, ages)

            images_counter += MINIBATCH_SIZE
            summary_writer_train.add_summary(summary, images_counter)
        
        images, ages = dataset.get_test_set()
        summary = age_regressor.test(sess, images, ages)
        summary_writer_test.add_summary(summary, images_counter)
        
        epoch = dataset.epoch
