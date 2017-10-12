# File: HED.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference code: https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py

import tensorflow as tf

import tensorcv
from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel
from tensorcv.dataflow.image import *

def side_output(input_conv, o_height, o_width, name):
    with tf.variable_scope(name) as scope:
        side_out = conv(input_conv, 1, 1, 'conv', wd=0.0002,
                        init_w=tf.constant_initializer(),
                        init_b=tf.constant_initializer())
        side_out = tf.image.resize_bilinear(side_out, 
                    [o_height, o_width], name='output')
        return side_out

def  class_balanced_cross_entropy_with_logits(logits, label, 
                                name='class_balanced_cross_entropy'):
    '''
    original from 'Holistically-Nested Edge Detection (CVPR 15)'
    '''
    with tf.name_scope(name) as scope:
        logits = tf.cast(logits, tf.float32)
        label = tf.cast(label, tf.float32)
        num_pos = tf.reduce_sum(label)
        num_neg = tf.reduce_sum(1.0 - label)

        beta = num_neg / (num_neg + num_pos)
        pos_weight = beta / (1 - beta)

        cost = tf.nn.weighted_cross_entropy_with_logits(targets=label, 
                                                        logits=logits, 
                                                        pos_weight=pos_weight)
        loss = tf.reduce_mean((1 - beta) * cost)

        return tf.where(tf.equal(beta, 1.0), 0.0, loss)


class BaseHED(BaseModel):
    """ base of class activation map class """
    def __init__(self, num_class=2, 
                 num_channels=3, 
                 learning_rate=0.0001):

        self._learning_rate = learning_rate
        self._num_channels = num_channels
        self._num_class = num_class

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(tf.float32, name='image',
                            shape=[None, None, None, self._num_channels])
        self.label = tf.placeholder(tf.int32, [None, None, None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder([self.image, self.label])

    # def _create_conv(self, input_im):
    #     raise NotImplementedError()

    # def _get_loss(self):
    #     with tf.name_scope('loss'):
    #         side_loss = self._side_loss()
    #         tf.add_to_collection('losses', side_loss)

    #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #             logits=self.output, labels=self.label)
    #         cross_entropy_loss = tf.reduce_mean(cross_entropy, 
    #                             name='cross_entropy_loss') 
    #         tf.add_to_collection('losses', cross_entropy_loss)
    #         return tf.add_n(tf.get_collection('losses'), name='result') 

    # def _side_loss(self):
    #     raise NotImplementedError()     

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.5, learning_rate=self._learning_rate)

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, self.label)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32), 
                        name='result')

    def _setup_summary(self):
        tf.summary.scalar("train_accuracy", self.accuracy, collections=['train'])

    

class VGGHED(BaseHED):
    def __init__(self, num_class=2, 
                 num_channels=3, 
                 learning_rate=0.0001,
                 is_load=True,
                 pre_train_path=None):

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path 
        super(VGGHED, self).__init__(num_class=num_class, 
                                    num_channels=num_channels, 
                                    learning_rate=0.0001)

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]


        VGG_MEAN = [103.939, 116.779, 123.68]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, 
                                    value=input_im)
        input_bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path, encoding='latin1').item()

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu, trainable=False, data_dict=data_dict):
            conv1_1 = conv(input_im, 3, 64, 'conv1_1')
            conv1_2 = conv(conv1_1, 3, 64, 'conv1_2')
            pool1 = max_pool(conv1_2, 'pool1', padding='SAME')

            conv2_1 = conv(pool1, 3, 128, 'conv2_1')
            conv2_2 = conv(conv2_1, 3, 128, 'conv2_2')
            pool2 = max_pool(conv2_2, 'pool2', padding='SAME')

            conv3_1 = conv(pool2, 3, 256, 'conv3_1')
            conv3_2 = conv(conv3_1, 3, 256, 'conv3_2')
            conv3_3 = conv(conv3_2, 3, 256, 'conv3_3')
            conv3_4 = conv(conv3_3, 3, 256, 'conv3_4')
            pool3 = max_pool(conv3_4, 'pool3', padding='SAME')

            conv4_1 = conv(pool3, 3, 512, 'conv4_1')
            conv4_2 = conv(conv4_1, 3, 512, 'conv4_2')
            conv4_3 = conv(conv4_2, 3, 512, 'conv4_3')
            conv4_4 = conv(conv4_3, 3, 512, 'conv4_4')
            pool4 = max_pool(conv4_4, 'pool4', padding='SAME')

            conv5_1 = conv(pool4, 3, 512, 'conv5_1')
            conv5_2 = conv(conv5_1, 3, 512, 'conv5_2')
            conv5_3 = conv(conv5_2, 3, 512, 'conv5_3')
            conv5_4 = conv(conv5_3, 3, 512, 'conv5_4')
            # pool5 = max_pool(conv5_4, 'pool5', padding='SAME')

        with tf.name_scope('side_output') as scope:
            o_height, o_width = tf.shape(input_im)[1], tf.shape(input_im)[2]
            side_1 = side_output(conv1_2, o_height, o_width, 'side_1')
            side_2 = side_output(conv2_2, o_height, o_width, 'side_2')
            side_3 = side_output(conv3_4, o_height, o_width, 'side_3')
            side_4 = side_output(conv4_4, o_height, o_width, 'side_4')
            side_5 = side_output(conv5_4, o_height, o_width, 'side_5')

        with tf.variable_scope('output') as scope:
            side_mat = tf.concat([side_1, side_2, side_3, side_4, side_5], 3)
            self.output = conv(side_mat, 1, 1, 'fusion_weight',
                                wd=0.0002, use_bias=False,
                                init_w=tf.constant_initializer(0.2))
            prediction = tf.greater(tf.nn.sigmoid(self.output), 0.5)
            self.prediction = tf.cast(prediction, tf.int32, name='pre_label')
            self.output_list = [side_1, side_2, side_3, side_4, side_5, self.output]

    def _get_loss(self):
        with tf.name_scope('loss'):
            # cost = []
            for idx, out in enumerate(self.output_list):
                # TODO to be modified
                out = tf.squeeze(out, axis = -1)
                side_cost = class_balanced_cross_entropy_with_logits(out, self.label, name = 'cost_{}'.format(idx))
                tf.add_to_collection('losses', side_cost)
            
            return tf.add_n(tf.get_collection('losses'), name='result') 

            

         # with tf.name_scope('loss'):
         #    side_loss = self._side_loss()
         #    tf.add_to_collection('losses', side_loss)

         #    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         #        logits=self.output, labels=self.label)
         #    cross_entropy_loss = tf.reduce_mean(cross_entropy, 
         #                        name='cross_entropy_loss') 
         #    tf.add_to_collection('losses', cross_entropy_loss)
         #    return tf.add_n(tf.get_collection('losses'), name='result') 




if __name__ == '__main__':
    model = VGGHED(is_load=False,
           pre_train_path = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy')


            
    model.create_graph()

    grads = model.get_grads()
    opt = model.get_optimizer()
    train_op = opt.apply_gradients(grads, name = 'train')

    writer = tf.summary.FileWriter('E:\\GITHUB\\workspace\\CNN\\other\\')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
    writer.close()

