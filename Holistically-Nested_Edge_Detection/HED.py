# File: HED.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference code: https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py

import tensorflow as tf

import tensorcv
from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel
from tensorcv.dataflow.image import *
from tensorcv.utils.common import deconv_size, apply_mask, apply_mask_inverse

def bilinear_upsample(input_im, up, out_shape = None):
# https://github.com/BVLC/caffe/blob/master/include%2Fcaffe%2Ffiller.hpp#L244
# https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py#L145
    def dconv_filter(filter_shape):
        f = np.ceil(filter_shape) / 2.
        c = (filter_shape - 1) / (2. * f)
        ref = np.zeros((filter_shape, filter_shape), dtype='float32')
        for x in range(0, filter_shape):
            for y in range(0, filter_shape):
                ref[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ref

    with tf.name_scope('deter_bilinear_upsample'):
        input_s = input_im.shape.as_list()

        assert len(input_s) == 4, '[bilinear_upsample] Input tensor must be BHWC tensor!'
        input_ch = input_s[-1]

        up = int(up)
        f_shape = up * 2
        f_w = dconv_filter(f_shape)
        f_w = np.repeat(f_w, input_ch * input_ch).reshape((f_shape, f_shape, input_ch, input_ch))
        f_w = tf.constant(f_w, tf.float32,
                                 shape=(f_shape, f_shape, input_ch, input_ch),
                                 name='bilinear_upsample_filter')

        in_shape = tf.shape(input_im)
        if out_shape is None:
            out_s = tf.stack([in_shape[0], tf.multiply(in_shape[1], up), 
                            tf.multiply(in_shape[2], up), in_shape[3]])
        else:
            out_s = tf.stack([out_shape[0], out_shape[1], 
                                  out_shape[2], out_shape[3]])

        dconv = tf.nn.conv2d_transpose(input_im, f_w, 
                                   output_shape=out_s, 
                                   strides=[1, up, up, 1], 
                                   padding='SAME', 
                                   name='bilinear_upsample')
        
        # 
        if out_shape is None:
            if input_s[1]:
                input_s[1] = input_s * up
            if input_s[2]:
                input_s[2] = input_s * up
            dconv.set_shape(input_s)
        else:
            dconv.set_shape([None, None, None, input_ch])

        return dconv

def deconv_tensor_size(s, stride = 2):
    stride_tensor = tf.constant(stride, tf.float32)
    new_h = tf.cast(tf.ceil(tf.cast(s[1], tf.float32) / stride_tensor), tf.int32)
    new_w = tf.cast(tf.ceil(tf.cast(s[2], tf.float32) / stride_tensor), tf.int32) 
    return tf.stack([s[0], new_h, new_w, s[3]])
           
def side_output(input_conv, up, o_shape, name):
    with tf.variable_scope(name) as scope:
        side_out = conv(input_conv, 1, 1, 'conv', 
                        wd=0.0002,
                        init_w=tf.constant_initializer(),
                        init_b=tf.constant_initializer())

        if up >= 2:
            
            out_shape = tf.stack([o_shape[0], o_shape[1], o_shape[2], 1])
            shape_list = [out_shape,]
            prev_s = out_shape
            while up > 2:
                prev_s = deconv_tensor_size(prev_s)
                shape_list.append(prev_s)
                up = up / 2

            for o_s in shape_list[::-1]:
                side_out = bilinear_upsample(side_out, 2, out_shape = o_s)

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
        self.consensus_label = tf.cast(tf.greater(self.label, 1), tf.int32)

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
        # learning_rate = tf.train.exponential_decay(self._learning_rate, self.get_global_step,
        #                                       80000, 0.1, False)
        return tf.train.AdamOptimizer(learning_rate=self._learning_rate)

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            num_pos = tf.reduce_sum(self.consensus_label)
            num_neg = tf.reduce_sum(1 - self.consensus_label)
            neg_ratio = num_neg / (num_neg + num_pos)
            pos_ration = 1 - neg_ratio

            true_pos = apply_mask(tf.equal(self.prediction, self.consensus_label), self.consensus_label)
            true_neg = apply_mask_inverse(tf.equal(self.prediction, self.consensus_label), self.consensus_label)

            correct_prediction = neg_ratio * tf.reduce_mean(tf.cast(true_pos, tf.float64)) +\
                               pos_ration * tf.reduce_mean(tf.cast(true_neg, tf.float64))
            self.accuracy = tf.identity(correct_prediction, name='result')

    def _setup_summary(self):
        tf.summary.scalar("train_accuracy", self.accuracy, collections=['train'])
        tf.summary.image('GT', tf.expand_dims(tf.cast(self.consensus_label, tf.float32), -1), collections=['infer'])
        with tf.name_scope('side_out'):
            for idx, out in enumerate(self.output_list):
                tf.summary.image("side_{}".format(idx), tf.cast(out, tf.float32), collections=['infer'])
        with tf.name_scope('prediction_out'): 
            tf.summary.image('prediction', tf.expand_dims(tf.cast(self.prediction_prob, tf.float32), -1), collections=['infer'])


    

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

    # @property    
    # def _consensus_label(self, label):
    #     return tf.greater(label, 1)

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
            # conv3_4 = conv(conv3_3, 3, 256, 'conv3_4')
            pool3 = max_pool(conv3_3, 'pool3', padding='SAME')

            conv4_1 = conv(pool3, 3, 512, 'conv4_1')
            conv4_2 = conv(conv4_1, 3, 512, 'conv4_2')
            conv4_3 = conv(conv4_2, 3, 512, 'conv4_3')
            # conv4_4 = conv(conv4_3, 3, 512, 'conv4_4')
            pool4 = max_pool(conv4_3, 'pool4', padding='SAME')

            conv5_1 = conv(pool4, 3, 512, 'conv5_1')
            conv5_2 = conv(conv5_1, 3, 512, 'conv5_2')
            conv5_3 = conv(conv5_2, 3, 512, 'conv5_3')
            # conv5_4 = conv(conv5_3, 3, 512, 'conv5_4')
            # pool5 = max_pool(conv5_4, 'pool5', padding='SAME')

        shape_list = tf.stack([tf.shape(conv1_2), tf.shape(conv2_2), tf.shape(conv3_3), tf.shape(conv4_3), tf.shape(conv5_3)])
        s = tf.identity(shape_list, name='check_shape')

        o_shape = tf.shape(o_im)
        side_1 = side_output(conv1_2, 1, o_shape, 'side_1')
        side_2 = side_output(conv2_2, 2, o_shape, 'side_2')
        side_3 = side_output(conv3_3, 4, o_shape, 'side_3')
        side_4 = side_output(conv4_3, 8, o_shape, 'side_4')
        side_5 = side_output(conv5_3, 16, o_shape, 'side_5')

    
        with tf.variable_scope('output') as scope:
            side_mat = tf.concat([side_1, side_2, side_3, side_4, side_5], 3)
            self.output = conv(side_mat, 1, 1, 'fusion_weight',
                                wd=0.0002, use_bias=False,
                                init_w=tf.constant_initializer(0.2))
            self.output_list = list(map(tf.nn.sigmoid, [side_1, side_2, side_3, side_4, side_5, self.output]))
            self.prediction_prob = tf.reduce_mean(tf.concat(self.output_list, 3), axis=-1, name='pre_prob')
            prediction = tf.greater(self.prediction_prob, 0.5)
            self.prediction = tf.cast(prediction, tf.int32, name='pre_label')
            
    def _get_loss(self):
        with tf.name_scope('loss'):
            # cost = []
            for idx, out in enumerate(self.output_list):
                # out = tf.nn.sigmoid(out)
                out = tf.squeeze(out, axis = -1)
                
                side_cost = class_balanced_cross_entropy_with_logits(out, self.consensus_label, name = 'cost_{}'.format(idx))
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
    model = VGGHED(is_load=False)
        # pre_train_path = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy')

#     num_class = 257
#     num_channels = 3

#     vgg_cam_model = VGGCAM(num_class = num_class, 
#                            inspect_class = None,
#                            num_channels = num_channels, 
#                            learning_rate = 0.0001,
#                            is_load = True,
#                            pre_train_path = 'E:\\GITHUB\\workspace\\CNN\pretrained\\vgg19.npy')
            
    model.create_graph()

    grads = model.get_grads()
    opt = model.get_optimizer()
    train_op = opt.apply_gradients(grads, name = 'train')

    writer = tf.summary.FileWriter('D:\\Qian\\GitHub\\workspace\\CNN\\other\\')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
    writer.close()

