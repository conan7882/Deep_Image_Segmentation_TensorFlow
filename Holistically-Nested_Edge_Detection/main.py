# File: main.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse

import tensorflow as tf

import tensorcv
from tensorcv.dataflow.dataset import *
from tensorcv.callbacks import *
from tensorcv.train.config import TrainConfig
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.predicts import *

from HED import VGGHED
import config as config_path



def get_config(FLAGS):
    # data for training
    dataset_train = BSDS500HED('train', resize=400,
                            data_dir=config_path.data_dir_train)
    # dataset_train = BSDS500('train', resize=400,
    #                         data_dir=config_path.data_dir_val)

    dataset_val = BSDS500('val', resize=400,
                          data_dir=config_path.data_dir_val)

    dataset_test = BSDS500('infer', resize=400,
                          data_dir=config_path.data_dir_val)


    inference_list_validation = [
                          InferScalars(['loss/result'], ['loss']),
                    ]
    inference_list_test = [
                          # InferImages('output/pre_prob', 'edge'),
                          # InferMat('test_loss', ['loss/gt_0', 'loss/side_cost_0', 'loss/side_sigmoid_out_0', 'loss/cost_0/logistic_loss'], 
                          #          ['gt_0', 'side_cost_0', 'side_sigmoid_out_0', 'logistic_loss_mat']),
                          InferScalars('accuracy/result', 'test_accuracy_cat'),
                          InferScalars('loss/result', 'loss_cat'),
                    ]

    return TrainConfig(
                 dataflow = dataset_train, 
                 model = VGGHED(is_load=True,
                            learning_rate=5e-6,
                            pre_train_path=config_path.vgg_dir),
                 is_load=True,
                 model_name='model-60000',
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    ModelSaver(periodic=1000),
                    TrainSummary(key=['train'], periodic=100),
                    FeedInferenceBatch(dataset_val, 
                                  periodic=100, 
                                  batch_count=400, 
                                  infer_batch_size=1,
                                  inferencers=inference_list_validation),
                    FeedInferenceBatch(dataset_test, 
                                  periodic=100, 
                                  batch_count=1, 
                                  infer_batch_size=1,
                                  extra_cbs = TrainSummary(key='infer'),
                                  inferencers=inference_list_test),
                    CheckScalar(['lr','loss/result'], periodic=10),
                    # CheckScalar(['accuracy/result','loss/result'], periodic=10),
                    # CheckScalar(['loss/cost_0/check_loss', 'loss/cost_0/check_loss_2', 'loss/cost_1/check'], periodic = 1)
                    # CheckScalar(['check_shape'], periodic=1),
                    # CheckScalar(['loss/out_shape', 'loss/gt_shape'], periodic=10),
                  ],
                 batch_size=10, 
                 max_epoch=20,
                 summary_periodic=100,
                 default_dirs=config_path)

if __name__ == '__main__':
    FLAGS = None
    config = get_config(FLAGS)
    SimpleFeedTrainer(config).train()