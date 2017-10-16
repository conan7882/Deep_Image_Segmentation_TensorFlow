# File: config.py
# Author: Qian Ge <geqian1001@gmail.com>

# directory of pre-trained vgg parameters
vgg_dir = 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg16.npy'
# vgg_dir = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg16.npy'

# directory of training data
data_dir_val = 'D:\\Qian\\GitHub\\workspace\\dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\'
data_dir_train = 'D:\\Qian\\GitHub\\workspace\\dataset\\Segmentation\\HED-BSDS\\HED-BSDS\\data\\'

# data_dir_val = 'E:\\GITHUB\\workspace\\CNN\\dataset\\BSR_bsds500\\BSR\\BSDS500\\data\\'

# root = 'E:\\GITHUB\\\workspace\\'
root = 'D:\\Qian\\GitHub\\workspace\\'
# directory of testing data
test_data_dir = root + 'CNN\\test\\'

# directory of inference data
infer_data_dir = root + 'CNN\\infer_data\\'

# directory for saving inference data
infer_dir = root + 'CNN\\infer\\'

# directory for saving summary
summary_dir = root + 'CNN\\model\\'

# directory for saving checkpoint
checkpoint_dir = root + 'CNN\\model\\'

# directory for restoring checkpoint
model_dir = root + 'CNN\\model_4\\'

# directory for saving prediction results
result_dir = root + 'CNN\\result\\'




