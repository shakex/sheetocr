from data import alphabets

# -----------
# DATA

# data analysis (data/analysis.py)
zh_label_fp = 'data/dataset/financial_sheet/phrases/zh.txt' # path to chinese label file                  
num_label_fp = 'data/dataset/financial_sheet/phrases/num.txt' # path to number label file
fig_save_dir = 'data/figs' # dir to analytical figures

# Chinese data augmentation (data/augmentation.py)
gen_data_type = 'chinese' # generate data type: ['chinese', 'number']
gen_num = 14 # generate numbers
gen_save_dir = 'dataset/financial_sheet/word/zh_word_aug' # path to generated image and label save dir
gen_imgH = 26 # generated image height
gen_font_path = 'simsun.ttc' # path to font file

# dataset
alphabet = alphabets.alphabet
trainroot = '' # dir to training set
valroot = '' # dir to val set

# create dataset (data/create_dataset.py)
lmdb_dir = 'data/dataset/baidu_zh/train/lmdb/val' # path to folder (must be empty dir)
img_dir = 'data/dataset/baidu_zh/train/val' # path to folder which contains the images
label_path = 'data/dataset/baidu_zh/train/val.txt' # path to file which contains the image path and label


# ----------
# TRAINING & INFERENCE (Text Recognition with CRNN)
# (reference: Holmeyoung/crnn-pytorch: https://github.com/Holmeyoung/crnn-pytorch)

# about data and net
alphabet = alphabets.alphabet
keep_ratio = True # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
imgH = 32 # the height of the input image to network
imgW = 100 # the width of the input image to network, if keep_ratio == True, will change imgW to ratio*imgH (ratio=w/h, ratio >= 1)
nh = 256 # size of the lstm hidden state
nc = 1  # number of input channels
pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'recognition/expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

# training process
displayInterval = 100 # interval to be print the train loss
valInterval = 1000 # interval to val the model loss and accuray
saveInterval = 1000 # interval to save model
n_val_disp = 80 # number of samples to display when val the model

# finetune
nepoch = 100 # number of epochs to train for
batchSize = 1 # input batch size
lr = 0.0001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = True # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)

# inference
model_arch = 'crnn' # model name used for inference
model_path = 'recognition/pretrained/crnn.pth' # trained model used for inference


# ---------
# SERVING

ocr_dir = 'serve/ocr' # dir to images before ocr
png_dir = 'serve/png' # dir to images after ocr  
xml_dir = 'serve/xml' # dir to .xml file after ocr
csv_dir = 'serve/csv' # dir to .csv file ocr
working_dir = 'serve/working' # working dir, save ache images
word_dir = 'serve/word_detected' # dir to detected text images
sleep_time = 5