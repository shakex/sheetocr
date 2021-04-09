# SheetOCR
Implementation of modern text detection and recognition models for OCR on financial sheets. (wip)

## Requirements
- numpy
- matplotlib
- yaml
- pillow
- tqdm
- scipy
- lmdb
- torch
- torchvision
- six

## Code structure
```shell
├── data
│   ├── 3rd/                # 3rd-party tools for text data processing        
│   ├── dataset/            # dir to dataset
│   ├── figs/               # dir to analytical figures
│   ├── alphabets.py        # alphabet list
│   ├── word_lists.py       # word list            
│   ├── analysis.py         # data analysis
│   ├── augmentation.py     # data augmentation using EDA
│   ├── create_dataset.py   # create lmdb 
│   ├── data_utils.py       # divide train/val data
│   └── simsun.ttc          # font file
├── demo
│   ├── demo.sh             # run demo
│   ├── csv/                # dir to output csv
│   ├── img/                # dir to input images
│   ├── plot/               # dir to output result in png
│   ├── word/               # dir to detected word images
│   └── xml/                # dir to output xml
├── detection
│   ├── 3rd/                # code for other text detection methods
│   └── split_image.py      # text detection impl
├── tools
│   ├── xml_vis.py          # export result (before and after ocr)
│   └── xml2csv.py          # transform xml to csv format
├── recognition
│   ├── model/              # text recognition models
│   ├── load_model.py       # model loader
│   ├── recognize_word.py   # recognize word images and post_processing
│   ├── sheetLoader.py      # data loader
│   ├── train.py            # train model
│   └── utils.py            
├── serve
│   ├── word_detected/      # dir to detected word images
│   ├── working/            # working cache
│   ├── ocr_serve.py        # ocr service
│   ├── start.sh            # start ocr service
│   ├── stop.sh             # stop ocr service
├── inference.py            # ocr inference (single/batch)
├── params.py               # project configuration
├── requirements.txt        # dependency
├── LICENSE
└── README.md
```

## Install
`pip install -r requirements.txt`

## Dataset
### Data augmentation (with EDA)
```python
# params.py
gen_data_type = 'chinese' # generate data type: ['chinese', 'number']
gen_num = 14 # generate numbers
gen_save_dir = 'dataset/financial_sheet/word/zh_word_aug' # path to generated image and label save dir
gen_imgH = 26 # generated image height
gen_font_path = 'simsun.ttc' # path to font file

# data/augmentation.py 
generator = TextGenerator()
generator.gen_eda(params.gen_save_dir)
```

### Generate DB file for training and validation (using lmdb)
```python
python data/create_dataset.py --out lmdb/data/output/path --folder path/to/folder
```

## Train text recognition model (with CRNN)
### configs
```python
## in parmas.py
# Please reference: https://github.com/Holmeyoung/crnn-pytorch

# dataset
alphabet = alphabets.alphabet
trainroot = '' # dir to training set
valroot = '' # dir to val set

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
```

### Usage
```python
# start training
python train.py
```

## Inference
```python
# single image inference
python inference.py -i [image path] -o [xml dir]

# Batch images inference
python inference.py -i [image dir] -o [xml dir]
```

## Tools
```python
# Export to .png
python xml_vis.py -i [image path] -x [xml path] -o [output path]

# Export to .csv
python xml2csv.py -x [xml path] -o [csv path]
python xml2csv.py -x [xml dir] -o [csv dir]
```

## Demo
`sh demo.sh [image path] [csv path] [is_plot]`

## Serving
### Config
```python
# in params.py
ocr_dir = 'serve/ocr' # dir to images before ocr
png_dir = 'serve/png' # dir to images after ocr  
xml_dir = 'serve/xml' # dir to .xml file after ocr
csv_dir = 'serve/csv' # dir to .csv file ocr
working_dir = 'serve/working' # working dir, save ache images
word_dir = 'serve/word_detected' # dir to detected text images
sleep_time = 5
```
### Usage
```bash
# start server
sh start.sh

# stop server
sh stop.sh
```

## Contact us
For any problems, please contact kxie_shake [AT] outlook.com
