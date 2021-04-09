"""
Des: remove duplicate data from detection results, rgb2gray and create train and val set
Date: 2020-11-29 17:31:26
@Author: kxie
"""

import time
import os
import cv2
import shutil
from aip import AipOcr
from collections import defaultdict

dict_filetype=["png"]
# imgDir = '/Users/shake/Documents/Project/project_ocr/dataset/sheets/out/zcfzb_1_out'
imgDir = '/Users/shake/Documents/Project/project_ocr/dataset/sheet_words/words_gray'
saveDir = '/Users/shake/Documents/Project/project_ocr/dataset/all'


"""
move files from detection dir to saveDir
"""

# # move
# for root, dirs, files in os.walk(imgDir):
#     for dir in dirs:
#         if int(dir[dir.find('_')+1:dir.find('_')+4]) < 150:
#             for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
#                 for name in files2:
#                     if (dir.find("lrb_000") != -1 or dir.find("xjllb_000") != -1 or dir.find("zcfzb_000") != -1) and name.split('.')[-1] in dict_filetype and name.find("_0_") != -1:
#                         img_path = os.path.join(root2, name)
#                         img = cv2.imread(img_path)
#                         new_img_path = os.path.join(saveDir, name)
#                         cv2.imwrite(new_img_path, img)
#                     elif name.split('.')[-1] in dict_filetype and (name.find("_1_") != -1 or name.find("_H") != -1):
#                         img_path = os.path.join(root2, name)
#                         img = cv2.imread(img_path)
#                         new_img_path = os.path.join(saveDir, name)
#                         cv2.imwrite(new_img_path, img)
#                     else:
#                         continue
#
# # move all head
# # remove duplicate
# # 0.0000: [26,57],[28,57]
# # <blank>: [26,221],[28,222],[26,210],[26,194]
# cnt_all = 0
# cnt_remove = 0
# for root, dirs, files in os.walk(saveDir):
#     for i, name in enumerate(files):
#         if name.split('.')[-1] in dict_filetype:
#             img_path = os.path.join(root, name)
#             img = cv2.imread(img_path)
#             row, col, channel = img.shape
#             if (row == 26 and col == 57) or (row == 28 and col == 57):
#             # if (row == 26 and col == 221) or (row == 28 and col == 222) or (row == 26 and col == 210) or (row == 26 and col == 194):
#                 os.remove(img_path)
#                 cnt_remove = cnt_remove + 1
#         cnt_all = cnt_all + 1
#
# print("detect images moved to {}, #total: {}, #after remove duplicate value images: {}".format(saveDir, cnt_all, cnt_all - cnt_remove))
# print("starting to remove duplicate key images...")


# # Your APPID AK SK
# APP_ID = '18723573'
# API_KEY = 'HMDEKmk1BpulbKhVvjbIKuw2'
# SECRET_KEY = 'CULDhdjz2TD9KaGzQbjjRR0Cs7FQFNVW'
#
# # APP_ID = '23064133'
# # API_KEY = 'NkSUYYtElxZNGuf3AyXMG8Wq'
# # SECRET_KEY = 'kc478rqhdk7LS3rjAkGNIRdK1waQ4WPz'
# client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
#
# dict = defaultdict(list)
# options = {}
# options['language_type'] = 'CHN_ENG'
#
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
# cnt = 0
# for root, dirs, files in os.walk(saveDir):
#     for i, name in enumerate(files):
#         time.sleep(0.3)
#         print(i)
#         if name.split('.')[-1] in dict_filetype:
#             img_path = os.path.join(root, name)
#
#             img = get_file_content(img_path)
#             res = client.basicGeneral(img, options)
#             print(res)
#             if 'words_result' in res.keys() and res['words_result']:
#                 res_label = res['words_result'][0]['words'].replace(' ', '')
#                 dict[res_label].append(img_path)
#             else:
#                 print("{}: error".format(img_path))
#         cnt = cnt + 1
#
# # remove duplicate
# for key, value in dict.items():
#     if len(value) > 1:
#         for i, img_path in enumerate(value):
#             if i > 0:
#                 os.remove(img_path)
# print("images in {}, #total: {}, #after remove duplicates: {}".format(saveDir, cnt, len(dict)))



# select train_val images from value numbers
# for root, dirs, files in os.walk(imgDir):
#     for i, name in enumerate(files):
#         # if name.split('.')[-1] in dict_filetype and name.find("_H") != -1:
#         if name.split('.')[-1] in dict_filetype and i % 10 == 0:
#             img_path = os.path.join(root, name)
#             img = cv2.imread(img_path)
#             new_img_path = os.path.join(saveDir, name)
#             cv2.imwrite(new_img_path, img)


# for root, dirs, files in os.walk(imgDir):
#     for i, name in enumerate(files):
#         if name.split('.')[-1] in dict_filetype:
#             img_path = os.path.join(root, name)
#             img = cv2.imread(img_path)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             new_img_path = os.path.join(saveDir, name)
#             cv2.imwrite(new_img_path, gray)


# train&val data folder

# train_label_file = '/Users/shake/Documents/Project/project_ocr/dataset/train.txt'
# val_label_file = '/Users/shake/Documents/Project/project_ocr/dataset/val.txt'
# train_dir = '/Users/shake/Documents/Project/project_ocr/dataset/train'
# val_dir = '/Users/shake/Documents/Project/project_ocr/dataset/val'
# fp_train = open(train_label_file, 'r')
# fp_val = open(val_label_file, 'r')
#
# while True:
#     name_train = fp_train.readline()
#     str_train = fp_train.readline()
#     if not name_train or not str_train:
#         break
#
#     name_train = name_train.replace('\r', '').replace('\n', '')
#     shutil.copy(os.path.join(saveDir, name_train), os.path.join(train_dir, name_train))
#
# while True:
#     name_val = fp_val.readline()
#     str_val = fp_val.readline()
#     if not name_val or not str_val:
#         break
#
#     name_val = name_val.replace('\r', '').replace('\n', '')
#     shutil.copy(os.path.join(saveDir, name_val), os.path.join(val_dir, name_val))
#
#
# print("done.")
        



# create label

# Your APPID AK SK
# APP_ID = '18723573'
# API_KEY = 'HMDEKmk1BpulbKhVvjbIKuw2'
# SECRET_KEY = 'CULDhdjz2TD9KaGzQbjjRR0Cs7FQFNVW'

# APP_ID = '23064133'
# API_KEY = 'NkSUYYtElxZNGuf3AyXMG8Wq'
# SECRET_KEY = 'kc478rqhdk7LS3rjAkGNIRdK1waQ4WPz'
# client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
#
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
#
# # set dir
# work_dir = os.getcwd()
# data_dir = os.path.join(os.path.abspath(os.path.join(work_dir, "../..")), 'dataset/sheet_words')
# trainval_dir = os.path.join(data_dir, 'words')
# train_dir = os.path.join(data_dir, 'imgs/train')
# trainDir = os.path.join(data_dir, 'imgs/val')
# train_label_file = os.path.join(data_dir, 'gt_train.txt')
# val_label_file = os.path.join(data_dir, 'gt_val.txt')
# trainval_label_file = os.path.join(data_dir, 'gt_trainval.txt')
# trainval_save_dir = os.path.join(data_dir, 'words_gray')
#
#
# dict_filetype=["png"]
# fp = open(trainval_label_file, 'w')
#
# for root, dirs, files in os.walk(trainval_dir):
#     for i, name in enumerate(files):
#         if name.split('.')[-1] in dict_filetype:
#             img_path = os.path.join(root, name)
#             img = cv2.imread(img_path)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#             options={}
#             options['language_type'] = 'CHN_ENG'
#             img_content = get_file_content(img_path)
#             result = client.basicGeneral(img_content, options)
#             if 'words_result' in result.keys() and result['words_result']:
#                 result_label = result['words_result'][0]['words'].replace(' ','')
#                 fp.write("{}\n{}\n".format(name, result_label))
#                 cv2.imwrite(os.path.join(trainval_save_dir, name), gray)
#                 print(result)
#             else:
#                 print("{}: error".format(img_path))
#                 print(result)
#
#             time.sleep(0.3)
#
# fp.close()
# print("done.")


def img_padding(img, padding=5):
    return cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def img_dir_padding(dir, save_dir):
    dict_filetype = ["png"]
    for root, dirs, files in os.walk(dir):
        for i, name in enumerate(files):
            if name.split('.')[-1] in dict_filetype:
                img_path = os.path.join(root, name)
                img = cv2.imread(img_path)
                img_new = img_padding(img)
                img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(save_dir, name), img_new)


def create_alphabet_from_label(fp, out_dir):
    out_path = os.path.join(os.path.abspath(out_dir), 'alphabets.py')
    reader = open(fp, 'r', encoding='utf-8')
    writer = open(out_path, 'w', encoding='utf-8')
    writer.write('# -*- coding: UTF-8 -*-' + '\n' + 'alphabet = """')

    char_set = set()
    while True:
        name = reader.readline()
        phrase = reader.readline()
        if not name or not phrase:
            break
        name = name.replace('\r', '').replace('\n', '').split('.')[0]
        phrase = phrase.replace('\r', '').replace('\n', '')
        for char in phrase:
            char_set.add(char)
    for char in char_set:
        writer.write(char + '\n')

    writer.write('"""')
    writer.close()


def baidu_label_file_transform(fp, out_dir):
    out_path = os.path.join(os.path.abspath(out_dir), 'label.txt')
    reader = open(fp, 'r', encoding='utf-8')
    writer = open(out_path, 'w', encoding='utf-8')

    while True:
        str_line = reader.readline()
        str_line = str_line.replace('\\', '/').replace('\n', '')
        if not str_line:
            break
        name = str_line.split('\t')[0]
        name = os.path.basename(name)
        phrase = str_line.split('\t')[1]
        writer.write(name + '\n' + phrase + '\n')
        
    writer.close()

if __name__ == "__main__":
    # dir = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/financial_sheet/train/set1/val'
    # save_dir = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/financial_sheet/train/set1/val_new'
    # img_dir_padding(dir, save_dir)

    out_dir = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/baidu_zh'
    label_path = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/baidu_zh/label.txt'
    create_alphabet_from_label(label_path, out_dir)
    
    # out_dir = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/baidu_zh'
    # fp = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/dataset/baidu_zh/train.txt'
    # baidu_label_file_transform(fp, out_dir)