"""
author: chenqiaoyuan
date: 2020-12-25
"""
import os
import csv


# def read_csv(fp):
#     ret = []
#     with open(fp, 'r', encoding='utf-8') as f:
#         for l in f.readlines():
#             ret.append(l.strip().split(','))
#     return ret


def read_csv(fp, encoding='utf-8'):
    ret = []
    num = 0
    with open(fp, encoding=encoding) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            ret.append(row)
            num += len(row)
    return ret, num


def get_one_acc(pred_fp, gt_fp):
    pred, pred_num = read_csv(pred_fp, encoding='utf-8')
    gt, gt_num = read_csv(gt_fp, encoding='gbk')
    same = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j].strip() == gt[i][j].strip():
                same += 1
    acc = same / pred_num
    print('gt_num:{}, pred_num:{}, same:{}, acc:{:.2f}%'.format(gt_num, pred_num, same, acc))
    return acc


gt_fold = './OCR真值文件/'
gt_lrb = os.listdir(os.path.join(gt_fold, 'lrb'))
gt_xjllb = os.listdir(os.path.join(gt_fold, 'xjllb'))
gt_zcfzb = os.listdir(os.path.join(gt_fold, 'zcfzb'))
pred_fold = './csv'
pred_fp_list = os.listdir(pred_fold)
for pred_fp in pred_fp_list:
    subfold = pred_fp.split('.')[0].split('_')[0]
    if subfold == 'lrb':
        gt_fp_list = gt_lrb
    elif subfold == 'xjllb':
        gt_fp_list = gt_xjllb
    elif subfold == 'zcfzb':
        gt_fp_list = gt_zcfzb
    index = int(pred_fp.split('.')[0].split('_')[1])
    pred_fp = os.path.join(pred_fold, pred_fp)
    gt_fp = os.path.join(gt_fold+subfold, gt_fp_list[index])

    acc = get_one_acc(pred_fp, gt_fp)
    print(pred_fp, gt_fp)
    exit(0)

# get_one_acc('lrb_106.csv', '潍柴动力利润表2016.csv')
