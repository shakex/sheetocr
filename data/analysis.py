import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import params
from data import word_lists
from data.alphabets import alphabet_zh


class Analysis(object):
    """
    对表格中出现的中文、英文、数字进行统计

    中文
    - 样本总数
    - 单词长度分布
    - 各字符出现频次（分表统计，合并统计）
    - 各词组出现频次（合并统计）

    金额
    - 数量
    - 金额长度分布
    - 数字字符及',''.''-'出现频次
    """

    def __init__(self,
                 zh_label_fp=params.zh_label_fp,
                 num_label_fp=params.num_label_fp):

        plt.rcParams['font.family'] = 'Heiti TC'
        plt.rcParams['axes.unicode_minus'] = False
        self.zh_label_fp = zh_label_fp
        self.num_label_fp = num_label_fp

    def _get_label_dict(self, fp):
        fpr = open(fp, 'r')
        label_dict = dict()
        len_list = []

        while True:
            name = fpr.readline()
            label = fpr.readline()
            if not name or not label:
                break
            name = name.replace('\r', '').replace('\n', '')
            label = label.replace('\r', '').replace('\n', '')
            label_dict[name] = label
            len_list.append(len(label))

        num_cnt = len(label_dict)

        return num_cnt, np.array(len_list), label_dict


    def _get_char_dict(self, label_dict, alphabet):
        char_dict = dict()

        for char in alphabet:
            char_dict.setdefault(char, 0)

        for name, label in label_dict.items():
            for char in label:
                if char in char_dict:
                    char_dict[char] = char_dict[char] + 1

        return char_dict

    def _get_word_dict(self, label_dict, word_list):
        word_dict = dict()
        for word in word_list:
            word_dict.setdefault(word, 0)
        for word in word_list:
            if word in word_dict:
                for name, label in label_dict.items():
                    word_dict[word] = word_dict[word] + label.count(word)

        return word_dict

    def get_analysis_res(self, data_type='zh'):
        """


        """
        num_cnt, len_dict, label_dict = self._get_label_dict(self.zh_label_fp)

        if data_type == 'chinese':
            alphabet_str = alphabet_zh.replace('\r', '').replace('\n', '')
            char_dict = self._get_char_dict(label_dict, alphabet_str)
            zh_word_list = word_lists.lrb_word_list + word_lists.xjllb_word_list + word_lists.zcfzb_word_list + word_lists.company_word_list + word_lists.title_word_list
            word_dict = self._get_word_dict(label_dict, zh_word_list)
            return num_cnt, len_dict, char_dict, word_dict

        elif data_type == 'number':
            alphabet_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            char_dict = self._get_char_dict(label_dict, alphabet_num)
            return num_cnt, len_dict, char_dict

        else:
            print("Wrong data type.")


    def plot_len_hist(self, len_array, data_type='chinese', save_path='figs/analysis-zh_len_hist.png'):
        """AI is creating summary for plot_zh_len_hist

        """
        num_bins = np.max(len_array) - np.min(len_array)
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        n, bins, patches = ax.hist(len_array, label='Numbers', density=True, histtype='bar', bins=num_bins, facecolor='skyblue', edgecolor='black')
        mu = np.mean(len_array)
        sigma = np.std(len_array)
        print('mu:{} '.format(mu))
        print('sigma:{}'.format(sigma))

        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, 'r--')

        if data_type == 'chinese':
            ax.set_xlabel("Word length (Chinese)")
            ax.set_ylabel("#Occurrence")
            ax.set_title(r"Histogram of Chinese words ($\mu$={:.1f}, $\sigma$={:.1f})".format(mu, sigma))

        if data_type == 'number':
            ax.set_xlabel("Word length (Numbers)")
            ax.set_ylabel("#Occurrence")
            ax.set_title(r"Histogram of Numbers ($\mu$={:.1f}, $\sigma$={:.1f})".format(mu, sigma))

        plt.legend(loc=2)
        # plt.savefig(save_path)
        plt.show()


    def plot_barh(self, zh_dict, type="word", top_n=10, save_path='gis/analysis-zh_word_bar.png'):
        """

        """
        dict_len = len(zh_dict)
        if top_n >= dict_len:
            top_n = dict_len
        figsize_height_min = 3
        fig, ax = plt.subplots(1, 1, figsize=(4, max(top_n / 4, figsize_height_min)))

        sorted_dict = dict(sorted(zh_dict.items(), key=lambda kv: (kv[1], kv[0])))
        x = range(len(sorted_dict))[dict_len - top_n:]
        y = list(sorted_dict.values())[dict_len - top_n:]
        label = list(sorted_dict.keys())[dict_len - top_n:]
        ax.barh(x, y, tick_label=label, label='Number character', facecolor='skyblue', edgecolor='black')

        if type == 'char':
            ax.set_title("Top {} character occurrence".format(top_n))
            ax.set_xlabel("#Occurrence")
            ax.set_ylabel("Character")

        if type == 'word':
            ax.set_title("Top {} word occurrence".format(top_n))
            ax.set_xlabel("#Occurrence")
            ax.set_ylabel("Word")

        plt.legend(loc=4)
        # plt.savefig(save_path)
        plt.show()


# if __name__=="__main__":
#     analyzer = Analysis()
#     zh_cnt, zh_len_array, zh_char_dict, zh_word_dict = analyzer.get_analysis_res(data_type='chinese')
#     num_cnt, num_len_array, num_char_dict = analyzer.get_analysis_res(data_type='number')

#     # plot
#     top_n = 20
#     analyzer.plot_len_hist(zh_len_array, zh_cnt, data_type="chinese", save_path=os.path.join(params.fig_save_dir, 'analysis-zh_len_hist.png'))
#     analyzer.plot_barh(zh_char_dict, type="char", top_n=top_n, save_path=os.path.join(params.fig_save_dir, 'analysis-zh_char_bar.png'))
#     analyzer.plot_barh(zh_word_dict, type="word", top_n=top_n, save_path=os.path.join(params.fig_save_dir, 'analysis-zh_word_bar.png'))
#     analyzer.plot_len_hist(num_len_array, num_cnt, data_type="number", save_path=os.path.join(params.fig_save_dir, 'analysis-num_len_hist.png'))
#     analyzer.plot_barh(num_char_dict, type="char", top_n=10, save_path=os.path.join(params.fig_save_dir, 'analysis-num_char_bar.png'))
#     print('Figs saved to {}'.format(params.fig_save_dir))
