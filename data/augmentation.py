from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
from itertools import combinations

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import params
from data.analysis import Analysis


class TextGenerator(object):
    """Generate new phrases in chinese from the original text images.
    Inspired from the EMNLP-IJCNLP paper:
    EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks.
    pdf: https://arxiv.org/abs/1901.11196
    """

    def __init__(self,
                 gen_num=params.gen_num,
                 gen_height=params.gen_imgH,
                 gen_font=params.gen_font_path,
                 zh_label_fp=params.zh_label_fp,
                 alpha_rr=0.2,
                 alpha_ri=0.2,
                 alpha_rs=0.2,
                 alpha_rd=0.2):
        """
        Args:
            gen_num: int, number of augmented phrases to generate per original phrase.
            gen_height: float, generated image height.
            gen_font: str, generated image font file.
            zh_label_fp: str, chinese phrase label file that contains phrase image path and labeled text.
            alpha_rr: float, how much to replace each.
            alpha_ri: float, how much to insert new words.
            alpha_rs: float, how much to swap words.
            alpha_rd: float, how much to delete words.
        """

        if alpha_rr == alpha_ri == alpha_rs == alpha_rd == 0:
            raise ValueError('At least one alpha should be greater than zero')

        analyzer = Analysis()
        _, _, _, self.zh_word_dict = analyzer.get_analysis_res(data_type='chinese')

        self.gen_num = gen_num
        self.gen_height = gen_height
        self.gen_font = gen_font
        self.zh_label_fp = zh_label_fp
        self.alpha_rr = alpha_rr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd

    def get_contain_word_list(self, phrase):
        """
        Returns:
            contain_word_list: list, word list in phrase.
        """
        contain_word_list = []
        for word, _ in self.zh_word_dict.items():
            if word in phrase:
                contain_word_list.append(word)
        return contain_word_list

    def random_insertion(self, phrase, n):
        """Randomly insert n words into the phrase. (Random Insertion, RI)
        Args:
            phrase: str, original phrase.
            n: int, number of words to insert.
        
        Returns:
            new_phrase: phrase after RI.
        """

        new_phrase = phrase
        insert_loc_list = []
        for word, _ in self.zh_word_dict.items():
            if word in phrase:
                insert_loc_list.append(phrase.find(word))
                insert_loc_list.append(phrase.find(word) + len(word))
        insert_loc_list = list(set(insert_loc_list))
        insert_loc_list.sort()

        for _ in range(n):
            new_phrase, insert_loc_list = self._add_word(new_phrase, insert_loc_list)

        return new_phrase

    def _add_word(self, phrase, insert_loc_list):
        """add one word to phrase."""
        random_loc = random.choice(insert_loc_list)
        random_word = random.choice(list(self.zh_word_dict))
        if random_loc > 0:
            phrase = phrase[:random_loc] + random_word + phrase[random_loc:]
        else:
            phrase = random_word + phrase[random_loc:]

        idx = 0
        for i, loc in enumerate(insert_loc_list):
            if loc == random_loc:
                idx = i
            if loc >= random_loc:
                insert_loc_list[i] = loc + len(random_word)
        insert_loc_list.insert(idx, random_loc)

        return phrase, insert_loc_list

    def random_deletion(self, phrase, p):
        """Randomly delete words from the phrase with probability p. (Random Deletion, RD)
        Args:
            phrase: str, original phrase.
            p: float, probability of deletion for each word in phrase.
        
        Returns:
            new_phrase: phrase after RD.
        """

        # search for words in phrase
        contain_word_list = self.get_contain_word_list(phrase)

        # obviously, if there's only one word, don't delete it
        if len(contain_word_list) < 2:
            return phrase

        while True:
            new_phrase = phrase
            delete_cnt = 0

            for _ in range(len(contain_word_list)):
                # randomly delete word with probability p
                word = random.choice(contain_word_list)
                r = random.uniform(0, 1)
                if (r < p) and (word in new_phrase):
                    delete_cnt = delete_cnt + 1
                    new_phrase = new_phrase.replace(word, '')

            # successfully delete >=0 words in phrase, return new phrase
            for word in contain_word_list:
                if word in new_phrase:
                    return new_phrase

            # if you end up deleting all words and the remaining phrase is null, just return a random word
            if len(new_phrase) == 0:
                return random.choice(contain_word_list)

            # if you end up deleting all words but the remaining phrase is not null,
            # return with probability p, or do deletion again.
            r = random.uniform(0, 1)
            if r < p:
                return random.choice(contain_word_list)

    def random_swap(self, phrase, n):
        """Randomly swap two words in the phrase n times. (Random Swap, RS)
        Args:
            phrase: str, original phrase.
            n: int, swap times.
        
        Returns:
            new_phrase: phrase after RS.
        """

        new_phrase = phrase
        contain_word_list = self.get_contain_word_list(phrase)

        # phrase contains less than two words, just return phrase
        if len(contain_word_list) < 2:
            return phrase
        for _ in range(n):
            new_phrase = self._swap_word(new_phrase, contain_word_list)

        # bug: new_phrase should be the same size as phrase, if not, return phrase
        if len(new_phrase) != len(phrase):
            return phrase

        return new_phrase

    @staticmethod
    def _random_select(idx1, word1, phrase, word_list):
        """randomly select one new word from the word_list and """
        word2 = random.choice(word_list)
        idx2 = phrase.find(word2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            word1, word2 = word2, word1

        return idx1, idx2, word2

    def _swap_word(self, phrase, word_list):
        """randomly swap two words in the phrase."""
        word_list_copy = word_list.copy()
        word1 = random.choice(word_list_copy)
        word_list_copy.remove(word1)
        word2 = random.choice(word_list_copy)

        if word1 == word2:
            print("warning: chinese word list contains duplicated words.")
            return phrase

        idx1 = phrase.find(word1)
        idx2 = phrase.find(word2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # case for two words overlap
        if (idx1 + len(word1) > idx2) or (idx1 == -1 or idx2 == -1):
            if len(word_list) <= 3:
                return phrase
            else:
                while idx1 + len(word1) > idx2 or idx2 == -1:
                    idx1, idx2, word2 = self._random_select(idx1, word1, phrase, word_list_copy)
                while idx1 == -1:
                    idx1, idx2, word1 = self._random_select(idx2, word2, phrase, word_list_copy)

        new_phrase = \
            phrase[:idx1] \
            + phrase[idx2:idx2 + len(word2)] \
            + phrase[idx1 + len(word1):idx2] \
            + phrase[idx1:idx1 + len(word1)] \
            + phrase[idx2 + len(word2):]

        return new_phrase

    def random_replacement(self, phrase, n):
        """Randomly replace n words in the phrase (Random Replacement, RR)
        Args:
            phrase: str, original phrase.
            n: int, number of words to be replaced.
        
        Returns:
            new_phrase: phrase after RR.
        """

        new_phrase = phrase
        contain_word_list = self.get_contain_word_list(phrase)
        if len(contain_word_list) < 1:
            return phrase
        for _ in range(n):
            new_phrase = self._replace_word(new_phrase, contain_word_list)

        return new_phrase

    def _replace_word(self, phrase, word_list):
        """randomly replace one word in phrase."""
        word1 = random.choice(word_list)
        word2 = random.choice(list(self.zh_word_dict))
        while word2 in word_list:
            word2 = random.choice(list(self.zh_word_dict))
        new_phrase = phrase.replace(word1, word2)

        return new_phrase

    def _text2image(self, phrase):
        """text to image."""

        img = np.ones((self.gen_height, 20 * len(phrase) + 6, 3), dtype=np.uint8)
        img[::] = 255
        img = Image.fromarray(img)
        fill_color = (0, 0, 0)
        position = (3, 3)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.gen_font, 20)
        draw.text(position, phrase, font=font, fill=fill_color)
        img_gray = img.convert('L')  # rgb2gray if needed

        return img_gray


    @staticmethod
    def _get_list_len_without_overlap(word_list):
        n_overlap = 0
        for str1, str2 in list(combinations(word_list, 2)):
            min_len = min(len(str1), len(str2))
            for i in range(min_len, 0, -1):
                if str1[-i:] == str2[:i] or str2[-i:] == str1[:i]:
                    n_overlap += 1
                    break
        return len(word_list) - n_overlap

    def eda(self, phrase, include_original=True):
        """Easy Data Augmentation for one phrase.
        Args:
            phrase:
            include_original:

        Returns:
            gen_phrase_list: list of phrases that generated from the original phrase.
        """

        contain_word_list = self.get_contain_word_list(phrase)
        n_words = self._get_list_len_without_overlap(contain_word_list)
        if n_words < 1:
            return list()
        n_rs_max = math.factorial(n_words) - 1
        n_rd_max = 2 ** n_words - 2  # max = C_n^1+C_n^2+...+C_n^{n-1} (n>1)

        # RD && RI
        n_gen_rd = min(n_rd_max, int(self.gen_num / 4))
        n_gen_ri = int(self.gen_num / 2) + 1 - n_gen_rd
        gen_phrase_rd_set = set()
        gen_phrase_ri_set = set()

        if self.alpha_rd > 0:
            while len(gen_phrase_rd_set) < n_gen_rd:
                a_phrase = self.random_deletion(phrase, self.alpha_rd)
                if a_phrase != phrase:
                    gen_phrase_rd_set.add(a_phrase)

        if self.alpha_ri > 0:
            n_ri = max(1, int(self.alpha_ri * n_words))
            while len(gen_phrase_ri_set) < n_gen_ri:
                a_phrase = self.random_insertion(phrase, n_ri)
                if a_phrase != phrase:
                    gen_phrase_ri_set.add(a_phrase)

        # RS && RR
        n_gen_rs = min(n_rs_max, int(self.gen_num / 4))
        n_gen_rr = int(self.gen_num / 2) + 1 - n_gen_rs
        gen_phrase_rs_set = set()
        gen_phrase_rr_set = set()

        if self.alpha_rs > 0:
            n_rs = max(1, int(self.alpha_rs * n_words))
            while len(gen_phrase_rs_set) < n_gen_rs:
                # TODO: solve endless-loop problem while n_gen_rs == n_rs_max
                a_phrase = self.random_swap(phrase, n_rs)
                if a_phrase != phrase:
                    gen_phrase_rs_set.add(a_phrase)

        if self.alpha_rr > 0:
            n_rr = max(1, int(self.alpha_rr * n_words))
            while len(gen_phrase_rr_set) < n_gen_rr:
                a_phrase = self.random_replacement(phrase, n_rr)
                if a_phrase != phrase:
                    gen_phrase_rr_set.add(a_phrase)

        gen_phrase_list = list(set.union(gen_phrase_rd_set, gen_phrase_ri_set,
                                         gen_phrase_rs_set, gen_phrase_rr_set))

        # trim so that we have the desired number of augmented phrases
        if self.gen_num >= 1:
            gen_phrase_list = gen_phrase_list[:self.gen_num]
        else:
            keep_prob = self.gen_num / len(gen_phrase_list)
            gen_phrase_list = [s for s in gen_phrase_list if random.uniform(0, 1) < keep_prob]

        # append the original phrase
        if include_original:
            gen_phrase_list.append(phrase)

        return gen_phrase_list

    def gen_eda(self, gen_save_dir):
        """Generate augmented Chinese text images and label file with eda."""

        out_label_file = os.path.join(os.path.abspath(gen_save_dir), 'label_aug.txt')
        out_img_dir = os.path.join(os.path.abspath(gen_save_dir), 'img_aug')
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        writer = open(out_label_file, 'w', encoding='utf-8')
        reader = open(self.zh_label_fp, 'r', encoding='utf-8')

        pbar = tqdm(total=(int(len(reader.readlines()) / 2)))
        gen_phrase_set = set()
        reader.seek(0, 0)
        total_cnt = 0
        while True:
            name = reader.readline()
            phrase = reader.readline()
            if not name or not phrase:
                break

            name = name.replace('\r', '').replace('\n', '').split('.')[0]
            phrase = phrase.replace('\r', '').replace('\n', '')
            gen_phrase_list = self.eda(phrase)
            for i, gen_phrase in enumerate(gen_phrase_list):
                if gen_phrase not in gen_phrase_set:
                    gen_phrase_set.add(gen_phrase)
                    img = self._text2image(gen_phrase)
                    if gen_phrase != phrase:
                        gen_phrase_name = name + '-' + str(i+1) + '.png'
                    else:
                        gen_phrase_name = name + '.png'
                    writer.write(gen_phrase_name + '\n' + gen_phrase + '\n')
                    img.save(os.path.join(out_img_dir, gen_phrase_name))
                    total_cnt += 1
            pbar.update(1)

        pbar.close()
        writer.close()

        print("{} phrases generated with eda, images and label file saved to {}"
              .format(total_cnt, os.path.abspath(gen_save_dir)))


if __name__ == "__main__":
    generator = TextGenerator()
    generator.gen_eda(params.gen_save_dir)
