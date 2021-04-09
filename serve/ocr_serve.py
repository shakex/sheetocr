# -*- coding: UTF-8 -*-
"""
date: 2020-12-21
author@kxie
usage: python ocr_serve.py
"""

import os
import sys
import shutil
import time
from inference import inference_batch
from export.xml2csv import xml2csv, xml2csv_batch

sys.path.append("..")
import params


def move_files(src_dir, dst_dir, file_type='.png'):
    src_files = os.listdir(src_dir)
    for file_name in src_files:
        if os.path.splitext(file_name)[1] == file_type:
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            shutil.move(src_path, dst_path)


if __name__ == '__main__':
    while (True):
        move_files(params.png_dir, params.working_dir, file_type='.png')
        working_files = os.listdir(params.working_dir)
        if len(working_files) > 0:
            inference_batch(params.working_dir, params.xml_dir)
            xml2csv_batch(params.xml_dir, params.csv_dir)

            # move files from working_dir to png_dir
            move_files(params.working_dir, params.png_dir, file_type='.png')

        else:
            print("{} - [INFO] - Nothing to process. Let me sleep for {}s".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), params.sleep_time))
            time.sleep(params.sleep_time)
            print("{} - [INFO] - I'm awake. Start working...".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
