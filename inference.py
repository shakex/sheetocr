import os
import time
import argparse

import params
from detection.split_image import calOneImage
from recognition.load_model import load_model
from recognition.recognize_word import recognize_word_dir, postprocess, write_xml

"""
date: 2020-12-18
author@kxie
input: image path
output: ocr result (xml)
usage: python inference.py -i [image path/dir] -o [output save dir]

e.g.
python inference.py -i demo/img/lrb000.png -o demo/xml (single image inference)
python inference.py -i demo/img -o demo/xml (batch images inference)
"""

# solve problem (macOS):
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def inference(input_path, output_dir):
    """single image inference: input an image, output ocr result in xml format.

    params:
    - input_path: input image path
    - output_dir: xml save dir
    """

    img_name = os.path.basename(input_path).split('.')[0]

    # detection
    time_det_start = time.time()

    word_img_dir = os.path.abspath(os.path.join(params.word_dir, img_name))
    calOneImage(input_path, word_img_dir, output_dir)

    time_det_end = time.time()
    time_det_elapse = time_det_end - time_det_start

    # recognition
    if os.path.isdir(word_img_dir) and len(os.listdir(word_img_dir)) > 0:
        time_rec_start = time.time()

        trained_model, device = load_model(params.model_arch, params.model_path)
        pred_dict = recognize_word_dir(word_img_dir, trained_model, device)
        pred_dict = postprocess(pred_dict)

        xml_path = os.path.abspath(os.path.join(output_dir, img_name + '.xml'))
        write_xml(pred_dict, xml_path)

        time_rec_end = time.time()
        time_rec_elapse = time_rec_end - time_rec_start
        print("{} - [INFO] - Success. "
              "{} word bbx detected in {}. "
              "det.time: {:.2f}s, "
              "rec.time: {:.2f}s, "
              "total {:.2f}s. "
              "(with {})"
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      len(os.listdir(word_img_dir)),
                      os.path.basename(input_path),
                      time_det_elapse,
                      time_rec_elapse,
                      time_det_elapse + time_rec_elapse,
                      device))
        print("\n{} - [INFO] - Detected word images saved in {}"
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), word_img_dir))
        print("{} - [INFO] - XML file saved in {}"
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), os.path.abspath(xml_path)))
    else:
        print("{} - [ERROR] - {}. No such directory or this directory is empty."
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), word_img_dir))


def inference_batch(input_dir, output_dir):
    """batch images inference: input multiple images, output ocr results in xml format.

    Args:
    - input_dir (str): input image dir
    - output_dir (str): xml save dir
    """

    count = 0
    count_all = 0
    time_elapse = 0

    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    file_list = os.listdir(input_dir)
    file_list.sort()

    # load model
    trained_model, device = load_model(params.model_arch, params.model_path)

    for i in range(0, len(file_list)):
        print("{} - [INFO] - {}/{}: "
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, len(file_list)), end='')
        input_path = os.path.join(input_dir, file_list[i])

        if os.path.isfile(input_path) and os.path.splitext(input_path)[1] in image_format:
            count_all = count_all + 1
            time_start = time.time()

            img_name = os.path.splitext(file_list[i])[0]
            word_img_dir = os.path.abspath(os.path.join(params.word_dir, img_name))

            try:
                # detection
                calOneImage(input_path, word_img_dir, output_dir)

                # recognition
                if os.path.isdir(word_img_dir) and len(os.listdir(word_img_dir)) > 0:
                    pred_dict = recognize_word_dir(word_img_dir, trained_model, device)
                    pred_dict = postprocess(pred_dict)

                    xml_path = os.path.abspath(os.path.join(output_dir, img_name + '.xml'))
                    write_xml(pred_dict, xml_path)

                    # if success
                    time_end = time.time()
                    time_elapse = time_elapse + (time_end - time_start)
                    count = count + 1
                    print("{} word bbx detected in {}. total {:.2f}s."
                          .format(len(os.listdir(word_img_dir)), file_list[i], time_end - time_start))
                else:
                    print("{} - [ERROR] - {}. No such directory or this directory is empty."
                          .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), word_img_dir))
            except:
                print("{} - [Error] - {}, inference failed, pass."
                      .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file_list[i]))
                continue
        else:
            print("{} - [Warning] - {}, pass."
                  .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file_list[i]))
            continue

    print("{} - [INFO] - Done. {}/{} success. total: {:.2f}s, avg.: {:.2f}s (with {})"
          .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  count,
                  count_all,
                  time_elapse,
                  time_elapse / count,
                  device))
    print("{} - [INFO] - Detected word images saved in {}"
          .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), params.word_dir))
    print("{} - [INFO] - XML saved in {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', required=True, help='path to input image')
    # parser.add_argument('-o', '--output', required=True, help='path to output xml file')
    args = parser.parse_args()

    args.input = '/home/pudding/data/project/SheetOCR/demo/img/lrb_000.png'
    args.output = '/home/pudding/data/project/SheetOCR/demo/xml/lrb_000.xml'

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if os.path.isfile(os.path.abspath(args.input)):
        inference(os.path.abspath(args.input), os.path.abspath(args.output))
    elif os.path.isdir(args.input):
        inference_batch(os.path.abspath(args.input), os.path.abspath(args.output))
    else:
        raise ValueError("invalid input or output")
