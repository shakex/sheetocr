# -*- coding: UTF-8 -*-
"""
date: 2020-12-17
author@kxie
usage:
python xml_vis.py -i [image path] -x [xml path] -o [output path] [-f [show rec or not]]

"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from xml.etree.ElementTree import ElementTree


def read_xml(in_path):
    """read and parse xml file.

    params:
    - in_path (str):
    
    return:
    - tree
    """

    tree = ElementTree()
    tree.parse(in_path)

    return tree


def save_xml(tree, out_path):
    """save xml file.

    params:
    - tree: xml tree
    - out_path (str): xml save path
    """
    
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


# --------------- search -----
def find_nodes(tree, path):
    return tree.findall(path)


def find_node_by_tag(parent_node, tag):
    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        if child.tag == tag:
            return child


def is_node_not_empty(parent_node):
    """

    params:
    - parent_node:

    return:
    - (bool): 
    """

    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        if child.tag == 'isNotEmpty':
            if child.text == 'True':
                return True
            else:
                return False


def write_loc_to_dict(parent_node, img_name, loc_dict):
    """

    params:
    - parent_node:
    - img_name:
    - loc_dict: {key: img_name, value: loc_list}
    """

    loc_list = []
    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        # loc_list: [x1, y1, x2, y2]
        if child.tag == 'leftUpIndex' and child.text is not None:
            loc_list.append(int(child.text.replace('[', '').replace(']', '').split(',')[1]))
            loc_list.append(int(child.text.replace('[', '').replace(']','').split(',')[0]))
        if child.tag == 'rightDownIndex' and child.text is not None:
            loc_list.append(int(child.text.replace('[', '').replace(']', '').split(',')[1]))
            loc_list.append(int(child.text.replace('[', '').replace(']', '').split(',')[0]))

    if len(loc_list) == 4:
        loc_dict[img_name] = loc_list

    return loc_dict


def write_content_to_dict(parent_node, img_name, content_dict):
    """write predicted content to content_dict

    params:
    - parent_node
    - img_name
    - content_dict: {key: img_name, value: content}
    """

    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        if child.tag == 'content' and child.text is not None:
            content_dict[img_name] = child.text

    return content_dict


def get_dict_from_xml(xml_path, dict_type='location'):
    """get detected bounding box location from xml file.
     
    params:
    - xml_path (str): 
    - dict_type: 
        - 'location'
        - 'content'

    return:
    - out_dict
    """

    tree = read_xml(xml_path)
    out_dict = {}
    
    # title
    title_node = find_nodes(tree, "ImageHead/Title")
    if(is_node_not_empty(title_node[0])):
        title_img = find_node_by_tag(title_node[0], 'imageName')
        if dict_type == 'location':
            out_dict = write_loc_to_dict(title_node[0], title_img.text.split('.')[0], out_dict)
        if dict_type == 'content':
            out_dict = write_content_to_dict(title_node[0], title_img.text.split('.')[0], out_dict)

    # title info
    titleInfo_Keys = find_nodes(tree, "ImageHead/TitleInfo/TitleInfoEle/TitleInfoKeyEle")
    for titleInfo_key in titleInfo_Keys:
        if(is_node_not_empty(titleInfo_key)):
            titleInfo_key_imgNode = find_node_by_tag(titleInfo_key, 'imageName')
            if dict_type == 'location':
                out_dict = write_loc_to_dict(titleInfo_key, titleInfo_key_imgNode.text.split('.')[0], out_dict)
            if dict_type == 'content':
                out_dict = write_content_to_dict(titleInfo_key, titleInfo_key_imgNode.text.split('.')[0], out_dict)

    titleInfo_Values = find_nodes(tree, "ImageHead/TitleInfo/TitleInfoEle/TitleInfoValueEle")
    for titleInfo_value in titleInfo_Values:
        if (is_node_not_empty(titleInfo_value)):
            titleInfo_value_imgNode = find_node_by_tag(titleInfo_value, 'imageName')
            if dict_type == 'location':
                out_dict = write_loc_to_dict(titleInfo_value, titleInfo_value_imgNode.text.split('.')[0], out_dict)
            if dict_type == 'content':
                out_dict = write_content_to_dict(titleInfo_value, titleInfo_value_imgNode.text.split('.')[0], out_dict)

    # table body
    Body_key_nodes = find_nodes(tree, "ImageBody/ImageBodyInfo/BodyInfoEle/BodyInfoKeyEle")
    for body_key_node in Body_key_nodes:
        if (is_node_not_empty(body_key_node)):
            body_key_img = find_node_by_tag(body_key_node, 'imageName')
            if dict_type == 'location':
                out_dict = write_loc_to_dict(body_key_node, body_key_img.text.split('.')[0], out_dict)
            if dict_type == 'content':
                out_dict = write_content_to_dict(body_key_node, body_key_img.text.split('.')[0], out_dict)

    Body_value_nodes = find_nodes(tree, "ImageBody/ImageBodyInfo/BodyInfoEle/BodyInfoValueEle")
    for body_value_node in Body_value_nodes:
        if (is_node_not_empty(body_value_node)):
            body_value_img = find_node_by_tag(body_value_node, 'imageName')
            if dict_type == 'location':
                out_dict = write_loc_to_dict(body_value_node, body_value_img.text.split('.')[0], out_dict)
            if dict_type == 'content':
                out_dict = write_content_to_dict(body_value_node, body_value_img.text.split('.')[0], out_dict)
    
    return out_dict


def detection_vis(img_path, xml_path, out_path, fill=True, color=(0, 255, 255), alpha=0.5):
    """xml file visualization (show detected bounding box on image).
    
    params:
    - img_path (str): path to image
    - xml_path (str): path to xml file
    - out_path (str): path to save visualization image
    - fill (bool): fill bouding box or not
    - color (tuple): fill color or border color
    - alpha (float): if 'fill==True', set tranparency of fill color
    """

    img_name = os.path.basename(img_path)
    xml_name = os.path.basename(xml_path)
    
    if os.path.isfile(os.path.abspath(img_path)) and \
        os.path.isfile(os.path.abspath(xml_path)) and \
        img_name.split('.')[0] == xml_name.split('.')[0]:
        img = cv2.imread(img_path)
        img_copy = img.copy()
        loc_dict = get_dict_from_xml(xml_path, dict_type='location')
        for _, loc in loc_dict.items():
            x1, y1, x2, y2 = loc
            if fill:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if fill:
            img = cv2.addWeighted(img, alpha, img_copy, 1-alpha, gamma=0)

        cv2.imwrite(out_path, img)
        
        print("Results saved to {}".format(out_path))

    else:
        print("Parameters Error: invalid image or xml path.")


def add_text(img, text, org, font_face, text_color=(0, 255, 0), text_size=15):
    """cv2 image add text (chinese).

    params:
    - img
    - text: Text string to be drawn
    - org: Bottom-left corner of the text string in the image
    - font_face
    - text_color
    - text_size

    return:
    -
    """

    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_face, text_size, encoding="utf-8")
    draw.text(org, text, fill=text_color, font=font)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def detection_recognition_vis(img_path, xml_path, out_path, text_color=(0, 0, 0)):
    """xml file visualization (show detected bounding box and recognized text on image).

    params:
    - img_path (str): path to image
    - xml_path (str): path to xml file
    - out_path (str): path to save visualization image

    """

    img_name = os.path.basename(img_path)
    xml_name = os.path.basename(xml_path)

    if os.path.isfile(os.path.abspath(img_path)) and \
        os.path.isfile(os.path.abspath(xml_path)) and \
        img_name.split('.')[0] == xml_name.split('.')[0]:
        img = cv2.imread(img_path)

        img_draw = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) + 255
        # TODO: add to params.py
        font_face = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/data/simsun.ttc'
        loc_dict = get_dict_from_xml(xml_path, dict_type='location')
        content_dict = get_dict_from_xml(xml_path, dict_type='content')
        for name, content in content_dict.items():
            x1, y1, x2, y2 = loc_dict[name]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            img_draw = add_text(img_draw, content, (x1, y1), font_face=font_face, text_color=text_color, text_size=18)

        img_new = np.hstack((img, img_draw))
        cv2.imwrite(out_path, img_new)
        print("Results saved to {}".format(os.path.abspath(out_path)))
    
    else:
        print("Parameters Error: invalid image or xml path.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--image', required=True, help='path to input image')
    # parser.add_argument('-x', '--xml', required=True, help='path to output xml file')
    # parser.add_argument('-o', '--output', required=True, help='path to output image')
    args = parser.parse_args()

    args.image = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/demo/img/lrb_000.png'
    args.xml = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/demo/xml/zcfzb_000.xml'
    args.output = '/Users/shake/Documents/Project/project_ocr/OCR_Hub/demo/plot/zcfzb_000_vis_crnn.png'

    if os.path.isfile(os.path.abspath(args.image)) and os.path.isfile(os.path.abspath(args.xml)):
        # detection_vis(os.path.abspath(args.image), os.path.abspath(args.xml), args.output)
        detection_recognition_vis(os.path.abspath(args.image), os.path.abspath(args.xml), args.output)
    else:
        print("Parameters Error: invalid input or output.")