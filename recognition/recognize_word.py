import os
from recognition import utils
import torch
from PIL import Image
from recognition.sheetLoader import collateResizeNormalize
from xml.etree.ElementTree import ElementTree
import params


def recognize_word(word_path, transformer, converter, model, device):
    """Single word image recognition using pretrained model.
    
    params:
    - word_path (str): path to word image
    - transformer (class): image transformer
    - converter (class): text converter, used for decode predicted text
    - model：pretrained model
    - device (str): use cpu or gpu

    return:
    - sim_pred: predict text 
    """

    img = Image.open(word_path).convert('L')
    img = transformer(img)
    img = img.view(1, *img.size())
    
    with torch.no_grad():
        img = img.to(device)
        pred = model(img)
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        pred_size = torch.LongTensor([pred.size(0)])
        raw_pred = converter.decode(pred.data, pred_size.data, raw=True)
        sim_pred = converter.decode(pred.data, pred_size.data, raw=False)

        # TODO: calculate confidence

    return sim_pred


def recognize_word_dir(word_dir, model, device):
    """Multiple word images recognition using pretrained model
    
    Args:
    - word_dir (str): word images dir that generate after detection 
    - model: model 
    - device (str): use cpu or gpu

    return:
    - pred_dict: python dictionary that save pred text and image path, {key: img_name, value: content}
    """

    converter = utils.strLabelConverter(params.alphabet)
    transformer = collateResizeNormalize(imgH=params.imgH, imgW=params.imgW, keep_ratio=True, min_ratio=1)

    count = 0
    pred_dict = {}
    model.eval()
    image_type = ["png", "jpg", "bmp"]
    for root, _, files in os.walk(word_dir):
        for i, name in enumerate(files):
            if name.split('.')[-1] in image_type:
                img_path = os.path.join(root, name)
                pred = recognize_word(img_path, transformer, converter, model, device)
                pred_dict[name.split('.')[0]] = pred

    return pred_dict


def currency_postprocess(txt):
    """Modify wrong predictions for '.'&',' in currency.

    params:
    - txt (str): predict text

    return:
    - txt_copy (str): modified text
    """

    txt_len = len(txt)

    if txt_len > 4 and txt.isdigit():
        is_negative = False
        if txt[0] == '-':
            txt = txt.replace('-','')
            is_negative = True
        
        txt_copy = txt
    
        i = txt_len - 1
        while i >= 0:
            if txt_len - i == 5:
                txt_copy = txt_copy[:i+1] + '.' + txt_copy[i+1:]
                i = i - 1
                continue
            if txt_len - i > 4 and (txt_len - 5 - i) % 3 == 0:
                txt_copy = txt_copy[:i+1] + ',' + txt_copy[i+1:]
                i = i - 1
                continue
            i = i - 1
    
        if is_negative:
            txt_copy = '-' + txt_copy

        return txt_copy
    else:
        return txt


def postprocess(pred_dict):
    """Prediction post processing.

    params:
    - pred_dict

    return:
    - pred_dict
    """    

    for name, txt in pred_dict.items():
        txt = currency_postprocess(txt)

        # TODO: add language model to refine predict results

        pred_dict[name] = txt

    return pred_dict


def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def save_xml(tree, out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


# --------------- search -----
def find_nodes(tree, path):
    return tree.findall(path)


def find_node_by_tag(parent_node, tag):
    """
    params:
    - parent_node:
    - tag (str):

    return:
    - child:
    """

    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        if child.tag == tag:
            return child


def change_node_by_tagText(parent_node, tag, text):
    """同过属性及属性值定位一个节点，并删除之

    params:
    - parent_node:
    - tag:
    - text:

    """
    '''
      nodelist: 父节点列表
      tag:子节点标签'''
    lenNode = parent_node.__len__()
    for i in range(lenNode):
        child = parent_node.__getitem__(i)
        if child.tag == tag:
            child.text = text
            break

def isNodeNotEmpty(parent_node):
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


def write_xml(pred_dict, xml_path):
    """write prediction result (pred_dict) to xml file.
     
    params:
    - pred_dict
    - xml_path
    """

    # 1. read xml file
    tree = read_xml(xml_path)
    
    # 2. add pred content
    # title
    title_node = find_nodes(tree, "ImageHead/Title")
    if(isNodeNotEmpty(title_node[0])):
        title_img = find_node_by_tag(title_node[0], 'imageName')
        # change_node_by_tagText(title_node[0], 'content', title_img.text.split('.')[0])
        change_node_by_tagText(title_node[0], 'content', pred_dict[title_img.text.split('.')[0]])

    # title info
    titleInfo_Keys = find_nodes(tree, "ImageHead/TitleInfo/TitleInfoEle/TitleInfoKeyEle")
    for titleInfo_key in titleInfo_Keys:
        if(isNodeNotEmpty(titleInfo_key)):
            titleInfo_key_imgNode = find_node_by_tag(titleInfo_key, 'imageName')
            change_node_by_tagText(titleInfo_key, 'content', pred_dict[titleInfo_key_imgNode.text.split('.')[0]])

    titleInfo_Values = find_nodes(tree, "ImageHead/TitleInfo/TitleInfoEle/TitleInfoValueEle")

    for titleInfo_value in titleInfo_Values:
        if (isNodeNotEmpty(titleInfo_value)):
            titleInfo_value_imgNode = find_node_by_tag(titleInfo_value, 'imageName')
            change_node_by_tagText(titleInfo_value, 'content', pred_dict[titleInfo_value_imgNode.text.split('.')[0]])

    # table body
    Body_key_nodes = find_nodes(tree, "ImageBody/ImageBodyInfo/BodyInfoEle/BodyInfoKeyEle")
    for body_key_node in Body_key_nodes:
        if (isNodeNotEmpty(body_key_node)):
            body_key_img = find_node_by_tag(body_key_node, 'imageName')
            change_node_by_tagText(body_key_node, 'content', pred_dict[body_key_img.text.split('.')[0]])
    Body_value_nodes = find_nodes(tree, "ImageBody/ImageBodyInfo/BodyInfoEle/BodyInfoValueEle")
    for body_value_node in Body_value_nodes:
        if (isNodeNotEmpty(body_value_node)):
            body_value_img = find_node_by_tag(body_value_node, 'imageName')
            change_node_by_tagText(body_value_node, 'content', pred_dict[body_value_img.text.split('.')[0]])
    
    save_xml(tree, xml_path)


if __name__ == "__main__":
    pred_dict = {}
    xml_path = ''
    write_xml(pred_dict, xml_path)
    


