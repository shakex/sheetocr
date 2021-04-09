# -*- coding: UTF-8 -*-
"""
date: 2020-12-15
author@jcdeng & kxie
usage:

"""

import os
import csv
import time
import argparse
import xml.dom.minidom as xmldom

import params

def xml2csv(xmlname, csvname):
    """convert xml to csv.
    
    params:
    - xml_path: path to xml file
    - csv_path: path to save csv file 
    """

    xml_content_list = []
    domobj = xmldom.parse(xmlname)

    #get root node
    root = domobj.documentElement
    # print("nodeName: %s, nodeValue: %s, nodeType: %s, nodesvalue: %s"%(root.nodeName, root.nodeValue, root.nodeType, root.getAttribute("name")))

    #get node by name 1st-level nodes
    nodeImageHead = root.getElementsByTagName("ImageHead")[0]
    nodeImageBody =  root.getElementsByTagName("ImageBody")[0]
    # print('1st level node: [%s,%s]'%(nodeImageHead.nodeName, nodeImageBody.nodeName))

    #get 2nd level nodes
    nodeTitle = nodeImageHead.getElementsByTagName("Title")[0]
    nodeTitleInfo = nodeImageHead.getElementsByTagName("TitleInfo")[0]
    nodeImageBodyInfo = nodeImageBody.getElementsByTagName("ImageBodyInfo")[0]
    # print('2nd level node: [%s, %s], [%s]'%(nodeTitle.nodeName, nodeTitleInfo.nodeName, nodeImageBodyInfo.nodeName))

    #get the leaf node of Title
    titleName = []
    for node in nodeTitle.childNodes:
    #node = nodeTitle.getElementsByTagName("imageName")[0]
        if(node.nodeType == node.ELEMENT_NODE):
            if (node.nodeName == 'content'):
                # print('nodeName: %s, value: %s'%(node.nodeName, '利润表'))
                titleName.append(node.childNodes[0].data)
                break
            # print('nodeName: %s, value: %s'%(node.nodeName, node.childNodes[0].data))
            leafNode = nodeTitle.getElementsByTagName(node.nodeName)[0]
            #print('value: %s'%(leafNode.childNodes[0].data))
    titleInfoALL = []
    #get 3rd level node of 2nd level node TitleInfo
    for node3rd in nodeTitleInfo.childNodes:
        #get 4th node of 2nd level node TitleInfo
        key_value = []
        for node4th in node3rd.childNodes:
            #get the leaf node of 2nd level node TitleInfo
            for node in node4th.childNodes:
                # if node.nodeType == node.TEXT_NODE:
                #     if(node.Text == None):
                #             print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
                if(node.nodeType == node.ELEMENT_NODE):
                    if (node.nodeName == 'content'):
                        if(len(node.childNodes)==1):
                            # print('nodeName: %s, value: %s'%(node.nodeName, node.childNodes[0].nodeValue))
                            key_value.append(node.childNodes[0].nodeValue)
                        else:
                            # print('nodeName: %s, value: %s'%(node.nodeName, ''))
                            key_value.append('')
                    # else:
                        # print('nodeName: %s, value: %s'%(node.nodeName, node.childNodes[0].nodeValue))
        if len(key_value)==2:
            titleInfoALL.append(key_value)

    # print(titleInfoALL)
    titleInfo = [[titleInfoALL[0][0]+'：'+titleInfoALL[0][1], titleInfoALL[1][0]+'：'+titleInfoALL[1][1]],
                 [titleInfoALL[1][0]+'：'+titleInfoALL[1][1], titleInfoALL[2][0]+'：'+titleInfoALL[2][1]]]
    # print(titleInfo)

    #get 3rd level node of 2nd level node ImageBodyInfo
    bodyInfoALL = []
    for node3rd in nodeImageBodyInfo.childNodes:
        #get 4th node of 2nd level node TitleInfo
        key_value = []
        for node4th in node3rd.childNodes:
            #get the leaf node of 2nd level node TitleInfo
            if(node4th.nodeName == 'BodyInfoKeyEle'):
                isKey = True
            else:
                isKey =False
            for node in node4th.childNodes:
                if(node.nodeType == node.ELEMENT_NODE):
                    if (node.nodeName == 'content'):
                        # print('nodeName: %s, value: %s'%(node.nodeName, node.nodeValue))
                        if(len(node.childNodes)==1):
                            # print('nodeName: %s, value: %s'%(node.nodeName, node.childNodes[0].nodeValue))
                            key_value.append(node.childNodes[0].nodeValue)
                        else:
                            # print('nodeName: %s, value: %s'%(node.nodeName, ''))
                            key_value.append('')
                    # else:
                        # print('nodeName: %s, value: %s'%(node.nodeName, node.nodeValue))#node.childNodes[0].data
        if len(key_value)==2:
            bodyInfoALL.append(key_value)

    # print(bodyInfoALL)

    with open(csvname,'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(titleName)
        f_csv.writerows(titleInfo)
        f_csv.writerows(bodyInfoALL)

def xml2csv_batch(input_dir, output_dir):
    """
    params:
    - input_dir
    - output_dir
    
    """

    count = 0
    count_all = 0
    time_elapse = 0
    xml_format = [".xml"]
    file_list = os.listdir(input_dir)
    file_list.sort()

    for i in range(0, len(file_list)):
        print("[{}/{}] ".format(i+1, len(file_list)), end='')
        xml_path = os.path.join(input_dir, file_list[i])
        if os.path.isfile(xml_path) and os.path.splitext(xml_path)[1] in xml_format:
            count_all = count_all + 1
            time_start = time.time()

            name = os.path.splitext(file_list[i])[0]
            csv_path = os.path.join(params.csv_dir, name + '.csv')

            try:
                xml2csv(xml_path, csv_path)

                # if success
                time_end = time.time()
                time_elapse = time_elapse + (time_end - time_start)
                count = count + 1
                print("{} - [INFO] - Name: {}, time: {:.2f}s, CSV saved to {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file_list[i], time_end - time_start, csv_path))
            
            except:
                print("{} - [ERROR] - Name: {}. Save CSV failed, pass.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file_list[i]))
                continue

    print("\n{} - [INFO] - Done. {}/{} success. total: {:.2f}s, avg.: {:.2f}s\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), count, count_all, time_elapse, time_elapse / count))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xml', required=True, help='path to input xml')
    parser.add_argument('-o', '--output', required=True, help='path to output image')
    args = parser.parse_args()

    # args.xml = '../demo/xml/lrb_000.xml'
    # args.csv = '../demo/csv/lrb_000.csv'
    
    xml2csv(os.path.abspath(args.xml), args.output)
