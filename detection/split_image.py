# -*-coding:utf-8-*-
"""
date: 2020-12-15
author@jcdeng
"""

import cv2
import numpy as np
import xml.dom.minidom
import os


#计算二值化的图像,imageName为图像路径，threshold为转换二值化的阈值
def calimg_binary(imageName, threshold=200):
    #img_src = cv2.imdecode(np.fromfile(imageName,dtype=np.uint8),-1)
    img_src = cv2.imread(imageName)
    img = img_src
    if(img_src.shape[2] == 3 ):
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)  # 获取灰度图
    ret, img_binary = cv2.threshold(img,threshold,1,cv2.THRESH_BINARY_INV)  ##像素值大于200的像素值置为0,否则为1
    return img_binary


def findLinePos(img_binary, gamma=10):
    """
    函数：findLinePos(img_binary, gamma)
    函数参数：
    img_binary: 待处理的二值化图像
    gamma： 松弛变量，默认取10
    返回值：
    rowIdx=[[startRow1, endRow1],[startRow2, endRow2],...[startRowN, endRowN]]: 每行表格的开始行索引和结束行索引
    colIdx=[[leftStartCol,leftEndCol],[rightStartCol,rightEndCol]]: 左列起始索引和结束索引，右列起始索引和结束索引
    firstRowIdx: 数值型，表格开始的一条直线的行索引
    """

    img_row = img_binary.shape[0]#高
    img_col = img_binary.shape[1]#宽

    rowPixelMaxNum=0 #保存可能的表格行线像素点数量的最大值
    for row in range(img_row):
        rowBinary_val = img_binary[row,:] #获得第row行的像素点
        sumBinaryRow = np.sum(rowBinary_val)
        if(rowPixelMaxNum<sumBinaryRow):
            rowPixelMaxNum = sumBinaryRow
    #print(("最长横线的像素点数量为：%s")%rowPixelMaxNum)

    rowIdx=[] #保存表格直线的行索引
    firstRowIdx = 0 #保存表格开始的一条直线的行索引
    row = 0
    while(row < img_row):
        rowBinary_val = img_binary[row,:] #获得第row行的像素点
        sumBinaryRow = np.sum(rowBinary_val)
        #保存表格开始的一条直线的行索引
        if((firstRowIdx == 0) and (np.abs(rowPixelMaxNum-sumBinaryRow)<=gamma)):
            firstRowIdx = row
            while((row < img_row) and (np.abs(rowPixelMaxNum-np.sum(img_binary[row,:]))<=gamma)):
                row = row + 1
            #保存表格第一行的开始和结束行索引
            startRow = row
            while((row < img_row) and (np.abs(rowPixelMaxNum-np.sum(img_binary[row,:]))>gamma)):
                row = row + 1 #表格内容的像素点数量少，所以可以据此跳过表格内容直到末尾
            endRow = row-1
            rowIdx.append([startRow, endRow])
            continue
        #处理表格第二行直到最后一行的开始和结束行索引，排除末尾的空白行
        if((firstRowIdx > 0) and (np.abs(rowPixelMaxNum-sumBinaryRow) > gamma) and (sumBinaryRow > 0)):
            startRow = row
            while((row < img_row) and (np.abs(rowPixelMaxNum-np.sum(img_binary[row,:]))>gamma)):
                row = row + 1 #表格内容的像素点数量少，所以可以据此跳过表格内容直到末尾
            endRow = row-1
            rowIdx.append([startRow, endRow])
        row = row + 1

    imgBodyBinary = img_binary[firstRowIdx:img_row,:]
    colPixelMaxNum=0 #计算表格列线最大值
    for col in range(img_col):
        colBinary_val = imgBodyBinary[:,col] #获得第col列的像素点
        sumBinaryCol = np.sum(colBinary_val)
        if(colPixelMaxNum<sumBinaryCol):
            colPixelMaxNum = sumBinaryCol
    #print(("最长竖线的像素点的数量为：%s")%colPixelMaxNum)
    colIdx=[] #保存表格直线的列索引
    col = 0
    while(col < img_col):
        colBinary_val = imgBodyBinary[:, col] #获得第col列的像素点
        sumBinaryCol = np.sum(colBinary_val)

        #如果第col列像素值和最大列像素值数相差太大，则说明不是列线，而是表格内容，但是要过滤掉开始和结束若干列的空白情况
        if((np.abs(colPixelMaxNum - sumBinaryCol) > gamma)and (np.sum(imgBodyBinary[:, col]) > gamma)):
            startCol = col #记录表格内容的开始列索引
            while((col < img_col) and (np.abs(colPixelMaxNum - np.sum(imgBodyBinary[:, col]))>gamma)):
                col = col + 1
            endCol = col-1 #记录表格内容的结束列索引
            colIdx.append([startCol, endCol])
        col = col + 1
    return rowIdx,firstRowIdx,colIdx


def findHeadInfoColIdx(titleInfoimg_binary, gamma=5):
    """
    # 函数名称: findHeadInfoColIdx(titleInfoImg, titleInfoimg_binary)
    # 参数:
    # titleInfoImg：待检测列坐标的图像标题信息图像
    # titleInfoImg：待检测列坐标的图像标题信息二值化图像
    # gamma： 松弛变量，默认取5
    # 返回值三维列表：
    # [
    #  [[startCol1,endCol1],[startCol2,endCol2]],
    #  [[startCol3,endCol3],[startCol4,endCol4]]
    # ] 分别表示左边两个字符串和右边两个字符串的开始和结束列索引
    """

    #先从两边向内扫描，找到两段字符串的开始和结尾
    cols = titleInfoimg_binary.shape[1]
    col = 0
    leftStartCol = 0
    rightEndCol = cols
    while (col < cols):
        leftSumCol = np.sum(titleInfoimg_binary[:, col])
        rightSumCol = np.sum(titleInfoimg_binary[:, cols-col-1])
        if (leftSumCol > 0 and leftStartCol == 0):
            leftStartCol = col
        if(rightSumCol > 0 and rightEndCol == cols):
            rightEndCol = cols - col - 1
        if(leftStartCol!=0 and rightEndCol!=cols):
            break
        col = col + 1
    #再从中间向两边扫描，找到两段字符串的结尾和开始
    col = round(cols/2)
    leftEndCol = 0
    rightStartCol = cols
    while(col < cols):
        leftSumCol = np.sum(titleInfoimg_binary[:, cols - col - 1])
        rightSumCol = np.sum(titleInfoimg_binary[:, col])
        if (leftSumCol > 0 and leftEndCol == 0):
            leftEndCol = cols - col - 1
        if(rightSumCol > 0 and rightStartCol == cols):
            rightStartCol = col
        if(leftEndCol!=0 and rightStartCol!=cols):
            break
        col = col + 1
    #找出左边字符串的最大空格，以最大空格的开始和结束为界，将左边字符串其分割成两份
    leftSpaceStart = leftStartCol
    leftSpaceEnd = leftEndCol
    maxSpaceVal = 0
    col = leftStartCol
    while(col < leftEndCol+1):
        if(np.sum(titleInfoimg_binary[:,col]) == 0):
            startSpaceCol = col
            zeroEndKeepCol = col
            while ((col < leftEndCol+1) and (np.sum(titleInfoimg_binary[:, col]) <= gamma)):
                if(np.sum(titleInfoimg_binary[:, col]) == 0):
                    zeroEndKeepCol = col
                col = col + 1
            endSpaceCol = zeroEndKeepCol
            if(maxSpaceVal < (endSpaceCol - startSpaceCol+1)):
                maxSpaceVal = (endSpaceCol - startSpaceCol+1)
                leftSpaceStart = startSpaceCol
                leftSpaceEnd = endSpaceCol
        col = col + 1
    #找出右边字符串的最大空格，以空格的开始和结束索引为界，将右边字符串分割成两份
    col = rightStartCol
    rightSpaceStart = rightStartCol
    rightSpaceEnd = rightEndCol
    maxSpaceVal = 0
    while(col < rightEndCol+1):
        if(np.sum(titleInfoimg_binary[:,col]) == 0):
            startSpaceCol = col
            zeroEndKeepCol = col
            while ((col < rightEndCol + 1) and (np.sum(titleInfoimg_binary[:, col]) <= gamma)):
                if(np.sum(titleInfoimg_binary[:, col]) == 0):
                    zeroEndKeepCol = col
                col = col + 1
            endSpaceCol = zeroEndKeepCol
            if (maxSpaceVal <= (endSpaceCol - startSpaceCol + 1)):
                maxSpaceVal = (endSpaceCol - startSpaceCol + 1)
                rightSpaceStart = startSpaceCol
                rightSpaceEnd = endSpaceCol
        col = col + 1
    return [[[leftStartCol, leftSpaceStart],[leftSpaceEnd, leftEndCol]],[[rightStartCol, rightSpaceStart],[rightSpaceEnd, rightEndCol]]]


def findHeadPos(img_binary, firstRowIdx, gamma=5):
    """
    # 函数名称：findHeadPos(img_binary, firstRowIdx, gamma)
    # 参数：
    # img_binary: 待处理的二值化图像
    # firstRowIdx: 数值型，表格开始的一条直线的行索引
    # gamma： 松弛变量，默认取5
    # 返回值：
    # titleIdx = [[startX,startY], [endX, endY]] #标题的左上和右下坐标点
    # titleInfoIdx =
    # [
    #  [ [[leftStartRowKey1, leftStartColKey1], [leftEndRowKey1, leftEndColKey1]],
    #    [[leftStartRowValue1, leftStartColValue1], [leftEndRowValue1, leftEndColValue1]] ],
    #  [ [[rightStartRowKey1, rightStartColKey1], [rightEndRowKey1, rightEndColKey1]],
    #    [[rightStartRowValue1, rightStartColValue1], [rightEndRowValue1, rightEndColValue1]] ],
    #  [ [[leftStartRowKey2, leftStartColKey2], [leftEndRowKe2y, leftEndColKey2]],
    #    [[leftStartRowValue2, leftStartColValue2], [leftEndRowValue2, leftEndColValue2]] ],
    #  [ [[rightStartRowKey2, rightStartColKey2], [rightEndRowKey2, rightEndColKey2]],
    #    [[rightStartRowValue2, rightStartColValue2], [rightEndRowValue2, rightEndColValue2]] ],
    #   ...
    #]
    # 标题信息的左上和右下坐标，四维列表
    # maxTitleHeight: 返回切割head部分小图像块的最大高度
    """

    img_row = img_binary.shape[0]#高
    img_col = img_binary.shape[1]#宽

    headimg_binary = img_binary[0:firstRowIdx,:]

    #从上向下遍历图像标题信息
    wordRowIdx=[]
    row=0
    maxTitleHeight = 0
    while(row < headimg_binary.shape[0]):
        if(np.sum(headimg_binary[row,:]) > 0):
            row_start=row #如果扫描到像素点(文字内容)，则保存行索引，并且让游标一直向下走，直到再次碰到空白行
            while((row < headimg_binary.shape[0]) and (np.sum(headimg_binary[row,:]) > 0)):
                row = row + 1
            row_end=row-1
            if(maxTitleHeight < (row_end - row_start)):
                maxTitleHeight = np.abs(row_end - row_start)
            wordRowIdx.append([row_start, row_end]) #保存数字行的开始和结束
        row = row + 1

    #保存返回值，标题和标题信息的左上角和右下角坐标
    titleIdx = []
    titleInfoIdx = []
    
    #检测标题左上和右下坐标
    if(len(wordRowIdx)>0):
        [row_start, row_end] = wordRowIdx[0]
        titleimg_binary = headimg_binary[row_start:row_end, :]
        titleColIdx=[]
        col = 0
        while(col < titleimg_binary.shape[1]):
            if(np.sum(titleimg_binary[:,col]) > 0):
                titleColIdx.append(col)
                while((col < titleimg_binary.shape[1]) and (np.sum(titleimg_binary[:,col]) > 0)):
                    col = col + 1
                titleColIdx.append(col-1)
            col = col + 1
        titleIdx = [[row_start, titleColIdx[0]],[row_end, titleColIdx[len(titleColIdx)-1]]]

    #标题信息1的左右各两个字符串的索引
    if(len(wordRowIdx)>1):
        [row_start, row_end] = wordRowIdx[1]
        titleInfoimg_binary = headimg_binary[row_start:row_end, :]
        #[[startCol1,endCol1],[startCol2,endCol2],[startCol3,endCol3],[startCol4,endCol4]]
        titleInfoColIdx1 = findHeadInfoColIdx(titleInfoimg_binary, gamma)
        if(len(titleInfoColIdx1)>0):
            for elePixelIdx in titleInfoColIdx1:
                #elePixelIdx:
                #[[titleInfoStartColKey, titleInfoEndColKey], [titleInfoStartColValue, titleInfoEndColValue]]
                titleInfoKeyIdx = [[row_start, elePixelIdx[0][0]], [row_end,elePixelIdx[0][1]]]
                titleInfoValueIdx = [[row_start, elePixelIdx[1][0]], [row_end,elePixelIdx[1][1]]]
                titleInfoIdx.append([titleInfoKeyIdx, titleInfoValueIdx])

    #标题信息2的左右各两个字符串的索引
    if(len(wordRowIdx)>2):
        [row_start, row_end] = wordRowIdx[2]
        titleInfoimg_binary = headimg_binary[row_start:row_end, :]
        titleInfoColIdx2 = findHeadInfoColIdx(titleInfoimg_binary, gamma)
        if(len(titleInfoColIdx2)>0):
            for elePixelIdx in titleInfoColIdx2:
                #elePixelIdx:
                #[[titleInfoStartColKey, titleInfoEndColKey], [titleInfoStartColValue, titleInfoEndColValue]]
                titleInfoKeyIdx = [[row_start, elePixelIdx[0][0]], [row_end,elePixelIdx[0][1]]]
                titleInfoValueIdx = [[row_start, elePixelIdx[1][0]], [row_end,elePixelIdx[1][1]]]
                titleInfoIdx.append([titleInfoKeyIdx, titleInfoValueIdx])

    return titleIdx, titleInfoIdx, maxTitleHeight


def findBodyIdx(img_binary, rowIdx, colIdx):
    """
    # 函数名称：findBodyIdx(img_binary, rowIdx, colIdx)
    # 函数参数：
    # img_binary：二值化的图像
    # # rowIdx=[[startRow1, endRow1],[startRow2, endRow2],...[startRowN, endRowN]]: 每行表格的开始行索引和结束行索引
    # # colIdx=[[leftStartCol,leftEndCol],[rightStartCol,rightEndCol]]: 左列起始索引和结束索引，右列起始索引和结束索引
    # 返回值：
    # maxBodyHeight: 表格小图像的最高值
    # 四维列表bodyIdx，保存每个小图像的左上和右下坐标值：
    # [
    # [ [[startKeyRow1, startKeyCol1],[endKeyRow1, endKeyCol1]],
    #   [[startValueRow1, startValueCol1],[endValueRow1, endValueCol1]] ],
    # [ [[startKeyRow2, startKeyCol2],[endKeyRow2, endKeyCol2]],
    #   [[startValueRow2, startValueCol2],[endValueRow2, endValueCol2]] ],
    #                               ...
    # [ [[startKeyRowN, startKeyColN],[endKeyRowN, endKeyColN]],
    #   [[startValueRowN, startValueColN],[endValueRowN, endValueColN]] ]
    # ]
    """

    bodyIdx = []
    maxBodyHeight = 0
    [orignlStartKeyCol, orignlEndKeyCol] = colIdx[0]  ##获得左边键开始和结束列索引
    [orignlStartValueCol, orignlEndValueCol] = colIdx[1] ##获得右边值开始和结束列索引
    for[orignlStartRow, orignlEndRow] in rowIdx:
        #首先,找出左边单元格的开始和结束行索引 左-行索引
        startKeyRow = orignlStartRow
        endKeyRow = orignlEndRow
        row = orignlStartRow+1
        while(row < orignlEndRow):
            sumPiexlRow = np.sum(img_binary[row, orignlStartKeyCol:orignlEndKeyCol])
            if(sumPiexlRow > 0): #从上到下扫描到第一个像素点后，说明已经遍历过空白部分，遇到了单元格内的文字上侧
                startKeyRow = row #记录左边单元格文字的开始行索引
                while((row<orignlEndRow) and (np.sum(img_binary[row, orignlStartKeyCol:orignlEndKeyCol])>0)):
                    row = row + 1
                endKeyRow = row #记录左边单元格文字的结束行索引
            row = row + 1
        #其次,找出右边单元格的开始和结束行索引 右-行索引
        startValueRow = orignlStartRow
        endValueRow = orignlEndRow
        row = orignlStartRow+1
        while(row < orignlEndRow):
            sumPiexlRow = np.sum(img_binary[row, orignlStartValueCol:orignlEndValueCol])
            if(sumPiexlRow > 0): #从上到下扫描到第一个像素点后，说明已经遍历过空白部分，遇到了单元格内的文字上侧
                startValueRow = row #记录右边单元格文字的开始行索引
                while((row<orignlEndRow) and (np.sum(img_binary[row, orignlStartValueCol:orignlEndValueCol])>0)):
                    row = row + 1
                endValueRow = row #记录右边单元格文字的结束行索引
            row = row + 1
        #记录表格文字的最高高度
        if(maxBodyHeight < np.abs(endKeyRow - startKeyRow)):
            maxBodyHeight = np.abs(endKeyRow - startKeyRow)
        if(maxBodyHeight < np.abs(endValueRow - startValueRow)):
            maxBodyHeight = np.abs(endValueRow - startValueRow)
        #再次,找出左边单元格的开始和结束列索引 左-列索引
        startKeyCol = orignlStartKeyCol
        endKeyCol = orignlEndKeyCol
        col = orignlStartKeyCol + 1
        while(col < orignlEndKeyCol):
            sumStartPiexlCol = np.sum(img_binary[startKeyRow:endKeyRow, col])
            sumEndColLeft = np.sum(img_binary[startKeyRow:endKeyRow, orignlEndKeyCol - (col-orignlStartKeyCol)-1])
            if((sumStartPiexlCol > 0) and (startKeyCol == orignlStartKeyCol)): #从左向右扫描到第一个像素点后，说明已经遍历过空白部分，遇到了单元格内的文字左侧
                startKeyCol = col #记录左单元格文字的开始列索引
            if((sumEndColLeft > 0) and (endKeyCol == orignlEndKeyCol)):
                endKeyCol = orignlEndKeyCol - (col-orignlStartKeyCol)-1 #记录左单元格文字的结束列索引
            if((startKeyCol != orignlStartKeyCol) and (endKeyCol != orignlEndKeyCol)):
                break
            col = col + 1
        #最后,找出右边单元格的开始和结束列索引 右-列索引
        startValueCol = orignlStartValueCol
        endValueCol = orignlEndValueCol
        col = orignlStartValueCol + 1
        while(col < orignlEndValueCol):
            sumStartPiexlCol = np.sum(img_binary[startValueRow:endValueRow, col])
            sumEndPiexlCol = np.sum(img_binary[startValueRow:endValueRow, orignlEndValueCol - (col-orignlStartValueCol)-1])
            #print(orignlEndValueCol - (col-orignlStartValueCol))
            if((sumStartPiexlCol > 0) and (startValueCol == orignlStartValueCol)): #从左向右扫描到第一个像素点后，说明已经遍历过空白部分，遇到了单元格内的文字左侧
                startValueCol = col #记录右单元格文字的开始列索引
            if((sumEndPiexlCol > 0) and (endValueCol == orignlEndValueCol)):
                endValueCol = orignlEndValueCol - (col-orignlStartValueCol)-1 #记录右单元格文字的结束列索引
            if((startValueCol != orignlStartValueCol) and (endValueCol != orignlEndValueCol)):
                #print('%d %d'%(startValueCol, endValueCol))
                break
            col = col + 1
        leftIdx = [[startKeyRow, startKeyCol], [endKeyRow, endKeyCol]]
        rightIdx = [[startValueRow, startValueCol], [endValueRow, endValueCol]]
        bodyIdx.append([leftIdx, rightIdx])

    return bodyIdx, maxBodyHeight


def savaCutImg(max_height, img_src, img_binary, imgSaveFullPathName, cutIdx, isTitle = False):

    [row_start, col_start] = cutIdx[0]
    [row_end, col_end] = cutIdx[1]

    #先判断切割出来的小图像是否为含有内容
    #从上到下扫描行像素之和，如果存在大于0且小于小图像列数，则小图像不为空
    isNotEmpty = 'False'
    row = row_start
    while(row < row_end):
        sumPiexlRow = np.sum(img_binary[row, col_start:col_end])
        if (sumPiexlRow > 0 and sumPiexlRow<np.abs(col_end - col_start - 5)):
            isNotEmpty = 'True'
            break
        row = row + 1

    #处理左右两边的边框，使得两边留些空白，不至于切到线
    while(np.sum(img_binary[row_start:row_end, col_start])>0):
        col_start = col_start - 2
    while(np.sum(img_binary[row_start:row_end, col_end])>0):
        col_end = col_end + 2

    #保证上下高度都是max_height
    dis = max_height - (row_end - row_start)
    if(dis>0):
        row_start = row_start - round(dis/2)
        row_end = row_end + (dis - round(dis/2))

    #切割并保存图像
    if(isTitle is True ):
        smallPieceImg = img_src[row_start:row_end, col_start:col_end, :]
    else:
        smallPieceImg = img_src[row_start:row_end, col_start:col_end, :]
        smallPieceimg_binary = img_binary[row_start:row_end, col_start:col_end]
        for row in range(smallPieceimg_binary.shape[0]):
            if(np.sum(smallPieceimg_binary[row, :])==smallPieceimg_binary.shape[1]):
                smallPieceImg[row,:,:]=img_src[0,0,:]
    if(isNotEmpty == 'True'):
        cv2.imwrite(imgSaveFullPathName, smallPieceImg)
    return isNotEmpty


def splitImgToDir(imageName, srcImgName, imgSuffix, img_binary, resultPath, xmlPath, titleIdx, titleInfoIdx, bodyIdx, max_height):
    """
    # 函数名称: splitImgToDir(imageName, resultPath, titleIdx, titleInfoIdx, bodyIdx, max_height)
    # 参数：
    # imageName: 图像文件路径及名称；
    # resultPath: 切割小图像块的保存位置；
    # titleIdx: 标题索引；
    # titleInfoIdx: 标题信息索引；
    # bodyIdx: 主体表格内容索引；
    # max_height: 截取图像的高度；
    # 返回值：
    #     无
    """

    img_src = cv2.imread(imageName)
    if(os.path.exists(resultPath) is False):
        os.makedirs(resultPath)
    #在内存中创建一个空的xml文档,这个xml用于记录图像信息
    doc = xml.dom.minidom.Document()
    #创建一个根节点Image对象
    root = doc.createElement('Image')
    #设置根节点Image的name属性
    root.setAttribute('name', srcImgName+imgSuffix)
    #将根节点添加到文档对象中
    doc.appendChild(root)
    #创建ImageHead节点，为root的子节点
    nodeHead = doc.createElement('ImageHead')
    nodeTitle = doc.createElement('Title') #创建Title节点，为ImageHead的子节点
    nodeTitleInfo = doc.createElement('TitleInfo') #创建TitleInfo节点，为ImageHead的子节点
    #创建ImageBody节点，为root的子节点
    nodeBody = doc.createElement('ImageBody')
    nodeBodyInfo = doc.createElement('ImageBodyInfo')#创建ImageBodyInfo节点，为ImageBody的子节点
    firstKeyCol = bodyIdx[0][0][0][1]
    pixelPerSpace = (bodyIdx[1][0][0][1] - firstKeyCol)  #计算每个缩进占多少个像素

    #[[startRowValue, startColValue], [endRowValue, endColValue]] = bodyIdx[1][0]
##############################################开始截取图像##############################################
##############################################开始截取图像##############################################

##############################################截取标题##############################################
    [row_start, col_start] = titleIdx[0]
    [row_end, col_end] = titleIdx[1]
    #保存图像
    titleImgSaveName = srcImgName+"_" + "000" + "_0_H" + imgSuffix
    titleImgSaveFullPathName = os.path.join(resultPath, titleImgSaveName)
    isNotEmpty = savaCutImg(max_height, img_src, img_binary, titleImgSaveFullPathName, titleIdx, True)

    #写XML文件
    nodeTitleName = doc.createElement('imageName')
    nodeTitleName.setAttribute('description', 'the name of cutted image')
    nodeTitleName.appendChild(doc.createTextNode(titleImgSaveName)) #imageName内容为保存的图像名
    nodeLeftUpIdx = doc.createElement('leftUpIndex')
    nodeLeftUpIdx.setAttribute('description', 'left-up coordinate')
    nodeLeftUpIdx.appendChild(doc.createTextNode('[%d,%d]'%(row_start, col_start))) #imageName内容为保存的图像名
    nodeRightDownIdx = doc.createElement('rightDownIndex')
    nodeRightDownIdx.setAttribute('description', 'right-down coordinate')
    nodeRightDownIdx.appendChild(doc.createTextNode('[%d,%d]'%(row_end, col_end))) #imageName内容为保存的图像名
    nodeTitleContent = doc.createElement('content')
    nodeTitleContent.appendChild(doc.createTextNode(''))
    nodeIsNotEmpty = doc.createElement('isNotEmpty')
    nodeIsNotEmpty.appendChild(doc.createTextNode(isNotEmpty))
    #添加4个节点到Title节点下
    nodeTitle.appendChild(nodeTitleName)
    nodeTitle.appendChild(nodeLeftUpIdx)
    nodeTitle.appendChild(nodeRightDownIdx)
    nodeTitle.appendChild(nodeTitleContent)
    nodeTitle.appendChild(nodeIsNotEmpty)
    #添加Title节点到ImageHead节点下
    nodeHead.appendChild(nodeTitle)

##############################################截取标题信息##############################################
##############################################截取标题信息##############################################
    #titleInfoIdx = [
    #                [ [[leftStartRowKey1, leftStartColKey1], [leftEndRowKey1, leftEndColKey1]],
    #                  [[leftStartRowValue1, leftStartColValue1], [leftEndRowValue1, leftEndColValue1]] ],
    #           ... ]
    orderNum = 1
    for idx in range(len(titleInfoIdx)):
        #读取左上和右下坐标
        [[startRowKey, startColKey], [endRowKey, endColKey]] = titleInfoIdx[idx][0]
        [[startRowValue, startColValue], [endRowValue, endColValue]] = titleInfoIdx[idx][1]

        orderNumStr = "00%d"%orderNum if orderNum<10 else "0%d"%orderNum if orderNum<100 else "%d"%orderNum
        orderNum = orderNum + 1

        #截取左侧单元格的字符串，并定义保存名字
        titleInfoKeyImgSaveName = srcImgName+"_" + orderNumStr + "_0_H" + imgSuffix
        titleInfoKeyImgSaveFullPathName = os.path.join(resultPath, titleInfoKeyImgSaveName)
        isNotEmpty = savaCutImg(max_height, img_src, img_binary, titleInfoKeyImgSaveFullPathName, titleInfoIdx[idx][0])

        #将信息写入xml文件
        #创建TitleInfo节点，为ImageHead的子节点
        nodeTitleInfoEle = doc.createElement('TitleInfoEle')
        nodeTitleInfoKeyEle = doc.createElement('TitleInfoKeyEle') #创建TitleInfoKeyEle节点
        #创建TitleInfoKeyEle节点的子节点
        nodeTitleInfoName = doc.createElement('imageName')
        nodeTitleInfoName.setAttribute('description', 'the name of cutted image')
        nodeTitleInfoName.appendChild(doc.createTextNode(titleInfoKeyImgSaveName)) #imageName内容为保存的图像名
        nodeLeftUpIdx = doc.createElement('leftUpIndex')
        nodeLeftUpIdx.setAttribute('description', 'left-up coordinate')
        nodeLeftUpIdx.appendChild(doc.createTextNode('[%d,%d]'%(startRowKey, startColKey))) #imageName内容为保存的图像名
        nodeRightDownIdx = doc.createElement('rightDownIndex')
        nodeRightDownIdx.setAttribute('description', 'right-down coordinate')
        nodeRightDownIdx.appendChild(doc.createTextNode('[%d,%d]'%(endRowKey, endColKey))) #imageName内容为保存的图像名
        nodeTitleContent = doc.createElement('content')
        nodeTitleContent.appendChild(doc.createTextNode(''))
        nodeIsNotEmpty = doc.createElement('isNotEmpty')
        nodeIsNotEmpty.appendChild(doc.createTextNode(isNotEmpty))
        #将以上4个节点添加进TitleInfoKeyEle节点下
        nodeTitleInfoKeyEle.appendChild(nodeTitleInfoName)
        nodeTitleInfoKeyEle.appendChild(nodeLeftUpIdx)
        nodeTitleInfoKeyEle.appendChild(nodeRightDownIdx)
        nodeTitleInfoKeyEle.appendChild(nodeTitleContent)
        nodeTitleInfoKeyEle.appendChild(nodeIsNotEmpty)

        # 截取右侧单元格的字符串，并定义保存名字
        titleInfoValueImgSaveName = srcImgName + "_" + orderNumStr + "_1_H" + imgSuffix
        titleInfoValueImgSaveFullPathName = os.path.join(resultPath, titleInfoValueImgSaveName)
        isNotEmpty = savaCutImg(max_height, img_src, img_binary, titleInfoValueImgSaveFullPathName, titleInfoIdx[idx][1])

        #创建TitleInfoValueEle节点，为ImageHead的子节点
        nodeTitleInfoValueEle = doc.createElement('TitleInfoValueEle')
        nodeTitleInfoName = doc.createElement('imageName')
        nodeTitleInfoName.setAttribute('description', 'the name of cutted image')
        nodeTitleInfoName.appendChild(doc.createTextNode(titleInfoValueImgSaveName)) #imageName内容为保存的图像名
        nodeLeftUpIdx = doc.createElement('leftUpIndex')
        nodeLeftUpIdx.setAttribute('description', 'left-up coordinate')
        nodeLeftUpIdx.appendChild(doc.createTextNode('[%d,%d]'%(startRowValue, startColValue))) #imageName内容为保存的图像名
        nodeRightDownIdx = doc.createElement('rightDownIndex')
        nodeRightDownIdx.setAttribute('description', 'right-down coordinate')
        nodeRightDownIdx.appendChild(doc.createTextNode('[%d,%d]'%(endRowValue, endColValue))) #imageName内容为保存的图像名
        nodeTitleContent = doc.createElement('content')
        nodeTitleContent.appendChild(doc.createTextNode(''))
        nodeIsNotEmpty = doc.createElement('isNotEmpty')
        nodeIsNotEmpty.appendChild(doc.createTextNode(isNotEmpty))
        #将以上4个节点添加进TitleInfoValueEle节点下
        nodeTitleInfoValueEle.appendChild(nodeTitleInfoName)
        nodeTitleInfoValueEle.appendChild(nodeLeftUpIdx)
        nodeTitleInfoValueEle.appendChild(nodeRightDownIdx)
        nodeTitleInfoValueEle.appendChild(nodeTitleContent)
        nodeTitleInfoValueEle.appendChild(nodeIsNotEmpty)
        #将nodeTitleInfoKeyEle和TitleInfoValueEle节点添加到nodeTitleInfo节点下
        nodeTitleInfoEle.appendChild(nodeTitleInfoKeyEle)
        nodeTitleInfoEle.appendChild(nodeTitleInfoValueEle)
        nodeTitleInfo.appendChild(nodeTitleInfoEle)
    #添加TitleInfo节点到ImageHead节点下
    nodeHead.appendChild(nodeTitleInfo)

##############################################截取主体表格信息##############################################
##############################################截取主体表格信息##############################################

    #bodyIdx= [
    # [ [[startKeyRow1, startKeyCol1],[endKeyRow1, endKeyCol1]],
    #   [[startValueRow1, startValueCol1],[endValueRow1, endValueCol1]] ],
    #   ... ]
    for idx in range(len(bodyIdx)):
        #读取左上和右下坐标
        [[startRowKey, startColKey], [endRowKey, endColKey]] = bodyIdx[idx][0]
        [[startRowValue, startColValue], [endRowValue, endColValue]] = bodyIdx[idx][1]
        orderNumStr = "00%d"%orderNum if orderNum<10 else "0%d"%orderNum if orderNum<100 else "%d"%orderNum
        orderNum = orderNum + 1

        #定义左侧单元格保存的图像名及路径并保存图像
        bodyInfoKeyImgSaveName = srcImgName + "_" + orderNumStr + "_0_B" + imgSuffix
        bodyInfoKeyImgSaveFullPathName = os.path.join(resultPath, bodyInfoKeyImgSaveName)
        isNotEmpty = savaCutImg(max_height, img_src, img_binary, bodyInfoKeyImgSaveFullPathName, bodyIdx[idx][0])

        #将图像信息写入xml文件
        nodeBodyInfoEle = doc.createElement('BodyInfoEle') #创建BodyInfoKeyEle节点
        nodeBodyInfoKeyEle = doc.createElement('BodyInfoKeyEle') #创建BodyInfoKeyEle节点
        #创建BodyInfoKeyEle节点的子节点
        nodeBodyInfoName = doc.createElement('imageName')
        nodeBodyInfoName.setAttribute('description', 'the name of cutted image')
        nodeBodyInfoName.appendChild(doc.createTextNode(bodyInfoKeyImgSaveName)) #imageName内容为保存的图像名
        nodeLeftUpIdx = doc.createElement('leftUpIndex')
        nodeLeftUpIdx.setAttribute('description', 'left-up coordinate')
        nodeLeftUpIdx.appendChild(doc.createTextNode('[%d,%d]'%(startRowKey, startColKey))) #imageName内容为保存的图像名
        nodeRightDownIdx = doc.createElement('rightDownIndex')
        nodeRightDownIdx.setAttribute('description', 'right-down coordinate')
        nodeRightDownIdx.appendChild(doc.createTextNode('[%d,%d]'%(endRowKey, endColKey))) #imageName内容为保存的图像名
        countSpaceNum = round((startColKey-firstKeyCol)/pixelPerSpace)
        nodeTabNum = doc.createElement('tabNum')
        nodeTabNum.setAttribute('description', 'number of tabs at begin place')
        nodeTabNum.appendChild(doc.createTextNode('%d' %(countSpaceNum)))
        nodeBodyContent = doc.createElement('content')
        nodeBodyContent.appendChild(doc.createTextNode(''))
        nodeIsNotEmpty = doc.createElement('isNotEmpty')
        nodeIsNotEmpty.appendChild(doc.createTextNode(isNotEmpty))
        #将以上4个节点添加进BodyInfoKeyEle节点下
        nodeBodyInfoKeyEle.appendChild(nodeBodyInfoName)
        nodeBodyInfoKeyEle.appendChild(nodeLeftUpIdx)
        nodeBodyInfoKeyEle.appendChild(nodeRightDownIdx)
        nodeBodyInfoKeyEle.appendChild(nodeTabNum)
        nodeBodyInfoKeyEle.appendChild(nodeBodyContent)
        nodeBodyInfoKeyEle.appendChild(nodeIsNotEmpty)
        nodeBodyInfoValueEle = doc.createElement('BodyInfoValueEle') #创建BodyInfoValueEle节点

        # 定义右侧单元格保存的图像名及路径并保存图像
        bodyInfoValueImgSaveName = srcImgName + "_" + orderNumStr + "_1_B" + imgSuffix
        bodyInfoValueImgSaveFullPathName = os.path.join(resultPath, bodyInfoValueImgSaveName)
        isNotEmpty = savaCutImg(max_height, img_src, img_binary, bodyInfoValueImgSaveFullPathName, bodyIdx[idx][1])
        #创建BodyInfoValueEle节点的子节点
        nodeBodyInfoName = doc.createElement('imageName')
        nodeBodyInfoName.setAttribute('description', 'the name of cutted image')
        nodeBodyInfoName.appendChild(doc.createTextNode(bodyInfoValueImgSaveName)) #imageName内容为保存的图像名
        nodeLeftUpIdx = doc.createElement('leftUpIndex')
        nodeLeftUpIdx.setAttribute('description', 'left-up coordinate')
        nodeLeftUpIdx.appendChild(doc.createTextNode('[%d,%d]'%(startRowValue, startColValue))) #imageName内容为保存的图像名
        nodeRightDownIdx = doc.createElement('rightDownIndex')
        nodeRightDownIdx.setAttribute('description', 'right-down coordinate')
        nodeRightDownIdx.appendChild(doc.createTextNode('[%d,%d]'%(endRowValue, endColValue))) #imageName内容为保存的图像名
        nodeBodyContent = doc.createElement('content')
        nodeBodyContent.appendChild(doc.createTextNode(''))
        nodeIsNotEmpty = doc.createElement('isNotEmpty')
        nodeIsNotEmpty.appendChild(doc.createTextNode(isNotEmpty))
        #将以上4个节点添加进BodyInfoValueEle节点下
        nodeBodyInfoValueEle.appendChild(nodeBodyInfoName)
        nodeBodyInfoValueEle.appendChild(nodeLeftUpIdx)
        nodeBodyInfoValueEle.appendChild(nodeRightDownIdx)
        nodeBodyInfoValueEle.appendChild(nodeBodyContent)
        nodeBodyInfoValueEle.appendChild(nodeIsNotEmpty)
        #将nodeBodyInfoKeyEle和BodyInfoValueEle节点添加到nodeBodyInfo节点下
        nodeBodyInfoEle.appendChild(nodeBodyInfoKeyEle)
        nodeBodyInfoEle.appendChild(nodeBodyInfoValueEle)
        nodeBodyInfo.appendChild(nodeBodyInfoEle)
    #将nodeBodyInfo添加到nodeBody节点下
    nodeBody.appendChild(nodeBodyInfo)
    #将nodeHead和nodeBody添加到root节点下
    root.appendChild(nodeHead)
    root.appendChild(nodeBody)

    #写xml文件
    xmlFullPathName = os.path.join(xmlPath, srcImgName + '.xml')
    fp = open(xmlFullPathName, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")


def calOneImage(imageName, resultPath, xmlPath):

    # windows下用\\，linux要改为/
    srcImgName = os.path.basename(imageName).split('.')[0]
    # srcImgName = imageName[imageName.rfind("\\") + 1:imageName.rfind(".")]  # 获取图像名称
    imgSuffix = imageName[imageName.rfind("."):len(imageName)]  # 获取图像后缀
    img_binary = calimg_binary(imageName)
    rowIdx,firstRowIdx,colIdx = findLinePos(img_binary)
    titleIdx, titleInfoIdx, maxTitleHeight = findHeadPos(img_binary, firstRowIdx)
    bodyIdx,maxBodyHeight = findBodyIdx(img_binary, rowIdx, colIdx)
    max_height = np.maximum(maxTitleHeight, maxBodyHeight)
    if(max_height % 2==1):
        max_height = max_height+1
    splitImgToDir(imageName, srcImgName, imgSuffix, img_binary, resultPath, xmlPath, titleIdx, titleInfoIdx, bodyIdx, max_height)

def calImageFromDir(imageDirPath, resultPath, xmlPath):
    for dirpath, dirnames, filenames in os.walk(imageDirPath):
        for imagePureName in filenames:
            print('processiong %s' % imagePureName)
            imageName = imageDirPath + imagePureName

            srcImgName = imageName[imageName.rfind("\\") + 1:imageName.rfind(".")]  # 获取图像名称
            imgSuffix = imageName[imageName.rfind("."):len(imageName)]  # 获取图像后缀
            resultPath = resultPath + srcImgName + imgSuffix + ".Dir\\"

            img_binary = calimg_binary(imageName)
            rowIdx, firstRowIdx, colIdx = findLinePos(img_binary)
            titleIdx, titleInfoIdx, maxTitleHeight = findHeadPos(img_binary, firstRowIdx)
            bodyIdx, maxBodyHeight = findBodyIdx(img_binary, rowIdx, colIdx)
            max_height = np.maximum(maxTitleHeight, maxBodyHeight)
            if (max_height % 2 == 1):
                max_height = max_height + 1

            splitImgToDir(imageName, srcImgName, imgSuffix, img_binary, resultPath, xmlPath, titleIdx, titleInfoIdx, bodyIdx, max_height)
