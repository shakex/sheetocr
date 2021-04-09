from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from recognition import utils
from recognition.sheetLoader import sheetLoader, randomSequentialSampler, alignCollate, resizeNormalize
import recognition.model.crnn as net
import params

# solve problem (macOS):
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
# parser.add_argument('-train', '--trainroot', required=True, help='path to train dataset')
# parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""


def data_loader():
    # train
    train_dataset = sheetLoader(root=params.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
                                               shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                                               collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW,
                                                                       keep_ratio=params.keep_ratio))

    # val
    # val_dataset = sheetLoader(root=params.valroot, transform=resizeNormalize((params.imgW, params.imgH)))
    val_dataset = sheetLoader(root=params.valroot)
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize,
                                             num_workers=int(params.workers), \
                                             collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW,
                                                                     keep_ratio=params.keep_ratio))

    return train_loader, val_loader


train_loader, val_loader = data_loader()

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init():
    nclass = int((len(params.alphabet) + 1) / 2) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass,
                    params.nh)  # nc: # number of input channels; nh:  size of the lstm hidden state
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))

    return crnn


crnn = net_init()
print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        because train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity=True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)


# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()  # The blobal loss_avg is used by train

    # max_iter = len(val_loader)
    max_iter = 10
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))

        if sim_preds == cpu_texts_decode[0]:
            n_correct += 1

        # for pred, target in sim_preds, cpu_texts_decode:
        #     if pred == target:
        #         n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    print('%-10s \n=> %-10s, gt: %-10s' % (raw_preds, sim_preds, cpu_texts_decode))
    # for raw_pred, pred, gt in list(zip(raw_preds, sim_preds, cpu_texts_decode)):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)  # l: length, t: label vector
    utils.loadData(text, t)
    utils.loadData(length, l)

    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    for epoch in range(params.nepoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            cost = train(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % params.valInterval == 0:
                val(crnn, criterion)

            # do checkpointing
            if i % params.saveInterval == 0:
                torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))
