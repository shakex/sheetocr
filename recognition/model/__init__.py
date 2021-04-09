# -*- coding: UTF-8 -*-
import sys
from recognition.model.crnn import CRNN
import params


def get_model(model_arch, n_classes):
    name = model_arch
    model = _get_model_instance(name)

    if name=='crnn':
        model = model(imgH=params.imgH, nc=params.nc, nclass=n_classes, nh=params.nh)

    return model


def _get_model_instance(name):
    try:
        return {
            "crnn": CRNN,
        }[name]
    except:
        raise ("Model {} not available".format(name))

