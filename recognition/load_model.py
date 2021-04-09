import params
import torch
import sys
from recognition.model import get_model

def load_model(model_arch, model_path):
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    try:
        n_classes = int((len(params.alphabet) + 1) / 2) + 1
        model = get_model(model_arch, n_classes=n_classes).to(device)
        if params.multi_gpu:
            model = torch.nn.DataParallel(model)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)
    except:
        print("Model Error: Model \'" + model_arch + "\' import failed, please check the model file.")
        sys.exit()
    
    return model, device