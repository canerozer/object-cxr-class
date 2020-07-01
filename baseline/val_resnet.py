import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import pandas as pd
import argparse
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, auc

from dataset import ForeignObjectDataset
from utils import DictAsMember
from models import _get_classification_model


def eval():
    model_ft.eval()

    val_pred = []
    val_label = []
    running_corrects = 0.
    n_samples = 0

    for batch_i, (image, label, width, height) in enumerate(data_loader_val):
        image = image.to(device, dtype=torch.float32)

        output = model_ft(image)
        logit, pred = torch.max(output, 1)
        probs = F.softmax(output, dim=1)
        
        val_pred.append(probs.detach().squeeze().cpu().numpy()[1])
        val_label.append(label.cpu().numpy())

        running_corrects += torch.sum(pred.cpu() == label)
        n_samples += len(output)

    val_label = np.concatenate([label for label in val_label])
    acc = running_corrects.double() / n_samples  
    auc_init = roc_auc_score(val_label, val_pred)

    print('Val acc: %.4f' % acc, '| Val auc: %.4f' % auc_init)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CXR Object Localization')
    parser.add_argument('--yaml', type=str, metavar='YAML',
                        default="configs/faster_rcnn",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        exp_args = DictAsMember(yaml.safe_load(f))

    data_dir = 'data/'

    device = torch.device('cuda:0')
    num_classes = 2  # object (foreground); background

    meta_dev = data_dir + 'dev.csv'
    labels_dev = pd.read_csv(meta_dev, na_filter=False)

    print(f'{len(os.listdir(data_dir + "dev"))} pics in {data_dir}dev/')

    img_class_dict_dev = dict(zip(labels_dev.image_name,
                                  labels_dev.annotation))

    input_size = (exp_args.MODEL.INPUT_SIZE, exp_args.MODEL.INPUT_SIZE)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([transforms.Resize(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=img_mean,
                                                               std=img_std)]
                                         )

    dataset_dev = ForeignObjectDataset(datafolder=data_dir+'dev/',
                                       datatype='dev',
                                       transform=data_transforms,
                                       labels_dict=img_class_dict_dev)

    data_loader_val = DataLoader(dataset_dev,
                                 batch_size=1,
                                 shuffle=False, num_workers=0)

    model_ft = _get_classification_model(exp_args.MODEL.N_CLASS,
                                         exp_args.MODEL.NAME)
    model_ft.to(device)

    path = os.path.join(exp_args.MODEL.SAVE_TO, exp_args.MODEL.NAME,
                        exp_args.NAME + ".pt")
    model_ft.load_state_dict(torch.load(path, map_location=device))

    eval()
