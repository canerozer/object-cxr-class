import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
#import lamp
import logging

from dataset import ForeignObjectDataset, retrieve_data_transforms
from engine import train_one_epoch
from utils import DictAsMember
from models import _get_classification_model


OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

def train():
    auc_max = 0.

    for epoch in range(exp_args.SOLVER.EPOCH):
        train_one_epoch(model_ft, optimizer, data_loader, device, epoch,
                        print_freq=20,# my_logger=my_logger,
                        name=exp_args.MODEL.NAME, mode="classification")
        auc_max = eval(epoch, auc_max)
        lr_scheduler.step()


def eval(epoch, auc_max):
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
    """
    val_pred_label = []
    for i in range(len(val_pred)):
        if val_pred[i] >= 0.5:
            val_pred_label.append(1)
        else:
            val_pred_label.append(0)
            
    number = 0
    
    for i in range(len(val_pred_label)):
        if val_pred_label[i] == val_label[i]:
            number += 1
    """
    val_label = np.concatenate([label for label in val_label])
    acc = running_corrects.double() / n_samples  
    auc_init = roc_auc_score(val_label, val_pred)

    #my_logger.scatter(auc_init, index=epoch, 
    #                  win="AUC", trace=exp_args.MODEL.NAME)
    #my_logger.scatter(acc, index=epoch,
    #                  win="ACC", trace=exp_args.MODEL.NAME)

    print('Epoch:', epoch, '| val acc: %.4f' % acc, '| val auc: %.4f' % auc_init)

    if auc_init > auc_max:
        auc_max = auc_init
        print('Best Epoch: ', epoch, '| val acc: %.4f' % acc, '| Best val auc: %.4f' % auc_max)
        os.makedirs(exp_args.MODEL.SAVE_TO + "/" + exp_args.MODEL.NAME,
                    exist_ok=True)
        torch.save(model_ft.state_dict(), 
                   exp_args.MODEL.SAVE_TO + "/" + exp_args.MODEL.NAME + \
                   "/" + exp_args.NAME + ".pt")
    return auc_max

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CXR Object Localization')
    parser.add_argument('--yaml', type=str, metavar='YAML',
                        default="configs/faster_rcnn",
                        help='Enter the path for the YAML config')
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        exp_args = DictAsMember(yaml.safe_load(f))

    np.random.seed(0)
    torch.manual_seed(0)

    #logging.setLoggerClass(lamp.DataLogger)
    #my_logger = logging.getLogger(__name__)

    #vis_loss_handler = lamp.VisdomScalarHandler(logging.DATA,
    #                                            overwrite_window=False,
    #                                            )
    #vis_auc_handler = lamp.VisdomScatterHandler(logging.DATA,
    #                                            overwrite_window=False,
    #                                            )
    #vis_acc_handler = lamp.VisdomScatterHandler(logging.DEBUG,
    #                                            overwrite_window=False,
    #                                            )

    #my_logger.addHandler(vis_loss_handler)
    #my_logger.addHandler(vis_auc_handler)
    #my_logger.addHandler(vis_acc_handler)
    #my_logger.setLevel(logging.DATA)

    # ─── DATA_DIR
    #     ├── train
    #     │   ├── #####.jpg
    #     │   └── ...
    #     ├── dev
    #     │   ├── #####.jpg
    #     │   └── ...
    #     ├── train.csv
    #     └── dev.csv
    data_dir = 'data/'

    device = torch.device('cuda:0')
    num_classes = 2  # object (foreground); background

    meta_train = data_dir + 'train.csv'
    meta_dev = data_dir + 'dev.csv'

    labels_tr = pd.read_csv(meta_train, na_filter=False)
    labels_dev = pd.read_csv(meta_dev, na_filter=False)

    print(f'{len(os.listdir(data_dir + "train"))} pics in {data_dir}train/')
    print(f'{len(os.listdir(data_dir + "dev"))} pics in {data_dir}dev/')

    print(labels_tr['annotation'])
    #labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)]
    #labels_tr = labels_tr.reset_index(drop=True)
    img_class_dict_tr = dict(zip(labels_tr.image_name,
                                 labels_tr.annotation))
    img_class_dict_dev = dict(zip(labels_dev.image_name,
                                  labels_dev.annotation))

    input_size = (exp_args.MODEL.INPUT_SIZE, exp_args.MODEL.INPUT_SIZE)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    input_size = (exp_args.MODEL.INPUT_SIZE, exp_args.MODEL.INPUT_SIZE)
    data_transforms = retrieve_data_transforms(exp_args.DATA_AUG, input_size)
    #data_transforms = transforms.Compose([transforms.Resize(input_size),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize(mean=img_mean,
    #                                                           std=img_std)]
    #                                     )

    dataset_train = ForeignObjectDataset(datafolder=data_dir+'train/',
                                         datatype='train',
                                         transform=data_transforms,
                                         labels_dict=img_class_dict_tr)
    dataset_dev = ForeignObjectDataset(datafolder=data_dir+'dev/',
                                       datatype='dev',
                                       transform=data_transforms,
                                       labels_dict=img_class_dict_dev)

    data_loader = DataLoader(dataset_train,
                             batch_size=exp_args.MODEL.BATCH_SIZE,
                             shuffle=True, num_workers=0)

    data_loader_val = DataLoader(dataset_dev,
                                 batch_size=1,
                                 shuffle=False, num_workers=0)

    model_ft = _get_classification_model(exp_args.MODEL.N_CLASS,
                                         exp_args.MODEL.NAME)
    model_ft.to(device)

    params = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = SGD(params,
                    lr=exp_args.SOLVER.INIT_LR,
                    momentum=exp_args.SOLVER.MOMENTUM,
                    weight_decay=exp_args.SOLVER.WEIGHT_DECAY)
    lr_scheduler = MultiStepLR(optimizer,
                               exp_args.SCHEDULER.IN_EVERY,
                               gamma=exp_args.SCHEDULER.GAMMA)

    train()

