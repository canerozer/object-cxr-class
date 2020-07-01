import math
import sys
import time
import torch
import torch.nn as nn

import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    my_logger=None, name=None, warm_up=True,
                    mode="detection"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if epoch == 0 and warm_up:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for d, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device)
        #images = [img for img in images]
        #images = torch.Tensor(np.array(images)).to(device)
        #targets = [True if len(t["labels"])>0 else False for t in targets]
        #targets = torch.Tensor(targets).to(device)

        output = model(images)
        logit, pred = torch.max(output, 1)

        loss = criterion(output, targets)

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_corrects = torch.sum(pred == targets.data)
        acc = running_corrects.double() / len(pred)

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if mode == "classification":
            metric_logger.update(acc=acc)
