import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from arguments import get_args
from augmentations import get_aug
from models import get_model
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(log_writer, log_file, device, args):
    iter_count = 0

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        pin_memory=True, drop_last=True, num_workers=args.workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=True, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.test_bs,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for epoch in range(0, args.train.stop_at_epoch):
        model.train()
        loss_list = []
        print("number of iters this epoch: {}".format(len(train_loader)))
        for idx, ((images1, images2), labels) in enumerate(train_loader):
            iter_count += 1
            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            loss_list.append(loss.item())

        model.eval()

        test_loss_list = []
        for idx, ((images1, images2), labels) in enumerate(test_loader):
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            test_loss = data_dict['loss'].mean()
            test_loss_list.append(test_loss.item())

        write_dict = {
            'epoch': epoch,
            'loss': sum(loss_list) / len(loss_list),
            'lr': lr_scheduler.get_lr(),
            'test_loss': sum(test_loss_list) / len(test_loss_list),
        }
        log_writer.writerow(write_dict)
        log_file.flush()

        if (epoch+1) % args.log_freq == 0:
            model_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

    # Save checkpoint
    model_path = os.path.join(ckpt_dir, f"latest_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoints", f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')