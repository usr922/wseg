import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ClassificationDataset, ClassificationDatasetWithSaliency
from util import imutils
from util.imutils import Normalize


def get_dataloader(args):

    if args.network_type == 'cls':
        train_dataset = ClassificationDataset(
            args.train_list,
            img_root=args.data_root,
            transform=transforms.Compose([
                imutils.RandomResizeLong(args.resize_size[0], args.resize_size[1]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                np.asarray,
                Normalize(),
                imutils.RandomCrop(args.crop_size),
                imutils.HWC_to_CHW,
                torch.from_numpy
            ]))
    elif args.network_type == 'eps':
        train_dataset = ClassificationDatasetWithSaliency(
            args.train_list,
            img_root=args.data_root,
            saliency_root=args.saliency_root,
            crop_size=args.crop_size,
            resize_size=args.resize_size
        )
    elif args.network_type == 'eps_seam' or args.network_type == 'eps_seam_with_PCM':
        train_dataset = ClassificationDatasetWithSaliency(
            args.train_list,
            img_root=args.data_root,
            saliency_root=args.saliency_root,
            crop_size=args.crop_size,
            resize_size=args.resize_size
        )
    else:
        raise Exception("No appropriate train type")

    val_dataset = ClassificationDataset(
        args.val_list,
        img_root=args.data_root,
        transform=transforms.Compose([
                        np.asarray,
                        Normalize(),
                        imutils.CenterCrop(500),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
        ]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader
