import os
import torch
import argparse
from torch.backends import cudnn

from util import pyutils

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import train_cls, train_eps, train_contrast
from module.validate import validate

cudnn.enabled = True
torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    # session
    parser.add_argument("--session", default="eps", type=str)

    # data
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--saliency_root", type=str)
    parser.add_argument("--train_list", default="data/voc12/train_aug_id.txt", type=str)
    parser.add_argument("--val_list", default="data/voc12/val_id.txt", type=str)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--iter_size", default=2, type=int)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--resize_size", default=(448, 768))

    # network
    parser.add_argument("--network", default="network.vgg16_cls", type=str)
    parser.add_argument("--weights", required=True, type=str, default='pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')

    # optimizer
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--loss_type", default='mse', type=str)
    parser.add_argument("--eval", type=bool)
    parser.add_argument("--num_sample", default=21, type=int)
    parser.add_argument("--max_iters", default=10000, type=int)

    # hyper-parameters for EPS
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)

    args = parser.parse_args()


    if 'cls' in args.network:
        args.network_type = 'cls'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    elif 'contrast' in args.network:
        args.network_type = 'contrast'
    else:
        raise Exception('No appropriate model type')

    return args


if __name__ == '__main__':

    # get arguments
    args = get_arguments()

    # set log
    args.log_folder = os.path.join('train_log', args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    pyutils.Logger(os.path.join(args.log_folder, 'log_cls.log'))
    print(vars(args))

    # load dataset
    train_loader, val_loader = get_dataloader(args)

    max_step = (len(open(args.train_list).read().splitlines()) // args.batch_size) * args.max_epoches

    # load network and its pre-trained model
    model = get_model(args)

    # set optimizer
    optimizer = get_optimizer(args, model, max_step)

    # evaluate
    if args.eval:
        validate(model, val_loader, 0, args)
        exit()

    # train
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    if args.network_type == 'cls':
        train_cls(train_loader, model, optimizer, max_step, args)
    elif args.network_type == 'eps':
        train_eps(train_loader, model, optimizer, max_step, args)
    elif args.network_type == 'contrast':
        train_contrast(train_loader, model, optimizer, max_step, args)

    else:
        raise Exception('No appropriate model type')