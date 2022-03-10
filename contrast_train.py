import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance,
    # but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_contrast", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_contrast", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--bg_threshold", default=0.20, type=float)
    # parser.add_argument("--saved_dir", default='VOC2012', type=str)

    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net')()

    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                          saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d

        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss',
                                     'loss_cls',
                                     'loss_er',
                                     'loss_ecr',
                                     'loss_nce',
                                     'loss_intra_nce',
                                     'loss_cross_nce',
                                     'loss_cross_nce2')

    timer = pyutils.Timer("Session started: ")

    # Prototype
    PROTO1 = F.normalize(torch.rand(21, 128).cuda(), p=2, dim=1)
    PROTO2 = F.normalize(torch.rand(21, 128).cuda(), p=2, dim=1)

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):
            # scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1,
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]

            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            cam1, cam_rv1, f_proj1, cam_rv1_down = model(img1)
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])

            cam1 = F.interpolate(visualization.max_norm(cam1),
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True) * label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),
                                    size=(128, 128),
                                    mode='bilinear',
                                    align_corners=True) * label

            cam2, cam_rv2, f_proj2, cam_rv2_down = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label
            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))

            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2

            ################################################################################
            ###################### Contrastive Learning ####################################
            ################################################################################
            f_proj1 = F.interpolate(f_proj1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            cam_rv1_down = F.interpolate(cam_rv1_down, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            cam_rv2_down = cam_rv2_down
            with torch.no_grad():
                # source
                fea1 = f_proj1.detach()
                c_fea1 = fea1.shape[1]
                cam_rv1_down = F.relu(cam_rv1_down.detach())
                # ~(0,1)
                n1, c1, h1, w1 = cam_rv1_down.shape
                max1 = torch.max(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
                min1 = torch.min(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
                cam_rv1_down[cam_rv1_down < min1 + 1e-5] = 0.
                norm_cam1 = (cam_rv1_down - min1 - 1e-5) / (max1 - min1 + 1e-5)
                cam_rv1_down = norm_cam1
                cam_rv1_down[:, 0, :, :] = args.bg_threshold
                scores1 = F.softmax(cam_rv1_down * label, dim=1)

                pseudo_label1 = scores1.argmax(dim=1, keepdim=True)
                n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape
                scores1 = scores1.transpose(0, 1)
                fea1 = fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)

                top_values, top_indices = torch.topk(cam_rv1_down.transpose(0, 1).reshape(c_sc1, -1),
                                                     k=h_sc1 * w_sc1 // 8, dim=-1)
                prototypes1 = torch.zeros(c_sc1, c_fea1).cuda()  # [21, 128]
                for i in range(c_sc1):
                    top_fea = fea1[top_indices[i]]
                    prototypes1[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
                # L2 Norm
                prototypes1 = F.normalize(prototypes1, dim=-1)

                # target
                fea2 = f_proj2.detach()
                c_fea2 = fea2.shape[1]

                cam_rv2_down = F.relu(cam_rv2_down.detach())
                n2, c2, h2, w2 = cam_rv2_down.shape
                max2 = torch.max(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
                min2 = torch.min(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
                cam_rv2_down[cam_rv2_down < min2 + 1e-5] = 0.
                norm_cam2 = (cam_rv2_down - min2 - 1e-5) / (max2 - min2 + 1e-5)

                cam_rv2_down = norm_cam2
                cam_rv2_down[:, 0, :, :] = args.bg_threshold

                scores2 = F.softmax(cam_rv2_down * label, dim=1)
                pseudo_label2 = scores2.argmax(dim=1, keepdim=True)

                n_sc2, c_sc2, h_sc2, w_sc2 = scores2.shape
                scores2 = scores2.transpose(0, 1)
                fea2 = fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)
                top_values2, top_indices2 = torch.topk(cam_rv2_down.transpose(0, 1).reshape(c_sc2, -1),
                                                       k=h_sc2 * w_sc2 // 8, dim=-1)
                prototypes2 = torch.zeros(c_sc2, c_fea2).cuda()

                for i in range(c_sc2):
                    top_fea2 = fea2[top_indices2[i]]
                    prototypes2[i] = torch.sum(top_values2[i].unsqueeze(-1) * top_fea2, dim=0) / torch.sum(
                        top_values2[i])

                # L2 Norm
                prototypes2 = F.normalize(prototypes2, dim=-1)

            # for source
            n_f, c_f, h_f, w_f = f_proj1.shape
            f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj1 = F.normalize(f_proj1, dim=-1)
            pseudo_label1 = pseudo_label1.reshape(-1)
            positives1 = prototypes2[pseudo_label1]
            negitives1 = prototypes2

            # for target
            n_f, c_f, h_f, w_f = f_proj2.shape
            f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj2 = F.normalize(f_proj2, dim=-1)
            pseudo_label2 = pseudo_label2.reshape(-1)
            positives2 = prototypes1[pseudo_label2]
            negitives2 = prototypes1

            # 1. cross-view contrastive learning
            # 1.1 cross-prototype
            A1 = torch.exp(torch.sum(f_proj1 * positives1, dim=-1) / 0.1)
            A2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
            loss_nce1 = torch.mean(-1 * torch.log(A1 / A2))

            A3 = torch.exp(torch.sum(f_proj2 * positives2, dim=-1) / 0.1)
            A4 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            loss_nce2 = torch.mean(-1 * torch.log(A3 / A4))

            loss_cross_nce = 0.1 * (loss_nce1 + loss_nce2) / 2

            # 1.2 cross-pseudo-label
            A1_view1 = torch.exp(torch.sum(f_proj1 * positives2, dim=-1) / 0.1)
            A2_view1 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives2.transpose(0, 1)) / 0.1), dim=-1)
            loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view1 / A2_view1))

            A3_view2 = torch.exp(torch.sum(f_proj2 * positives1, dim=-1) / 0.1)
            A4_view2 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives1.transpose(0, 1)) / 0.1), dim=-1)

            loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

            loss_cross_nce2 = 0.1 * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2

            # 2. intra-view contrastive learning
            # semi-hard prototype mining
            positives_intra1 = prototypes1[pseudo_label1]
            negitives_intra1 = prototypes1
            similarity_intra1 = (torch.sum(f_proj1 * positives_intra1, dim=-1) + 1) / 2.
            A1_intra_view1 = torch.exp(torch.sum(f_proj1 * positives_intra1, dim=-1) / 0.1)
            neg_scores = torch.matmul(f_proj1, negitives_intra1.transpose(0, 1))
            with torch.no_grad():  # random 50%
                random_indices = torch.tensor([random.sample(range(21), 10) for _ in range(n_f * h_f * w_f)]).long()
            with torch.no_grad():
                _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
                lower_indices = lower_indices[:, 3:]
            negitives_intra1 = negitives_intra1.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
            random_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), random_indices]
            lower_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), lower_indices]
            negitives_intra1 = torch.cat([positives_intra1.unsqueeze(1), lower_negitives_intra1], dim=1)
            A2_intra_view1 = torch.sum(torch.exp(torch.matmul(f_proj1.unsqueeze(1), negitives_intra1.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)
            loss_intra_nce1 = torch.zeros(1).cuda()
            C = 0
            exists = np.unique(pseudo_label1.cpu().numpy()).tolist()
            # hard pixel sampling
            for i_ in range(21):  # for each class
                if not i_ in exists:
                    continue
                C += 1
                A1_intra_view1_class = A1_intra_view1[pseudo_label1 == i_]
                A2_intra_view1_class = A2_intra_view1[pseudo_label1 == i_]
                similarity_intra1_class = similarity_intra1[pseudo_label1 == i_]
                len_class = A1_intra_view1_class.shape[0]
                if len_class < 2:
                    continue

                with torch.no_grad():  # random 50%
                    random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
                random_A1_intra_view1 = A1_intra_view1_class[random_indices]  # (n, hw//2)
                random_A2_intra_view1 = A2_intra_view1_class[random_indices]

                with torch.no_grad():
                    _, lower_indices = torch.topk(similarity_intra1_class, k=int(len_class * 0.6), largest=False)
                    lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

                lower_A1_intra_view1 = A1_intra_view1_class[lower_indices]
                lower_A2_intra_view1 = A2_intra_view1_class[lower_indices]

                A1_intra_view1_class = torch.cat([random_A1_intra_view1, lower_A1_intra_view1], dim=0)  # (hw)
                A2_intra_view1_class = torch.cat([random_A2_intra_view1, lower_A2_intra_view1], dim=0)
                A1_intra_view1_class = A1_intra_view1_class.reshape(-1)
                A2_intra_view1_class = A2_intra_view1_class.reshape(-1)
                loss_intra_nce1 += torch.mean(-1 * torch.log(A1_intra_view1_class / A2_intra_view1_class))

            # mean over classes
            loss_intra_nce1 = loss_intra_nce1 / C

            # for target
            # semi-hard prototype mining
            positives_intra2 = prototypes2[pseudo_label2]
            negitives_intra2 = prototypes2
            similarity_intra2 = (torch.sum(f_proj2 * positives_intra2, dim=-1) + 1) / 2.
            A3_intra_view2 = torch.exp(torch.sum(f_proj2 * positives_intra2, dim=-1) / 0.1)
            neg_scores = torch.matmul(f_proj2, negitives_intra2.transpose(0, 1))  # (n*h*w, 21)
            with torch.no_grad():
                random_indices = torch.tensor([random.sample(range(21), 10) for _ in range(n_f * h_f * w_f)]).long()
            with torch.no_grad():
                _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
                lower_indices = lower_indices[:, 3:]
            negitives_intra2 = negitives_intra2.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
            random_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), random_indices]
            lower_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), lower_indices]
            negitives_intra2 = torch.cat([positives_intra2.unsqueeze(1), lower_negitives_intra2], dim=1)
            A4_intra_view2 = torch.sum(torch.exp(torch.matmul(f_proj2.unsqueeze(1), negitives_intra2.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)
            loss_intra_nce2 = torch.zeros(1).cuda()
            C = 0
            exists = np.unique(pseudo_label2.cpu().numpy()).tolist()
            # hard pixel sampling
            for i_ in range(21):
                if not i_ in exists:
                    continue
                C += 1
                A3_intra_view2_class = A3_intra_view2[pseudo_label2 == i_]
                A4_intra_view2_class = A4_intra_view2[pseudo_label2 == i_]
                similarity_intra2_class = similarity_intra2[pseudo_label2 == i_]
                len_class = A3_intra_view2_class.shape[0]

                if len_class < 2:
                    continue

                with torch.no_grad():
                    random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
                random_A3_intra_view2 = A3_intra_view2_class[random_indices]  # (n, hw//2)
                random_A4_intra_view2 = A4_intra_view2_class[random_indices]
                with torch.no_grad():  # lowest 50%
                    _, lower_indices = torch.topk(similarity_intra2_class, k=int(len_class * 0.6), largest=False)
                    lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

                lower_A3_intra_view2 = A3_intra_view2_class[lower_indices]
                lower_A4_intra_view2 = A4_intra_view2_class[lower_indices]
                A3_intra_view2_class = torch.cat([random_A3_intra_view2, lower_A3_intra_view2], dim=0)
                A4_intra_view2_class = torch.cat([random_A4_intra_view2, lower_A4_intra_view2], dim=0)
                A3_intra_view2_class = A3_intra_view2_class.reshape(-1)
                A4_intra_view2_class = A4_intra_view2_class.reshape(-1)

                loss_intra_nce2 += torch.mean(-1 * torch.log(A3_intra_view2_class / A4_intra_view2_class))

            # mean over classes
            loss_intra_nce2 = loss_intra_nce2 / C

            loss_intra_nce = 0.1 * (loss_intra_nce1 + loss_intra_nce2) / 2

            # 3. total nce loss
            loss_nce = loss_cross_nce + loss_cross_nce2 + loss_intra_nce

            # 4. total loss
            loss = loss_cls + loss_er + loss_ecr + loss_nce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_intra_nce': loss_intra_nce.item(),
                           'loss_cross_nce': loss_cross_nce.item(),
                           'loss_cross_nce2': loss_cross_nce2.item()})

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d | ' % (optimizer.global_step - 1, max_step),
                      'loss: %.4f | loss_cls: %.4f | loss_er: %.4f | loss_ecr: %.4f | '
                      'loss_nce: %.4f | loss_intra_nce: %.4f | loss_cross_nce: %.4f | loss_cross_nce2: %.4f'
                      % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_nce', 'loss_intra_nce',
                                      'loss_cross_nce', 'loss_cross_nce2'),
                      'imps:%.1f | ' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s | ' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item(),
                             'loss_nce': loss_nce.item(),
                             'loss_intra_nce': loss_intra_nce.item(),
                             'loss_inter_nce': loss_cross_nce.item()}

                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)

        else:
            print('')
            timer.reset_stage()
    print(args.session_name)

    torch.save(model.module.state_dict(), args.session_name + '.pth')
