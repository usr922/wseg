import os
import time
import imageio
import argparse
import importlib
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from torch.multiprocessing import Process

from util import imutils, pyutils
from util.imutils import HWC_to_CHW
from network.resnet38d import Normalize
from data.dataset import load_img_id_list, load_img_label_list_from_npy


start = time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", nargs='*', type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)
    parser.add_argument("--img_root", default='VOC2012', type=str)
    parser.add_argument("--crf", default=None, type=str)
    parser.add_argument("--crf_alpha", nargs='*', type=int)
    parser.add_argument("--crf_t", nargs='*', type=int)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--cam_png", default=None, type=str)
    parser.add_argument("--thr", default=0.20, type=float)
    args = parser.parse_args()

    # model information
    if 'cls' in args.network:
        args.network_type = 'cls'
    elif 'eps_seam' in args.network:
        args.network_type = 'eps_seam'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    else:
        raise Exception('No appropriate model type')

    # save path
    args.save_type = list()
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)
    if args.cam_png is not None:
        os.makedirs(args.cam_png, exist_ok=True)
        args.save_type.append(args.cam_png)
    if args.crf:
        args.crf_list = list()
        for t in args.crf_t:     #  crf次数, 3,5,10
            for alpha in args.crf_alpha:   #  alpha 系数, 1, 4, 10, 24
                crf_folder = os.path.join(args.crf, 'crf_{}_{}'.format(t, alpha))  # 保存路径在crf下面
                args.crf_folder = crf_folder
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(args.crf_folder)
                # args.save_type.append(crf_folder)

    # processors
    args.n_processes_per_gpu = [int(_) for _ in args.n_processes_per_gpu]
    args.n_total_processes = sum(args.n_processes_per_gpu)
    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)
    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])
    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())
    return multi_scale_flipped_image_list


def predict_cam(model, image, label, gpu, network_type):

    original_image_size = np.asarray(image).shape[:2]
    # preprocess image
    multi_scale_flipped_image_list = preprocess(image, scales, transform)

    cam_list = list()
    model.eval()
    for i, image in enumerate(multi_scale_flipped_image_list):
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.cuda(gpu)
            cam = model.forward_cam(image)

            if network_type == 'cls':
                cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]

                cam = cam.cpu().numpy() * label.reshape(20, 1, 1)

                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)

            elif network_type == 'contrast':
                cam = F.softmax(cam, dim=1)
                cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]

                cam_fg = cam[:-1]
                cam_bg = cam[-1:]

                cam_fg = cam_fg.cpu().numpy() * label.reshape(20, 1, 1)
                cam_bg = cam_bg.cpu().numpy()

                if i % 2 == 1:
                    cam_fg = np.flip(cam_fg, axis=-1)
                    cam_bg = np.flip(cam_bg, axis=-1)
                cam_list.append((cam_fg, cam_bg))

            elif network_type == 'eps':
                cam = F.softmax(cam, dim=1)
                cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]

                cam_fg = cam[:-1]
                cam_bg = cam[-1:]

                cam_fg = cam_fg.cpu().numpy() * label.reshape(20, 1, 1)
                cam_bg = cam_bg.cpu().numpy()

                if i % 2 == 1:
                    cam_fg = np.flip(cam_fg, axis=-1)
                    cam_bg = np.flip(cam_bg, axis=-1)
                cam_list.append((cam_fg, cam_bg))
            else:
                raise Exception('No appropriate model type')

    return cam_list


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]
    return n_crf_al


def infer_cam_mp(process_id, image_ids, label_list, cur_gpu):
    print('process {} starts...'.format(os.getpid()))

    print(process_id, cur_gpu)
    print('GPU:', cur_gpu)
    print('{} images per process'.format(len(image_ids)))

    model = getattr(importlib.import_module(args.network), 'Net')()
    model = model.cuda(cur_gpu)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    torch.no_grad()

    for i, (img_id, label) in enumerate(zip(image_ids, label_list)):

        # load image
        img_path = os.path.join(args.img_root, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        org_img = np.asarray(img)

        # infer cam_list
        cam_list = predict_cam(model, img, label, cur_gpu, args.network_type)

        if args.network_type == 'cls':
            sum_cam = np.sum(cam_list, axis=0)
        elif args.network_type == 'eps_seam':
            cam_np = np.array(cam_list)
            cam_fg = cam_np[:, 0]
            sum_cam = np.sum(cam_fg, axis=0)
        elif args.network_type == 'eps':
            cam_np = np.array(cam_list)
            cam_fg = cam_np[:, 0]
            sum_cam = np.sum(cam_fg, axis=0)
        else:
            raise Exception('No appropriate model type')

        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
        # min-max norm
        # sum_cam[sum_cam < 0] = 0
        # cam_max = np.max(sum_cam, (1, 2), keepdims=True)
        # cam_min = np.min(sum_cam, (1, 2), keepdims=True)
        # sum_cam[sum_cam < cam_min+1e-5] = 0
        # norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for j in range(20):
            if label[j] > 1e-5:
                cam_dict[j] = norm_cam[j]

        h, w = list(cam_dict.values())[0].shape
        tensor = np.zeros((21, h, w), np.float32)
        for key in cam_dict.keys():
            tensor[key + 1] = cam_dict[key]
        tensor[0, :, :] = args.thr
        pred = np.argmax(tensor, axis=0).astype(np.uint8)

        # save cam
        if args.cam_npy is not None:

            np.save(os.path.join(args.cam_npy, img_id + '.npy'), cam_dict)

        if args.cam_png is not None:
            imageio.imwrite(os.path.join(args.cam_png, img_id + '.png'), pred)

        if args.crf is not None:
            for folder, t, alpha in args.crf_list:
                cam_crf = _crf_with_alpha(org_img, cam_dict, alpha, t=t)
                np.save(os.path.join(folder, img_id + '.npy'), cam_crf)
        if i % 10 == 0:
            print('PID{}, {}/{} is complete'.format(process_id, i, len(image_ids)))


def main_mp():
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids)
    n_total_images = len(image_ids)
    assert len(image_ids) == len(label_list)

    saved_list = sorted([file[:-4] for file in os.listdir(args.save_type[0])])
    n_saved_images = len(saved_list)
    new_image_ids = list()
    new_label_list = list()
    for i, name in enumerate(image_ids):
        if name not in saved_list:
            new_image_ids.append(name)
            new_label_list.append(label_list[i])
    image_ids = new_image_ids
    label_list = new_label_list

    n_total_processes = args.n_total_processes
    print('===========================')
    print('OVERALL INFORMATION')
    print('n_gpus:', n_gpus)
    print('n_processes_per_gpu', args.n_processes_per_gpu)
    print('n_total_processes:', n_total_processes)
    print('n_total_images:', n_total_images)
    print('n_saved_images:', n_saved_images)
    print('n_images_to_proceed', len(image_ids))
    print('===========================')

    sub_image_ids = list()
    sub_label_list = list()

    # split model and data
    split_size = len(image_ids) // n_total_processes
    for i in range(n_total_processes):
        # split image ids and labels
        if i == n_total_processes - 1:
            sub_image_ids.append(image_ids[split_size * i:])
            sub_label_list.append(label_list[split_size * i:])
        else:
            sub_image_ids.append(image_ids[split_size * i:split_size * (i + 1)])
            sub_label_list.append(label_list[split_size * i:split_size * (i + 1)])

    # multi-process
    gpu_list = list()
    for idx, num in enumerate(args.n_processes_per_gpu):
        gpu_list.extend([idx for i in range(num)])
    processes = list()
    for idx, process_id in enumerate(range(n_total_processes)):
        proc = Process(target=infer_cam_mp,
                       args=(process_id, sub_image_ids[idx], sub_label_list[idx], gpu_list[idx]))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == '__main__':
    crf_alpha = (4, 32)
    args = parse_args()

    n_gpus = args.n_gpus
    scales = (0.5, 1.0, 1.5, 2.0)
    normalize = Normalize()
    transform = torchvision.transforms.Compose([np.asarray, normalize, HWC_to_CHW])

    main_mp()

    print(time.time() - start)
