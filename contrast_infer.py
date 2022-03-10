import os
import argparse
import numpy as np
import torch
import voc12.data
import importlib
import imageio
import torchvision
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from PIL import Image
from torch.utils.data import DataLoader
from tool import imutils, pyutils
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_contrast", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--out_cam", default=None, type=str)  # cam_npy
    parser.add_argument("--out_crf", default=None, type=str)  # crf_png
    parser.add_argument("--out_cam_pred", default=None, type=str)  # cam_png
    parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)  # cam_png_bg_score
    parser.add_argument("--crf_iters", default=10, type=float)

    args = parser.parse_args()
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  scales=[0.5, 1.0, 1.5, 2.0],
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [np.asarray,
                                                       model.normalize,
                                                       imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in tqdm(enumerate(infer_data_loader), total=len(infer_data_loader)):
        img_name = img_name[0]
        label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]


        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    _, cam, f, ff = model_replicas[i % n_gpus](img.cuda())
                    cam = F.upsample(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam


        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0,
                                            processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1, 2), keepdims=True)
        cam_min = np.min(sum_cam, (1, 2), keepdims=True)
        sum_cam[sum_cam < cam_min + 1e-5] = 0
        norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:

            if not os.path.exists(args.out_cam_pred):
                os.makedirs(args.out_cam_pred)

            bg_score = [np.ones_like(norm_cam[0]) * args.out_cam_pred_alpha]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            imageio.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))


        def _crf(cam_dict, bg_score=0.26):

            h, w = list(cam_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            tensor[0, :, :] = bg_score
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            img = Image.open(os.path.join('./VOC2012/JPEGImages', img_name + '.jpg')).convert("RGB")
            img = np.array(img)
            crf_score = _crf_inference(img, predict)
            return np.argmax(crf_score, axis=0).astype(np.uint8)

        def _crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.7):

            h, w = img.shape[:2]
            d = dcrf.DenseCRF2D(w, h, n_labels)
            unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

            q = d.inference(t)

            return np.array(q).reshape((n_labels, h, w))


        if args.out_crf is not None:
            crf_pred = _crf(cam_dict)
            folder = args.out_crf
            if not os.path.exists(folder):
                os.makedirs(folder)
            imageio.imsave(os.path.join(folder, img_name + '.png'), crf_pred.astype(np.uint8))
