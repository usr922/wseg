import numpy as np
import os
import argparse
from PIL import Image
import pandas as pd
import multiprocessing
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_list", default="./VOC2012/ImageSets/Segmentation/trainaug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--cam_dir", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument("--crf_iters", default=10, type=float)
    parser.add_argument("--alpha", default=4, type=float)

    args = parser.parse_args()

    assert args.cam_dir is not None

    if args.out_crf:
        if not os.path.exists(args.out_crf):
            os.makedirs(args.out_crf)

    df = pd.read_csv(args.infer_list, names=['filename'])
    name_list = df['filename'].values


    # https://github.com/pigcv/AdvCAM/blob/fa08f0ad4c1f764f3ccaf36883c0ae43342d34c5/misc/imutils.py#L156
    def _crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.7):
        h, w = img.shape[:2]
        d = dcrf.DenseCRF2D(w, h, n_labels)
        U = unary_from_labels(labels, 21, gt_prob=gt_prob, zero_unsure=False)
        d.setUnaryEnergy(U)
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(t)

        return np.array(Q).reshape((n_labels, h, w))


    def _infer_crf_with_alpha(start, step, alpha):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            cam_file = os.path.join(args.cam_dir, '%s.npy' % name)
            cam_dict = np.load(cam_file, allow_pickle=True).item()
            h, w = list(cam_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            tensor[0, :, :] = np.power(1 - np.max(tensor, axis=0, keepdims=True), alpha)

            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            img = Image.open(os.path.join('./VOC2012/JPEGImages', name + '.jpg')).convert("RGB")
            img = np.array(img)
            crf_array = _crf_inference(img, predict)

            crf_folder = args.out_crf + ('/%.2f' % alpha)
            if not os.path.exists(crf_folder):
                os.makedirs(crf_folder)

            np.save(os.path.join(crf_folder, name + '.npy'), crf_array)


    alpha_list = [4, 8, 16, 24, 32]

    for alpha in alpha_list:
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=_infer_crf_with_alpha, args=(i, 8, args.alpha))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        print(f'Info: Alpha {alpha} done!')
