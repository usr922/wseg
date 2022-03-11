import os
import pandas as pd
import numpy as np
import multiprocessing
import argparse
import imageio

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_save(predict_folder,
            save_folder,
            name_list,
            num_process=8):

    def compare(start, step):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            predict_file = os.path.join(predict_folder, '%s.npy' % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in predict_dict.keys():
                tensor[key] = predict_dict[key]
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            imageio.imwrite(os.path.join(save_folder, name + '.png'), predict)

    p_list = []
    for i in range(num_process):
        p = multiprocessing.Process(target=compare, args=(i, num_process))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datalist', type=str, default='trainaug.txt')
    parser.add_argument('--crf_pred', type=str)
    parser.add_argument("--label_save_dir", type=str)
    parser.add_argument('--num-process', default=8, type=int)
    args = parser.parse_args()

    os.makedirs(args.label_save_dir, exist_ok=True)

    df = pd.read_csv(args.datalist, names=['filename'])
    name_list = df['filename'].values
    do_save(args.crf_pred, args.label_save_dir, name_list, 8)
