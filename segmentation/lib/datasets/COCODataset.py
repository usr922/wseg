# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import pickle
import torch
import pandas as pd
import cv2
import json
from tqdm import trange
from skimage import io
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from datasets.transform import *
from utils.registry import DATASETS

class COCODataset(Dataset):
    def __init__(self, cfg, period):
        super(Dataset, self).__init__()
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data','MSCOCO')
        self.dataset_dir = self.root_dir
        
        self.period = period
        self.year = cfg.DATA_YEAR
        self.img_dir = os.path.join(self.dataset_dir, 'images','%s%s'%(self.period,self.year))
        self.ann_dir = os.path.join(self.dataset_dir, 'annotations/instances_%s%s.json'%(self.period,self.year))
        self.ids_file = os.path.join(self.dataset_dir, 'annotations/instances_%s%s_ids.mx'%(self.period,self.year))
        self.rst_dir = os.path.join(self.dataset_dir,'results','%s%s'%(self.period,self.year))
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.totensor = ToTensor()
        self.cfg = cfg
        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}


        self.coco2voc = [0]*91
        for voc_idx in range(len(self.voc2coco)):
            for coco_idx in self.voc2coco[voc_idx]:
                self.coco2voc[coco_idx] = voc_idx
                
        self.coco = COCO(self.ann_dir)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
#       self.imgIds = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()
        from pycocotools import mask
        self.coco_mask = mask
        if os.path.exists(self.ids_file):
            with open(self.ids_file, 'rb') as f:
                self.imgIds = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.imgIds = ids#self._preprocess(ids, self.ids_file)

        self.transforms = []
        if self.period == 'train':
            if cfg.DATA_RANDOMROTATION > 0:
                self.transforms.append(RandomRotation(cfg.DATA_RANDOMROTATION))
            if cfg.DATA_RANDOMSCALE != 1:
                self.transforms.append(RandomScale(cfg.DATA_RANDOMSCALE))
            if cfg.DATA_RANDOMFLIP > 0:
                self.transforms.append(RandomFlip(cfg.DATA_RANDOMFLIP))
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.transforms.append(RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V))
            if cfg.DATA_RANDOMGAUSSIAN > 0:
                self.transforms.append(RandomGaussian(cfg.DATA_RANDOMGAUSSIAN))
            if cfg.DATA_RANDOMCROP > 0:
                self.transforms.append(RandomCrop(cfg.DATA_RANDOMCROP))
        else:
            self.transforms.append(Multiscale(self.cfg.TEST_MULTISCALE))

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        raise NotImplementedError

@DATASETS.register_module
class COCOSmtSegDataset(COCODataset):
    def __init__(self, cfg, period):
        super(COCOSmtSegDataset, self).__init__(cfg, period)

    def __getitem__(self, idx):
        img_ann = self.coco.loadImgs(self.imgIds[idx])
        name = os.path.join(self.img_dir, img_ann[0]['file_name'])
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        
        if self.period == 'train':
            annIds = self.coco.getAnnIds(imgIds=self.imgIds[idx])
            anns = self.coco.loadAnns(annIds)
            segmentation = np.zeros((r,c),dtype=np.uint8)
            for ann_item in anns:
                mask = self.coco.annToMask(ann_item)
                segmentation[mask>0] = self.json_category_id_to_contiguous_id[ann_item['category_id']]
            if np.max(segmentation)>91:
                print(np.max(segmentation))
                raise ValueError('segmentation > 91')
            if np.max(segmentation)>20:
                print(np.max(segmentation))
                raise ValueError('segmentation > 20')
            sample['segmentation'] = segmentation

        for t in self.transforms:
            sample = t(sample)

        if 'segmentation' in sample.keys():
            sample['segmentation_onehot'] = onehot(sample['segmentation'], self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample
 
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap
    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            c = self.json_category_id_to_contiguous_id[cat]
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask


    def _preprocess(self, ids, ids_file):
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            if(mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.\
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    def do_python_eval(self):
        TP = []
        P = []
        T = []
        ids = list(sorted(self.coco.imgs.keys()))
        for i in range(num_cls):
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        
        def compare(start,step,TP,P,T):
            for idx in range(start,len(ids),step):
                name = self.coco.loadImgs(ids[idx])[0]['file_name'][:-4]
                predict_file = os.path.join(self.rst_dir,'%s.png'%name)
                predict = np.array(Image.open(predict_file))
    
                annIds = self.coco.getAnnIds(imgIds=ids[idx])
                anns = self.coco.loadAnns(annIds)
                gt = np.zeros(predict.shape,dtype=np.uint8)
                for ann_item in anns:
                    mask_item = self.coco.annToMask(ann_item)
                    gt[mask_item>0] = self.json_category_id_to_contiguous_id[ann_item['category_id']]
                cal = gt<255
                mask = (predict==gt) * cal
          
                for i in range(num_cls):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        T_TP = []
        P_TP = []
        FP_ALL = []
        FN_ALL = [] 
        for i in range(num_cls):
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
            T_TP.append(T[i].value/(TP[i].value+1e-10))
            P_TP.append(P[i].value/(TP[i].value+1e-10))
            FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
            FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        loglist = {}
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
            loglist[categories[i]] = IoU[i] * 100
                   
        miou = np.mean(np.array(IoU))
        t_tp = np.mean(np.array(T_TP)[1:])
        p_tp = np.mean(np.array(P_TP)[1:])
        fp_all = np.mean(np.array(FP_ALL)[1:])
        fn_all = np.mean(np.array(FN_ALL)[1:])
        miou_foreground = np.mean(np.array(IoU)[1:])
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('%11s:%7.3f'%('T/TP',t_tp))
        print('%11s:%7.3f'%('P/TP',p_tp))
        print('%11s:%7.3f'%('FP/ALL',fp_all))
        print('%11s:%7.3f'%('FN/ALL',fn_all))
        print('%11s:%7.3f'%('miou_foreground',miou_foreground))
        loglist['mIoU'] = miou * 100
        loglist['t_tp'] = t_tp
        loglist['p_tp'] = p_tp
        loglist['fp_all'] = fp_all
        loglist['fn_all'] = fn_all
        loglist['miou_foreground'] = miou_foreground 
        return loglist

@DATASETS.register_module
class COCOInsDataset(COCODataset):
    def __init__(self, cfg, period):
        super(COCOInsSegDataset, self).__init__(cfg, period)

    def __getitem__(self, idx):
        img_ann = self.coco.loadImgs(self.imgIds[idx])
        name = os.path.join(self.img_dir, img_ann[0]['file_name'])
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        if self.period == 'train':
            annIds = self.coco.getAnnIds(imgIds=self.imgIds[idx])
            anns = self.coco.loadAnns(annIds)
            segmentation = np.zeros((r,c),dtype=np.uint8)
            mask_list = []
            cls_list = []
            for ann_item in anns:
                mask_list.append(self.coco.annToMask(ann_item).reshape(1,r,c))
                cls_list.append(self.json_category_id_to_contiguous_id[ann_item['category_id']])
            sample['segmentation_ins'] = np.concatenate(mask_list, axis=0)
            sample['category_ins'] = np.array(cls_list)
            sample['num_ins'] = len(anns) 
            sample['adj_ins'] = self.__mask2mat(sample['segmentation_ins'])

        for t in self.transforms:
            sample = t(sample)

        sample = self.totensor(sample)

        return sample

    def __mask2mat(self, mask):
        k, h, w = mask.shape
        v = mask.reshape(k, h*w)
        mat = np.mul(v.T, v)
        return mat

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.imgIds[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.imgIds[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.catIds[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.imgIds[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.catIds[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if isinstance(seg, tuple):
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.catIds[label]
                if isinstance(segms[i]['counts'], bytes):
                    segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def segm2json_segm(dataset, results):
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.imgIds[idx]
        seg = results[idx]
        for label in range(len(seg)):
            masks = seg[label]
            for i in range(len(masks)):
                mask_score = masks[i][1]
                segm = masks[i][0]
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score)
                data['category_id'] = dataset.catIds[label]
                segm['counts'] = segm['counts'].decode()
                data['segmentation'] = segm
                segm_json_results.append(data)
    return segm_json_results


def results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        json.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        json.dump(json_results[0], result_files['bbox'])
        json.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        json.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files


def results2json_segm(dataset, results, out_file):
    result_files = dict()
    json_results = segm2json_segm(dataset, results)
    result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
    json.dump(json_results, result_files['segm'])

    return result_files

