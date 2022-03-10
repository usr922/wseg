# NEED TO SET
DATASET_ROOT=./VOC2012
WEIGHT_ROOT=./weights
SESSION=resnet38_contrast

GPU=0,1,2



BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# train classification network with Contrastive Learning
CUDA_VISIBLE_DEVICES=${GPU} python contrast_train.py \
  --voc12_root ${DATASET_ROOT} \
  --weights ${BASE_WEIGHT} \
  --session_name ${SESSION} \
  --batch_size 9


# 2. inference CAM
DATA=trainaug  # train / trainaug / val
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint_contrast.pth
CAM_NPY_DIR=store/cam_npy/${DATA}
CAM_PNG_DIR=store/cam_png/${DATA}
CRF_PNG_DIR=store/crf_png/${DATA}

CUDA_VISIBLE_DEVICES=${GPU} python contrast_infer.py \
  --weights ${TRAINED_WEIGHT} \
  --infer_list voc12/${DATA}.txt \
  --out_cam ${CAM_NPY_DIR} \
  --out_cam_pred ${CAM_PNG_DIR} \
  --out_crf ${CRF_PNG_DIR}


# 3. evaluate CAM
DATA=train  # train / val
LIST=VOC2012/ImageSets/Segmentation/${DATA}.txt
RESULT_DIR=${CAM_PNG_DIR}
COMMENT=YOURCOMMENT
GT_ROOT=${DATASET_ROOT}/SegmentationClass/


python eval.py \
  --list ${LIST} \
  --predict_dir ${RESULT_DIR} \
  --gt_dir ${GT_ROOT} \
  --comment ${COMMENT} \
  --type png \
  --curve True
