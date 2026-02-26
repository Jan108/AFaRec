#!/usr/bin/env bash
set -eo pipefail


for cls in "all" "bird" "cat" "cat_like" "dog" "dog_like" "horse_like" "small_animals"; do
  PYTHONPATH='/mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/train.py \
  --train-data-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_${cls}_train.txt" \
  --train-data-images "/mnt/data/afarec/data/OAFI_full/images/train" \
  --weights "$(dirname "$0")/work_dir/retinaface_pretrained/retinaface_r34.pth" \
  --network 'resnet34' \
  --batch-size 16 \
  --print-freq 40 \
  --save-dir "./work_dir/retinaface_${cls}/"

  PYTHONPATH='/mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/evaluate_widerface.py \
  -w "$(dirname "$0")/work_dir/retinaface_${cls}/resnet34_final.pth" \
  --network "resnet34" \
  --origin_size \
  --save-folder "$(dirname "$0")/work_dir/retinaface_${cls}/results/" \
  --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test" \
  --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_${cls}_test.txt"
done


PYTHONPATH='/mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/':$PYTHONPATH \
python /mnt/data/afarec/code/face_detection/retinaface/retinaface-pytorch/evaluate_widerface.py \
  -w "$(dirname "$0")/work_dir/retinaface_pretrained/retinaface_r34.pth" \
  --network "resnet34" \
  --origin_size \
  --save-folder "$(dirname "$0")/work_dir/retinaface_pretrained/results/" \
  --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test" \
  --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_all_test.txt"