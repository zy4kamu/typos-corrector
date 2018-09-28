#!/bin/bash

echo 'creating dataset ...'
/home/stepan/git-repos/typos-corrector/build/dataset-generator/dataset-generator-app

echo 'training ...'
vw -d dataset/train --oaa 40      \
   -f output/predictor.vw         \
   --loss_function logistic       \
   -b 20                          \
   --readable_model output/predictor.txt
  #--l1 0.00000001 --l2 0.0000001 \
  #--passes 2                     \

echo ''
echo 'testing on train ...'
vw -d dataset/train -t \
   -i output/predictor.vw \
   -r output/train_raw_predictions.txt \
   -p output/train_predictions.txt

echo ''
echo 'testing on test ...'
vw -d dataset/test -t \
   -i output/predictor.vw \
   -r output/test_raw_predictions.txt \
   -p output/test_predictions.txt

echo 'train accuracy:'
python check_accuracy.py output/train_raw_predictions.txt dataset/train

echo 'test accuracy:'
python check_accuracy.py output/test_raw_predictions.txt dataset/test

echo 'converting model ...'
python convert_model.py output/predictor.txt model/data
cp dataset/countries model/labels

echo 'done ...'
