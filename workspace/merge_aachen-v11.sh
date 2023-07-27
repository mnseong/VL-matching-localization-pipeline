# 1. Define paths and params
PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}
DATASETS_PATH=${ROOT_DIR}/datasets
DATASET="Aachen-Day-Night-v1.1"
#DATASET="${DATASET_DIR}/Aachen-Day-Night-v1.1"

cd ${WORKING_DIR}

kapture_merge.py -v debug \
  -i ${DATASETS_PATH}/${DATASET}/query_day ${DATASETS_PATH}/${DATASET}/query_night \
  -o ${DATASETS_PATH}/${DATASET}/query \
  --image_transfer link_relative
