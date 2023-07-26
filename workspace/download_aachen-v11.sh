PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}

DATASETS_PATH=${ROOT_DIR}/datasets
DATASET="Aachen-Day-Night-v1.1"
mkdir -p ${WORKING_DIR}/${DATASET}

cd ${WORKING_DIR}

# Download Aachen-Day-Night-v1.1 dataset
mkdir ${DATASETS_PATH}
kapture_download_dataset.py --install_path ${DATASETS_PATH} update
kapture_download_dataset.py --install_path ${DATASETS_PATH} install ${DATASET}_mapping ${DATASET}_query_day ${DATASET}_query_night
