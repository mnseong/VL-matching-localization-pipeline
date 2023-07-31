# 1. Define paths and params
PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}
mkdir Aachen-Day-Night-v1.1

WORK_PATH=${WORKING_DIR}/Aachen-Day-Night-v1.1
DATASET_PATH=$ROOT_DIR/datasets/Aachen-Day-Night-v1.1

TOPK=20  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract

cd ${WORKING_DIR}

# 2. Create temporal mapping & query sets
mkdir -p ${WORK_PATH}/mapping/sensors
cp -rf ${DATASET_PATH}/mapping/sensors/*.txt ${WORK_PATH}/mapping/sensors/
ln -s ${DATASET_PATH}/mapping/sensors/records_data ${WORK_PATH}/mapping/sensors/records_data

mkdir -p ${WORK_PATH}/query/sensors
cp -rf ${DATASET_PATH}/query/sensors/*.txt ${WORK_PATH}/query/sensors/
ln -s ${DATASET_PATH}/query/sensors/records_data ${WORK_PATH}/query/sensors/records_data


# 3) Merge mapping and query kaptures (this will make it easier to extract the local and global features and it will be used for the localization step)
kapture_merge.py -v debug \
  -i ${WORK_PATH}/mapping ${WORK_PATH}/query \
  -o ${WORK_PATH}/map_plus_query \
  --image_transfer link_relative
