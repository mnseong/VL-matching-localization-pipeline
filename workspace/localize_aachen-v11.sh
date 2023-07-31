# 1. Define paths and params
PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}
#mkdir Aachen-Day-Night-v1.1

WORK_PATH=${WORKING_DIR}/Aachen-Day-Night-v1.1
DATASET_PATH=$ROOT_DIR/datasets/Aachen-Day-Night-v1.1

TOPK=20  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract

#cd ${WORKING_DIR}

# 2. Create temporal mapping & query sets
#mkdir -p ${WORK_PATH}/mapping/sensors
#cp -rf ${DATASET_PATH}/mapping/sensors/*.txt ${WORK_PATH}/mapping/sensors/
#ln -s ${DATASET_PATH}/mapping/sensors/records_data ${WORK_PATH}/mapping/sensors/records_data

#mkdir -p ${WORK_PATH}/query/sensors
#cp -rf ${DATASET_PATH}/query/sensors/*.txt ${WORK_PATH}/query/sensors/
#ln -s ${DATASET_PATH}/query/sensors/records_data ${WORK_PATH}/query/sensors/records_data


# 3) Merge mapping and query kaptures (this will make it easier to extract the local and global features and it will be used for the localization step)
#kapture_merge.py -v debug \
#  -i ${WORK_PATH}/mapping ${WORK_PATH}/query \
#  -o ${WORK_PATH}/map_plus_query \
#  --image_transfer link_relative


# 4) Extract global features (we will use AP-GeM here)
#cd $ROOT_DIR/deep-image-retrieval
#python -m dirtorch.extract_kapture --kapture-root ${WORK_PATH}/map_plus_query/ --checkpoint dirtorch/models/Resnet101-AP-GeM-LM18.pt --gpu 0

# move to right location
#mkdir -p ${WORK_PATH}/global_features/Resnet101-AP-GeM-LM18/global_features
#mv ${WORK_PATH}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18/* ${WORK_PATH}/global_features/Resnet101-AP-GeM-LM18/global_features/
#rm -rf ${WORK_PATH}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18

# 5) Extract local features (we will use R2D2 here)
#cd $ROOT_DIR/r2d2
#python extract_kapture.py --model models/r2d2_WASF_N8_big.pt --kapture-root ${WORK_PATH}/map_plus_query/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k 20000

# move to right location
#mkdir -p ${WORK_PATH}/local_features/r2d2_WASF_N8_big/descriptors
#mv ${WORK_PATH}/map_plus_query/reconstruction/descriptors/r2d2_WASF_N8_big/* ${WORK_PATH}/local_features/r2d2_WASF_N8_big/descriptors/
#mkdir -p ${WORK_PATH}/local_features/r2d2_WASF_N8_big/keypoints
#mv ${WORK_PATH}/map_plus_query/reconstruction/keypoints/r2d2_WASF_N8_big/* ${WORK_PATH}/local_features/r2d2_WASF_N8_big/keypoints/


LOCAL=r2d2_WASF_N8_big
GLOBAL=Resnet101-AP-GeM-LM18

# 6) mapping pipeline
#python ../mapping/kapture_pipeline_mapping.py -v debug -f \
#  -i ${WORK_PATH}/mapping \
#  -kpt ${WORK_PATH}/local_features/${LOCAL}/keypoints \
#  -desc ${WORK_PATH}/local_features/${LOCAL}/descriptors \
#  -gfeat ${WORK_PATH}/global_features/${GLOBAL}/global_features \
#  -matches ${WORK_PATH}/local_features/${LOCAL}/NN_no_gv/matches \
#  -matches-gv ${WORK_PATH}/local_features/${LOCAL}/NN_colmap_gv/matches \
#  --colmap-map ${WORK_PATH}/colmap-sfm/${LOCAL}/${GLOBAL} \
#  --topk ${TOPK}

# 7) localization pipeline
python ../localization/kapture_pipeline_localize.py -v debug -f \
  -i ${WORK_PATH}/mapping \
  --query ${WORK_PATH}/query \
  -kpt ${WORK_PATH}/local_features/${LOCAL}/keypoints \
  -desc ${WORK_PATH}/local_features/${LOCAL}/descriptors \
  -gfeat ${WORK_PATH}/global_features/${GLOBAL}/global_features \
  -matches ${WORK_PATH}/local_features/${LOCAL}/NN_no_gv/matches \
  -matches-gv ${WORK_PATH}/local_features/${LOCAL}/NN_colmap_gv/matches \
  --colmap-map ${WORK_PATH}/colmap-sfm/${LOCAL}/${GLOBAL} \
  -o ${WORK_PATH}/colmap-localize/${LOCAL}/${GLOBAL} \
  --topk ${TOPK} \
  --config 2
