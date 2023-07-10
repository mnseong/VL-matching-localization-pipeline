# 1. Define Paths
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}
DATASET_DIR=virtual
DATASET="${DATASET_DIR}/virtual_gallery_tutorial"
mkdir -p ${WORKING_DIR}/${DATASET}

TOPK=3  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract


# 2. Prepare virtual data
echo "cp -rf $ROOT_DIR/datasets/virtual_gallery_tutorial ${WORKING_DIR}/virtual/"
cp -rf $ROOT_DIR/datasets/virtual_gallery_tutorial ${WORKING_DIR}/virtual/


# 3. Mapping pipeline
LOCAL=r2d2_500
GLOBAL=AP-GeM-LM18
GLOBAL_TOP=AP-GeM-LM18_top5

python ../mapping/kapture_pipeline_mapping.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping \
    -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
    -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
    -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
    -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
    -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
    --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL_TOP} \
    --topk ${TOPK}

cd ${WORKING_DIR}


# 4. Localization pipeline
python ../localization/kapture_pipeline_localize.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping \
    --query ${WORKING_DIR}/${DATASET}/query \
    -kpt ${WORKING_DIR}/${DATASET}/local_features/r2d2_500/keypoints \
    -desc ${WORKING_DIR}/${DATASET}/local_features/r2d2_500/descriptors \
    -gfeat ${WORKING_DIR}/${DATASET}/global_features/AP-GeM-LM18/global_features \
    -matches ${WORKING_DIR}/${DATASET}/local_features/r2d2_500/NN_no_gv/matches \
    -matches-gv ${WORKING_DIR}/${DATASET}/local_features/r2d2_500/NN_colmap_gv/matches \
    --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/r2d2_500/AP-GeM-LM18_top5 \
    -o ${WORKING_DIR}/${DATASET}/colmap-localization/r2d2_500/AP-GeM-LM18_top5/AP-GeM-LM18_top5/ \
    --topk 5 \
    --config 2
