# 1. Define paths and params
PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}

# DATASET_PATH=${ROOT_DIR}/"datasets"/"GangnamStation"
DATASETS_PATH=$ROOT_DIR/datasets
DATASET=GangnamStation
mkdir -p ${DATASETS_PATH}

TOPK=20  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract

cd ${WORKING_DIR}

for SCENE in B1 B2; do
  DATASET=GangnamStation/${SCENE}/release
  # 1) Create temporal mapping and query sets (they will be modified)
  mkdir -p ${WORKING_DIR}/${DATASET}/mapping/sensors
  cp -rf ${DATASETS_PATH}/${DATASET}/mapping/sensors/*.txt ${WORKING_DIR}/${DATASET}/mapping/sensors/
  echo 1
  ln -s ${DATASETS_PATH}/${DATASET}/mapping/sensors/records_data ${WORKING_DIR}/${DATASET}/mapping/sensors/records_data

  mkdir -p ${WORKING_DIR}/${DATASET}/test/sensors
  cp -rf ${DATASETS_PATH}/${DATASET}/test/sensors/*.txt ${WORKING_DIR}/${DATASET}/test/sensors/
  ln -s ${DATASETS_PATH}/${DATASET}/test/sensors/records_data ${WORKING_DIR}/${DATASET}/test/sensors/records_data

  # 2) Merge mapping and test kaptures (this will make it easier to extract the local and global features and it will be used for the localization step)
  kapture_merge.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping ${WORKING_DIR}/${DATASET}/test \
    -o ${WORKING_DIR}/${DATASET}/map_plus_test \
    --image_transfer link_relative

  # 3) Extract global features (we will use AP-GeM here)
  cd $ROOT_DIR/deep-image-retrieval
  python -m dirtorch.extract_kapture --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_test/ --checkpoint dirtorch/models/Resnet101-AP-GeM-LM18.pt --gpu 0
  
  mkdir -p ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features
  mv ${WORKING_DIR}/${DATASET}/map_plus_test/reconstruction/global_features/Resnet101-AP-GeM-LM18/* ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features/
  rm -rf ${WORKING_DIR}/${DATASET}/map_plus_test/reconstruction/global_features/Resnet101-AP-GeM-LM18
    
  # 5) Extract local features (we will use R2D2 here)
  cd $ROOT_DIR/r2d2
  python extract_kapture.py --model models/r2d2_WASF_N8_big.pt --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_test/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k ${KPTS}

  # move to right location
  mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors
  mv ${WORKING_DIR}/${DATASET}/map_plus_test/reconstruction/descriptors/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors/
  mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints
  mv ${WORKING_DIR}/${DATASET}/map_plus_test/reconstruction/keypoints/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints/
  

  # 6) mapping pipeline
  
  cd ${WORKING_DIR}
  LOCAL=r2d2_WASF_N8_big
  GLOBAL=Resnet101-AP-GeM-LM18
  python ../mapping/kapture_pipeline_mapping.py -v debug -f \
  -i ${WORKING_DIR}/${DATASET}/mapping \
  -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
  -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
  -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
  -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
  -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
  --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
  --topk ${TOPK}

  # 7) localization pipeline
  
  cd ${WORKING_DIR}
  python ../localization/kapture_pipeline_localize.py -v debug -f \
  -i ${WORKING_DIR}/${DATASET}/mapping \
  --query ${WORKING_DIR}/${DATASET}/query \
  -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
  -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
  -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
  -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
  -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
  --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
  -o ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL} \
  --topk ${TOPK} \
  --config 2
  --benchmark-style Gangnam_Station


  # 7) cat the output files in order to generate one file for benchmark submission
  cat ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL}/LTVL2020_style_result.txt >> ${WORKING_DIR}/GangnamStation/GangnamStation_LTVL2020_style_result_all_scenes_${LOCAL}_${GLOBAL}.txt
done









# # move to right location
# mkdir -p ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features
# mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18/* ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features/
# rm -rf ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18

# # 5) Extract local features (we will use R2D2 here)
# cd $ROOT_DIR/r2d2
# python extract_kapture.py --model models/r2d2_WASF_N8_big.pt --kapture-root ../workspace/aachenv11/Aachen-Day-Night-v1.1/map_plus_query/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k 20000

# # move to right location
# mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors
# mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/descriptors/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors/
# mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints
# mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/keypoints/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints/



# # 6) mapping pipeline
# LOCAL=r2d2_WASF_N8_big
# GLOBAL=Resnet101-AP-GeM-LM18
# python ../mapping/kapture_pipeline_mapping.py -v debug -f \
#   -i ${WORKING_DIR}/${DATASET}/mapping \
#   -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
#   -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
#   -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
#   -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
#   -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
#   --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
#   --topk ${TOPK}

# # 7) localization pipeline
# python ../localization/kapture_pipeline_localize.py -v debug -f \
#   -i ${WORKING_DIR}/${DATASET}/mapping \
#   --query ${WORKING_DIR}/${DATASET}/query \
#   -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
#   -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
#   -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
#   -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
#   -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
#   --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
#   -o ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL} \
#   --topk ${TOPK} \
#   --config 2
#   --benchmark-style Gangnam_Station
