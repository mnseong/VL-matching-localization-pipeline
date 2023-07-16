# 1. Define paths and params
PYTHONBIN=3.8.17
WORKING_DIR=${PWD}
cd ../
ROOT_DIR=`pwd`
cd ${WORKING_DIR}
DATASET_DIR=aachenv11
DATASET="${DATASET_DIR}/Aachen-Day-Night-v1.1"
mkdir -p ${WORKING_DIR}/${DATASET}

TOPK=20  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract


# 0b) Get extraction code for local and global features
# ! skip if already done !
# Deep Image retrieval - AP-GeM
# pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm
#cd ${WORKING_DIR}
#git clone https://github.com/naver/deep-image-retrieval.git
#cd deep-image-retrieval
#mkdir -p dirtorch/data/
#cd dirtorch/data/
#gdown --id 1r76NLHtJsH-Ybfda4aLkUIoW3EEsi25I # downloads a pre-trained model of AP-GeM
#unzip Resnet101-AP-GeM-LM18.pt.zip
#rm -rf Resnet101-AP-GeM-LM18.pt.zip
# R2D2
cd ${WORKING_DIR}
#git clone https://github.com/naver/r2d2.git

# 1) Download dataset
# Note that you will be asked to accept or decline the license terms before download.
# mkdir ${DATASETS_PATH}
# kapture_download_dataset.py --install_path ${DATASETS_PATH} update
# kapture_download_dataset.py --install_path ${DATASETS_PATH} install ${DATASET}_mapping ${DATASET}_query_day ${DATASET}_query_night
# rm -rf ${DATASETS_PATH}/${DATASET}/mapping/reconstruction # remove the keypoints and 3D points that come with the dataset (this is Aachen specific)
# kapture_merge.py -v debug \
#   -i ${DATASETS_PATH}/${DATASET}/query_day ${DATASETS_PATH}/${DATASET}/query_night \
#   -o ${DATASETS_PATH}/${DATASET}/query \
#   --image_transfer link_relative



# 2) Create temporal mapping and query sets (they will be modified)
mkdir -p ${WORKING_DIR}/${DATASET}/mapping/sensors
cp -rf $ROOT_DIR/datasets/Aachen-Day-Night-v1.1/mapping/sensors/*.txt ${WORKING_DIR}/${DATASET}/mapping/sensors/
ln -s $ROOT_DIR/datasets/Aachen-Day-Night-v1.1/mapping/sensors/records_data ${WORKING_DIR}/${DATASET}/mapping/sensors/records_data

mkdir -p ${WORKING_DIR}/${DATASET}/query/sensors
cp -rf $ROOT_DIR/datasets/Aachen-Day-Night-v1.1/query/sensors/*.txt ${WORKING_DIR}/${DATASET}/query/sensors/
ln -s $ROOT_DIR/datasets/Aachen-Day-Night-v1.1/query/sensors/records_data ${WORKING_DIR}/${DATASET}/query/sensors/records_data




# 3) Merge mapping and query kaptures (this will make it easier to extract the local and global features and it will be used for the localization step)
kapture_merge.py -v debug \
  -i ${WORKING_DIR}/${DATASET}/mapping ${WORKING_DIR}/${DATASET}/query \
  -o ${WORKING_DIR}/${DATASET}/map_plus_query \
  --image_transfer link_relative

# 4) Extract global features (we will use AP-GeM here)
cd $ROOT_DIR/deep-image-retrieval
python -m dirtorch.extract_kapture --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_query/ --checkpoint dirtorch/models/Resnet101-AP-GeM-LM18.pt --gpu 0

# move to right location
mkdir -p ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18/* ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features/
rm -rf ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/Resnet101-AP-GeM-LM18

# 5) Extract local features (we will use R2D2 here)
cd $ROOT_DIR/r2d2
python extract_kapture.py --model models/r2d2_WASF_N8_big.pt --kapture-root ../workspace/aachenv11/Aachen-Day-Night-v1.1/map_plus_query/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k 20000

# move to right location
mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/descriptors/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors/
mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/keypoints/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints/



# 6) mapping pipeline
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
