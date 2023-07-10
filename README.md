# VL-matching-localization-pipeline
Matching &amp; Localization algorithmic workflow of Visual Localization pipeline (Summer 2023, Research at the ETRI)

---
### 1. Clone this repository
```bash
git clone https://github.com/mnseong/VL-matching-localization-pipeline.git
```
<br/>

Please refer to our directory structure
```
VL-matching-localization-pipeline
    ├─ datasets
    │   └─my_dataset
    │      ├─ global_features
    │      ├─ local_features
    │      ├─ mapping
    │      ├─ query
    │      └─ map_plus_query # kapture_merge.py with mapping/query inputs
    ├─ mapping
    │   ├─ kapture_pipeline_mapping.py
    │   └─ pipeline_import_paths.py
    ├─ localization
    │   ├─ kapture_pipeline_localize.py
    │   └─ pipeline_import_paths.py
    └─ workspace
        ├─ my_dataset # This directory is copy from dataset, It will be used to workspace.
        └─ run_my_dataset.sh # run script
```
<br/>

### 2. Install the required Python library with the following command
```bash
pip install -r requirements.txt
```
<br/>

### 3. Run shell script file for the data set you want
You can use the pipeline in "/workspace".
```bash
cd workspace
./run_virtual.sh # example command for virtual_gallery dataset
```
<br/>

### 4. Check the results
Go to results directory.
```bash
cd virtual/virtual_gallery_tutorial # example command for virtual_gallery dataset
```
<br/>

##### Results of Matching pipeline
```bash
# matching
colmap gui \
    --database_path ./colmap-sfm/r2d2_500/AP-GeM-LM18_top5/colmap.db \
    --image_path ./mapping/sensors/records_data \
    --import_path ./colmap-sfm/r2d2_500/AP-GeM-LM18_top5/reconstruction/
```
<br/>

##### Results of Localization pipeline
```bash
# localization
colmap gui \
    --database_path ./colmap-localization/r2d2_500/AP-GeM-LM18_top5/AP-GeM-LM18_top5/colmap_localized/colmap.db \
    --image_path query/sensors/records_data \
    --import_path ./colmap-localization/r2d2_500/AP-GeM-LM18_top5/AP-GeM-LM18_top5/colmap_localized/reconstruction/
```