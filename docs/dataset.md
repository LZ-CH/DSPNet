### Prepare EmbodiedScan Data

Given the licenses of respective raw datasets, we recommend users download the raw data from their official websites and then organize them following the below guide.
Detailed steps are shown as follows.

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the folder to this level of directory.
2. Download ScanQA data [HERE](https://github.com/ATR-DBI/ScanQA). Link or move the folder to this level of directory.
3. Download SQA data [HERE](https://github.com/SilongYong/SQA3D). Link or move the folder to this level of directory.
The directory structure should be as below.
```
data
├── scannet
│   ├── meta_data
│   ├── scans
│   │   ├── <scene_id>
│   │   ├── ...
│   ├── scans_test
│   │   ├── <scene_id>
│   │   ├── ...
├── qa
│   ├── ScanQA_v1.0_test_w_obj.json
│   ├── ...
├── sqa_task
│   ├── answer_dict.json
│   ├── balanced
│   │   ├── v1_balanced_questions_test_scannetv2.json
│   │   ├── ...
```
4. Preprocess scannet's point clounds:
```bash
cd ./data/scannet
python batch_load_scannet_data.py
```

5. Enter the project root directory, extract images:
```bash
python embodiedqa/converter/generate_image_scannetv2.py --dataset_folder ./data/scannet/ --fast
```
6. Preprocess scannet's annotation:
```bash
python embodiedqa/converter/create_scannetv2_info_pkl.py --dataset_folder ./data/scannet/ --output_dir ./data
```

7. 
The directory structure should be as below after that

```
data
├── scannet
│   ├── meta_data
│   ├── scans
│   │   ├── <scene_id>
│   │   ├── ...
│   ├── scans_test
│   │   ├── <scene_id>
│   │   ├── ...
│   ├── posed_images
│   │   ├── <scene_id>
│   │   |   ├── *.jpg
│   │   |   ├── *.png
│   │   ├── ...
│   ├── posed_images_test
│   │   ├── <scene_id>
│   │   |   ├── *.jpg
│   │   |   ├── *.png
│   │   ├── ...
│   ├── scannet_data
│   │   ├── <scene_id>_aligned_vert.npy
│   │   ├── ...
├── qa
│   ├── ScanQA_v1.0_test_w_obj.json
│   ├── ...
├── sqa_task
│   ├── answer_dict.json
│   ├── balanced
│   │   ├── v1_balanced_questions_test_scannetv2.json
│   │   ├── ...
├── mv_scannetv2_infos_test.pkl
├── mv_scannetv2_infos_train.pkl
├── mv_scannetv2_infos_val.pkl
```