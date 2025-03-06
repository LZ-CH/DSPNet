# Copyright (c) OpenRobotLab. All rights reserved.
import os
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union, Sequence
import collections

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from embodiedqa.registry import DATASETS
from embodiedqa.structures import get_box_type
import time
import csv
class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-1):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            return self.ignore_idx
        return self.vocab[v]
    def __len__(self):
        return len(self.vocab)
@DATASETS.register_module()
class MultiViewScanQADataset(BaseDataset):
    # NOTE: category "step" -> "steps" to avoid potential naming conflicts in
    # TensorboardVisBackend
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 qa_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'Depth',
                 serialize_data: bool = False,
                 filter_empty_gt: bool = True,
                 dontcare_objects: List[str] = ['wall', 'ceiling', 'floor'],
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 anno_indices : Optional[Union[int, Sequence[int]]] = None,
                 **kwargs) -> None:
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.dontcare_objects = dontcare_objects
        self.remove_dontcare = remove_dontcare
        self.load_eval_anns = load_eval_anns
        
        
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         metainfo=metainfo,
                         pipeline=pipeline,
                         serialize_data=serialize_data,
                         test_mode=test_mode,
                         **kwargs)
        self.qa_file = osp.join(self.data_root, qa_file)
        self.convert_info_to_scan()
        self.data_list = self.load_language_data()
        if anno_indices is not None:
            self.data_list = self._get_unserialized_subset(anno_indices)
    def process_metainfo(self):
        """This function will be processed after metainfos from ann_file and
        config are combined."""
        assert 'categories' in self._metainfo

        if 'classes' not in self._metainfo:
            care_objects = [obj for obj in list(self._metainfo['categories'].keys()) if obj not in self.dontcare_objects]
            self._metainfo.setdefault(
                'classes', care_objects)
        else:
            self._metainfo['classes'] = self._metainfo['classes']
        self.label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        for key, value in self._metainfo['categories'].items():
            if key in self._metainfo['classes']:
                self.label_mapping[value] = self._metainfo['classes'].index(
                    key)
            elif 'others' in self._metainfo['classes'] and key not in self.dontcare_objects:
                self.label_mapping[value] = self._metainfo['classes'].index(
                    'others')
        self.num_answers, answer_candidates = self.build_answer_candidates()
        self.answer_vocab = Answer(answer_candidates)
        self._metainfo.setdefault('answer_candidates', answer_candidates)
    def build_raw2nyu40class_label_mapping(self):
        label_classes2id = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
        self._metainfo['classes'] = list(label_classes2id.keys())
        
        label_classes_set = set(label_classes2id.keys())
        with open(os.path.join(self.data_root,'scannet/meta_data/scannetv2-labels.combined.tsv'), newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            raw2nyu40class = {row['raw_category']: row['nyu40class'] for row in reader}
        label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        for key, value in self._metainfo['categories'].items():
            nyu40class = raw2nyu40class[key]
            if nyu40class in self.dontcare_objects:
                label_mapping[value] = -1 #-1 is ignored
            elif nyu40class in label_classes_set:
                label_mapping[value]=label_classes2id[nyu40class]
            else:
                label_mapping[value]=label_classes2id['others']
        return label_mapping
    
    def build_answer_candidates(self):
        train_data = load(osp.join(self.data_root,'qa/ScanQA_v1.0_train.json'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_candidates = list(answer_counter.keys())
        num_answers = len(answer_candidates)
        print(f"total answers is {num_answers}")
        return num_answers, answer_candidates
    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    # need to compensate the scan_id info to the original pkl file
    def convert_info_to_scan(self):
        self.scans = dict()
        for data in self.data_list:
            scan_id = data['scan_id']
            self.scans[scan_id] = data
    def _ids2masks(self, ids, mask_length):
        """Change visible_instance_ids to visible_instance_masks."""
        masks = []
        for idx in range(len(ids)):
            mask = np.zeros((mask_length, ), dtype=bool)
            mask[ids[idx]] = 1
            masks.append(mask)
        return masks

    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        -1 indicates dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            if key in ['instances','gt_answer_labels']:
                img_filtered_annotations[key] = ann_info[key]
            elif key == 'visible_instance_masks':
                img_filtered_annotations[key] = []
                for idx in range(len(ann_info[key])):
                    img_filtered_annotations[key].append(
                        ann_info[key][idx][filter_mask])
            elif key in ['gt_occupancy', 'visible_occupancy_masks']:
                pass
            else:
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
        return img_filtered_annotations
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        self.process_metainfo()

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list
    def get_question_type(self,question):
        question = question.lstrip()
        if  'what' in question.lower():
            return 'what'
        elif 'where' in question.lower():
            return 'where'
        elif 'how' in question.lower():
            return 'how'
        elif 'which' in question.lower():
            return 'which'
        elif 'is' in question.lower():
            return 'is'
        else:
            return 'others' # others 
    def load_language_data(self):
        language_annotations = load(self.qa_file)
        # language_infos = [
        #     {
        #        'scan_id': 'scannet/' + anno['scene_id'],
        #        'question': anno['question'],
        #        'question_id': anno ['question_id']
        #     }
        #     for anno in language_annotations
        # ]
        # According to each object annotation,
        # find all objects in the corresponding scan
        language_infos = []
        for anno in mmengine.track_iter_progress(language_annotations):
            
            language_info = dict()
            language_info.update({
                'scan_id': 'scannet/' + anno['scene_id'],
                'question': anno['question'],
                'question_id': anno ['question_id']
            })
            data = self.scans[language_info['scan_id']]
            language_info['box_type_3d'] = data['box_type_3d']
            language_info['axis_align_matrix'] = data['axis_align_matrix']
            language_info['img_path'] = data['img_path']
            language_info['depth_img_path'] = data['depth_img_path']
            language_info['depth2img'] = data['depth2img']
            if 'cam2img' in data:
                language_info['cam2img'] = data['cam2img']
            language_info['scan_id'] = data['scan_id']
            language_info['depth_shift'] = data['depth_shift']
            language_info['depth_cam2img'] = data['depth_cam2img']

            scene_id = anno['scene_id']
            if 'test' in self.ann_file:
                language_info['clean_global_points_file_name'] = os.path.join(self.data_root,f'scannet/scannet_data',f'{scene_id}_vert.npy')
            else:
                language_info['clean_global_points_file_name'] = os.path.join(self.data_root,f'scannet/scannet_data',f'{scene_id}_aligned_vert.npy')
            
            ann_info = data['ann_info']

            # save the bounding boxes and corresponding labels
            language_anno_info = dict()
            
            language_anno_info['gt_bboxes_3d'] = ann_info['gt_bboxes_3d']  # BaseInstanceBboxes
            language_anno_info['gt_labels_3d'] = ann_info['gt_labels_3d']  # all box labels in the scan

            if 'answers' in anno:  # w/ ground truths
                answer_list = anno['answers']
                answer_id_list = [self.answer_vocab.stoi(answer) for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
                gt_answer_labels = np.zeros(self.num_answers, dtype=int)
                for i in answer_id_list:
                    gt_answer_labels[i] = 1
                language_anno_info['gt_answer_labels'] = gt_answer_labels
                language_info['answer_list'] = list(set(answer_list))
            else:
                language_anno_info['gt_answer_labels'] = np.zeros(self.num_answers, dtype=int)
                language_info['answer_list'] = []
            if 'object_ids' in anno:  # w/ ground truths
                bboxes_ids = ann_info['bboxes_ids']
                object_ids = anno['object_ids']
                target_objects_mask = np.zeros(len(ann_info['gt_labels_3d']), dtype=int)
                for i in object_ids:
                    target_objects_mask[bboxes_ids==i] = 1
                language_anno_info['target_objects_mask'] = target_objects_mask
            else:
                language_anno_info['target_objects_mask'] = np.zeros(0, dtype=int)
            if self.remove_dontcare:
                language_anno_info = self._remove_dontcare(language_anno_info)
            if not self.test_mode:
                language_info['ann_info'] = language_anno_info
            if self.test_mode and self.load_eval_anns:
                language_info['ann_info'] = language_anno_info
                language_info['eval_ann_info'] = language_info['ann_info']
                language_info['eval_ann_info'].update(dict(scan_id = 'scannet/' + anno['scene_id'],
                                                        question = anno['question'],
                                                        question_type = self.get_question_type(anno['question']),
                                                        question_id = anno ['question_id']),
                                                        answer_candidates = self._metainfo['answer_candidates'],
                                                      )
            language_infos.append(language_info)

        del self.scans

        return language_infos

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['box_type_3d'] = self.box_type_3d
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        # Because multi-view settings are different from original designs
        # we temporarily follow the ori design in ImVoxelNet
        info['img_path'] = []
        info['depth_img_path'] = []
        info['scan_id'] = info['sample_idx']
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0

        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info['images'])):
            img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                    info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                          info['images'][i]['depth_path'])

            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))
            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        info['depth2img'] = dict(extrinsic=extrinsics,
                                 intrinsic=cam2img,
                                 origin=np.array([.0, .0,
                                                  .5]).astype(np.float32))

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = info['ann_info']
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = None

        if 'instances' in info and len(info['instances']) > 0:
            # add s or gt prefix for most keys after concat
            # we only process 3d annotations here, the corresponding
            # 2d annotation process is in the `LoadAnnotations3D`
            # in `transforms`
            name_mapping = {
                'bbox_label_3d': 'gt_labels_3d',
                'bbox_label': 'gt_bboxes_labels',
                'bbox': 'gt_bboxes',
                'bbox_id': 'bboxes_ids',
                'bbox_3d': 'gt_bboxes_3d',
                'depth': 'depths',
                'center_2d': 'centers_2d',
                'attr_label': 'attr_labels',
                'velocity': 'velocities',
            }
            instances = info['instances']
            # empty gt
            if len(instances) == 0:
                return None
            else:
                keys = list(instances[0].keys())
                ann_info = dict()
                for ann_name in keys:
                    temp_anns = [item[ann_name] for item in instances]
                    # map the original dataset label to training label
                    if 'label' in ann_name and ann_name != 'attr_label':
                        temp_anns = [
                            self.label_mapping[item] for item in temp_anns
                        ]
                    if ann_name in name_mapping:
                        mapped_ann_name = name_mapping[ann_name]
                    else:
                        mapped_ann_name = ann_name

                    if 'label' in ann_name:
                        temp_anns = np.array(temp_anns).astype(np.int64)
                    elif ann_name in name_mapping:
                        temp_anns = np.array(temp_anns).astype(np.float32)
                    else:
                        temp_anns = np.array(temp_anns)

                    ann_info[mapped_ann_name] = temp_anns
                ann_info['instances'] = info['instances']

        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # post-processing/filtering ann_info if not empty gt
        if 'visible_instance_ids' in info['images'][0]:
            ids = []
            for i in range(len(info['images'])):
                ids.append(info['images'][i]['visible_instance_ids'])
            mask_length = ann_info['gt_labels_3d'].shape[0]
            ann_info['visible_instance_masks'] = self._ids2masks(
                ids, mask_length)

        # if self.remove_dontcare:
        #     ann_info = self._remove_dontcare(ann_info)
        ann_info['gt_bboxes_3d'] = self.box_type_3d(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5))
        return ann_info
