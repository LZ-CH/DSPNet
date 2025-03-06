_base_ = ['../default_runtime.py']
n_points = 40000
voxel_size = 0.01
use_color = True
use_clean_global_points=True

backend_args = None
if use_color:
    backbone_lidar_inchannels=6
else:
    backbone_lidar_inchannels=3
classes = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
            'window','bookshelf','picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'others')
model = dict(
    type='MultiViewVLMBase3DQA',
    voxel_size=voxel_size,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                        #    use_clip_mean_std = True,#VLM
                        #    use_imagenet_standard_mean_std=True,#BeiT
                           use_imagenet_default_mean_std = True,#Segformer,DinoV2,Swin
                           mean=[123.675, 116.28, 103.53], 
                           std=[58.394, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32,
                           furthest_point_sample=True, #only for test
                           num_points=n_points,
                           ),
    # backbone = dict(type='ViTModelWrapper', 
    #                 name='microsoft/beit-base-patch16-224-pt22k-ft22k',
    #                 out_channels=[768],
    #                 add_map=True,
    #                 used_hidden_layers=[12],
    #                 frozen=True,
    #                 ),
    backbone = dict(type='SwinModelWrapper', 
                    name='microsoft/swin-base-patch4-window7-224-in22k',
                    out_channels=[1024],
                    add_map=True,
                    frozen=True,
                    ),
    backbone_text = dict(type='TextModelWrapper', 
                         name='sentence-transformers/all-mpnet-base-v2', 
                         frozen=False,
                         ),
    # backbone_text = dict(type='TextModelWrapper', 
    #                      name='facebook/bart-base', 
    #                      frozen=False),

    backbone_fusion = dict(type='CrossModalityEncoder',
                           hidden_size=768, 
                           num_attention_heads=12,
                           num_hidden_layers = 4,
                           ),
    backbone_lidar=dict(
                    type='PointNet2SASSG',
                    in_channels=backbone_lidar_inchannels,
                    num_points=(2048, 1024, 512, 256),
                    radius=(0.2, 0.4, 0.8, 1.2),
                    num_samples=(64, 32, 16, 16),
                    sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                                (128, 128, 256)),
                    fp_channels=((256, 256), (256, 256)),
                    norm_cfg=dict(type='BN2d'),
                    sa_cfg=dict(
                        type='PointSAModule',
                        pool_mod='max',
                        use_xyz=True,
                        normalize_xyz=True),
                    frozen = False,
                    ),
    text_max_length=512,
    target_bbox_head=dict(type='RefLocHead',
                        bbox_coder=dict(
                            type='PartialBinBasedBBoxCoder',
                            num_sizes=18,
                            num_dir_bins=10,
                            with_rot=True,
                            mean_sizes=[[0.775, 0.949, 0.9654], 
                                        [1.869, 1.8321, 1.1922], 
                                        [0.6121, 0.6193, 0.7048], 
                                        [1.4411, 1.6045, 0.8365], 
                                        [1.0478, 1.2016, 0.6346], 
                                        [0.561, 0.6085, 1.7195], 
                                        [1.0789, 0.8203, 1.1692], 
                                        [0.8417, 1.3505, 1.6899], 
                                        [0.2305, 0.4764, 0.5657], 
                                        [1.4548, 1.9712, 0.2864], 
                                        [1.0786, 1.5371, 0.865], 
                                        [1.4312, 0.7692, 1.6498], 
                                        [0.6297, 0.7087, 1.3143], 
                                        [0.4393, 0.4157, 1.7], 
                                        [0.585, 0.5788, 0.7203], 
                                        [0.5116, 0.5096, 0.3129], 
                                        [1.1732, 1.0599, 0.5181], 
                                        [0.4329, 0.5193, 0.4844]]),
                        train_cfg=dict(
                            pos_distance_thr=0.3, neg_distance_thr=0.6),
                        num_classes = 1,
                        in_channels = 768,
                        hidden_channels = 768,
                        dropout = 0.3,
                        loss_weight=1.0,
                        ),
    target_cls_head=dict(type='RefClsHead',
                          num_classes = 18,
                          in_channels = 768*2,
                          hidden_channels = 768,
                          dropout = 0.3,
                          loss_weight=1.0,
                        ),
    qa_head = dict(type='QAHead',
                   num_classes=8864,
                   in_channels = 768,
                   hidden_channels = 768,
                   dropout=0.3,
                   ),
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mode='seed'),
    test_cfg=dict(
        sample_mode='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True),
    coord_type='DEPTH')

dataset_type = 'MultiViewScanQADataset'
data_root = 'data'

train_pipeline = [
    dict(type='LoadAnnotations3D',with_answer_labels=True,with_target_objects_mask=True),
    dict(type='MultiViewPipeline',
         n_images=20,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA',use_color=~use_clean_global_points&use_color),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(224, 224), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True, use_clean_global_points=use_clean_global_points,use_color=use_color),
    dict(type='PointSample', num_points=n_points),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[.9, 1.1],
         translation_std=[.1, .1, .1],
         shift_height=False),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d','gt_answer_labels','target_objects_mask'])
]
test_pipeline = [
    dict(type='LoadAnnotations3D',with_answer_labels=True),
    dict(type='MultiViewPipeline',
         n_images=20,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA',use_color=~use_clean_global_points&use_color),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(224, 224), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH', save_views_points=True, use_clean_global_points=use_clean_global_points,use_color=use_color),
    # dict(type='PointSample', num_points=n_points), #data_preprocessor will do this
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d','gt_answer_labels'])
]


# TODO: to determine a reasonable batch size
train_dataloader = dict(
    batch_size=12,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset',
                 times=1,
                 dataset=dict(type=dataset_type,
                              data_root=data_root,
                              ann_file='mv_scannetv2_infos_train.pkl',
                              qa_file='qa/ScanQA_v1.0_train.json',
                              metainfo = dict(classes=classes),
                              pipeline=train_pipeline,
                              anno_indices=None,
                              test_mode=False,
                              filter_empty_gt=True,
                              box_type_3d='Depth',
                              remove_dontcare=True)))

val_dataloader = dict(batch_size=12,
                      num_workers=12,
                      persistent_workers=True,
                      pin_memory=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='mv_scannetv2_infos_val.pkl',
                                   qa_file='qa/ScanQA_v1.0_val.json',
                                   metainfo = dict(classes=classes),
                                   pipeline=test_pipeline,
                                   anno_indices=None,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Depth',
                                   remove_dontcare=True))
test_dataloader = dict(batch_size=12,
                      num_workers=12,
                      persistent_workers=True,
                      pin_memory=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='mv_scannetv2_infos_val.pkl',
                                   qa_file='qa/ScanQA_v1.0_test_w_obj.json',
                                #    ann_file='mv_scannetv2_infos_test.pkl',
                                #    qa_file='qa/ScanQA_v1.0_test_wo_obj.json',
                                   metainfo = dict(classes=classes),
                                   pipeline=test_pipeline,
                                   anno_indices=None,
                                   test_mode=True,
                                   filter_empty_gt=False,
                                   box_type_3d='Depth',
                                   remove_dontcare=False))
val_evaluator = dict(type='ScanQAMetric',
                     )
test_evaluator = dict(type='ScanQAMetric',
                     format_only=True,
                    )


# training schedule for 1x
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
lr = 1e-4
optim_wrapper = dict(
                     type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=lr, weight_decay=1e-5),
                     paramwise_cfg=dict(
                         bypass_duplicate=True,
                         custom_keys={
                            'text_encoder': dict(lr_mult=0.1),
                         }),
                     clip_grad=dict(max_norm=10, norm_type=2),
                     accumulative_counts=1)
# learning rate
param_scheduler = [
    # 在 [0, max_epochs) Epoch时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=max_epochs,
         by_epoch=True,
         begin=0,
         end=max_epochs,
         convert_to_iter_based=True,
         eta_min_ratio=0.05),
    # 在 [0, 500) iter时使用线性学习率
    dict(type='LinearLR',
         start_factor=0.05,
         by_epoch=False,
         begin=0,
         end=500),
]

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True),
                ]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='EM@1',rule='greater', interval=1, max_keep_ckpts=3))


find_unused_parameters = True
load_from = './work_dirs/scannet-det/scannet-votenet-12xb12/epoch_12.pth'