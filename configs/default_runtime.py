default_scope = 'embodiedqa'
randomness = dict(seed=42,
                  diff_rank_seed=True)
default_hooks = dict(timer=dict(type='IterTimerHook'),
                     logger=dict(type='LoggerHook', interval=50),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook',
                                     interval=1,
                                     max_keep_ckpts=4),
                     sampler_seed=dict(type='DistSamplerSeedHook'))
# visualization=dict(type='Det3DVisualizationHook'))
compile_options = dict(backend='inductor', mode='max-autotune')
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# TODO: support auto scaling lr
