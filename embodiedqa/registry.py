from mmengine import DATASETS as MMENGINE_DATASETS
from mmengine import METRICS as MMENGINE_METRICS
from mmengine import MODELS as MMENGINE_MODELS
from mmengine import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine import HOOKS as MMENGINE_HOOKS
from mmengine import Registry

MODELS = Registry('model',
                  parent=MMENGINE_MODELS,
                  locations=['embodiedqa.models'])
DATASETS = Registry('dataset',
                    parent=MMENGINE_DATASETS,
                    locations=['embodiedqa.datasets'])
TRANSFORMS = Registry('transform',
                      parent=MMENGINE_TRANSFORMS,
                      locations=['embodiedqa.datasets.transforms'])
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=['embodiedqa.eval'])
TASK_UTILS = Registry('task util',
                      parent=MMENGINE_TASK_UTILS,
                      locations=['embodiedqa.models'])
HOOKS = Registry( 'hook', parent=MMENGINE_HOOKS, locations=['embodiedqa.engine.hooks'])