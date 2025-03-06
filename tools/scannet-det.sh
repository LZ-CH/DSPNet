CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/train.py \
configs/scannet-det/scannet-votenet-12xb12.py --work-dir=work_dirs/scannet-det/scannet-votenet-12xb12 --launcher pytorch

