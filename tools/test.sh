CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/test.py \
configs/scanqa/mv-scanqa-pointnetpp-swin-sbert-12xb12.py work_dirs/mv-scanqa/mv-scanqa-pointnetpp-swin-sbert-12xb12/best_EM@1_epoch_<epoch_id>.pth \
--work-dir=work_dirs/scanqa_test  --launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/test.py \
configs/sqa/mv-sqa-pointnetpp-swin-sbert-12xb12.py work_dirs/mv-sqa/mv-sqa-pointnetpp-swin-sbert-12xb12/best_EM@1_epoch_<epoch_id>.pth \
--work-dir=work_dirs/sqa_test  --launcher pytorch
