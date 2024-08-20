CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det 2
CUDA_VISIBLE_DEVICES=0 pyd tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/debug 
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det
CUDA_VISIBLE_DEVICES=1,4,5,7 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det 4
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det 2 --resume-from work_dirs/totaltext_det/latest.pth
CUDA_VISIBLE_DEVICES=1 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det
CUDA_VISIBLE_DEVICES=1,4,5,7 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det 4
CUDA_VISIBLE_DEVICES=6 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det-topk1
CUDA_VISIBLE_DEVICES=0 pyd tools/test.py configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det/best_det-hmean_epoch_98.pth --eval hmean-e2e

CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det 1 --resume-from work_dirs/totaltext_det_topk1/latest.pth
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det-topk7
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/lranet_ctw1500_det.py  --work-dir work_dirs/ctw1500-det2
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det-topk9
CUDA_VISIBLE_DEVICES=0 pyd tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det-debug
CUDA_VISIBLE_DEVICES=0 ./tools/train.py configs/lranet/lranet_totaltext_det.py  --work-dir work_dirs/totaltext_det-baseline2-ddas-k=3-shuffle
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/lranet/lranet_totaltext_det.py work_dirs/totaltext_det-baseline2-k=3-1 2 

CUDA_VISIBLE_DEVICES=0 pyd tools/train.py configs/lranet/chenchen_spinal_det.py  --work-dir work_dirs/debug 
