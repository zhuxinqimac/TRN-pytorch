CUDA_VISIBLE_DEVICES=0,3 python main.py ucf101 RGB \
    --arch resnet34 --num_segments 8 \
    --store_name TRN_ucf101_train_shuffle \
    --val_name TRN_ucf101_train_shufffle \
    --consensus_type TRN --batch-size 32 --train_shuffle
