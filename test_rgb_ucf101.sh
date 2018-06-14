CUDA_VISIBLE_DEVICES=3 python main.py ucf101 RGB \
    --arch resnet34 --num_segments 8 \
    --resume TRN_ucf101_train_shuffle_best.pth.tar \
    --val_name TRN_ucf101_train_shufffle_s_s \
    --consensus_type TRN --batch-size 32 \
    --evaluate --val_shuffle
