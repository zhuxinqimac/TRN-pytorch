CUDA_VISIBLE_DEVICES=0,3 python main.py something RGB \
    --arch resnet34 --num_segments 8 \
    --resume TRN_something_train_shuffle_best.pth.tar \
    --store_name train_seg8_res34_s_n \
    --val_name val_seg_8_res34_s_n \
    --consensus_type TRN --batch-size 32 \
    --evaluate
