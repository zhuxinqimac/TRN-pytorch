CUDA_VISIBLE_DEVICES=0,3 python main.py ucf101 RGB \
    --arch resnet34 --num_segments 8 \
    --resume TRN_ucf101_RGB_resnet34_TRN_segment8_checkpoint.pth.tar \
    --consensus_type TRN --batch-size 32
