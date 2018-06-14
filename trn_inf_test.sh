CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    TRN_something_RGB_resnet34_TRN_segment8_best.pth.tar \
    TRN_something_train_shuffle_best.pth.tar \
    something Something_results/inf --modality RGB \
    --arch resnet34 --num_segments 8 \
    --consensus_type TRN

#CUDA_VISIBLE_DEVICES=3 python score_inf.py \
    #TRN_ucf101_RGB_resnet34_TRN_segment8_best.pth.tar \
    #TRN_ucf101_train_shuffle_best.pth.tar \
    #ucf101 UCF101_results/inf --modality RGB \
    #--arch resnet34 --num_segments 8 \
    #--consensus_type TRN \
