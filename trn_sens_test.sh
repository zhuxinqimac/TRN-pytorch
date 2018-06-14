CUDA_VISIBLE_DEVICES=3 python score_sens.py \
    TRN_something_RGB_resnet34_TRN_segment8_best.pth.tar \
    something Something_results/sens --modality RGB \
    --arch resnet34 --num_segments 8 \
    --consensus_type TRN \

#CUDA_VISIBLE_DEVICES=3 python score_sens.py \
    #TRN_ucf101_RGB_resnet34_TRN_segment8_best.pth.tar \
    #ucf101 UCF101_results/sens --modality RGB \
    #--arch resnet34 --num_segments 8 \
    #--consensus_type TRN \
