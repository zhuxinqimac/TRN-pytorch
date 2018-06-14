CUDA_VISIBLE_DEVICES=1 python test_models.py ucf101 RGB \
    model/TRN_ucf101_RGB_resnet34_TRN_segment8_best.pth.tar \
    --arch resnet34 --crop_fusion_type TRN --test_segments 8 \
    --save_score ucf101_standard_test
