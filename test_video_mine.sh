# Make prediction from mp4 video file (ffmpeg is required)
#python test_video.py --video_file sample_data/juggling.mp4 --rendered_output sample_data/predicted_video.mp4 --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar --arch InceptionV3 --dataset moments

# Make prediction with input a a folder name with RGB frames
<<<<<<< HEAD
python test_video.py --frame_folder sample_data/juggling_frames \
    --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar \
    --arch InceptionV3 --dataset moments
=======
python test_video.py --video_file bolei_juggling.mp4 --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar --arch InceptionV3 --dataset moments
>>>>>>> 40634cd6ffc2de0cd23a89dd18d4d3b13b9d80e7
