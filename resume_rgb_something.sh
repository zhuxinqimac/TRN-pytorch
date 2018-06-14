CUDA_VISIBLE_DEVICES=0,3 python main.py something RGB \
    --arch resnet34 --num_segments 8 \
    --resume TRN_something_train_shuffle_checkpoint.pth.tar \
    --store_name TRN_something_train_shuffle \
    --val_name  val_TRN_something_train_shuffle\
    --consensus_type TRN --batch-size 32 \
    --train_shuffle --val_shuffle
