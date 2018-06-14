import argparse
import torch
import os
import time
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from dataset import TSNDataSet
from models import TSN
from transforms import *
# from temporal_transforms import ReverseFrames, ShuffleFrames
from main import AverageMeter, accuracy
import datasets_video

def evaluate(test_loader, model1, model2, criterion, eval_logger, softmax, 
        analysis_recorder):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    s_n_top1 = AverageMeter()
    s_n_top5 = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        # print(inputs.size())
        # input('...')
        input_var = torch.autograd.Variable(inputs[0], volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        v_path = inputs[1][0].replace(' ', '-')
        # compute output
        output1 = model1(input_var)
        output2 = model2(input_var)
        # print(output1.size())
        # print(output2.size())
        # input('...')
        output1_b = softmax(output1)
        output2_b = softmax(output2)
        # loss = criterion(output, target_var)
        loss = criterion(output1_b, output2_b)
        loss = torch.sqrt(loss)

        prec1, prec5 = accuracy(output1.data, target, topk=(1,5))
        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)
        prec1, prec5 = accuracy(output2.data, target, topk=(1,5))
        s_n_top1.update(prec1[0], 1)
        s_n_top5.update(prec5[0], 1)

        _, n_n_pred = output1_b.max(1)
        _, s_n_pred = output2_b.max(1)
        GT_class_name = class_to_name[target.cpu().numpy()[0]]
        if (n_n_pred.data == target).cpu().numpy():
            if_correct = 1
        else:
            if_correct = 0
        # print('v_path:', v_path)
        # print('n_n_pred:', n_n_pred)
        # print('s_n_pred:', s_n_pred)
        # print('target:', target)
        # print('GT_class_name:', GT_class_name)
        # print('if_correct:', if_correct)
        # input('...')

        losses.update(loss.data[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        analysis_data_line = ('{path} {if_correct} {loss.val:.4f} '
                '{GT_class_name} {GT_class_index} '
                '{n_n_pred} {s_n_pred}'.format(
                    path=v_path, if_correct=if_correct, loss=losses, 
                    GT_class_name=GT_class_name, 
                    GT_class_index=target.cpu().numpy()[0], 
                    n_n_pred=n_n_pred.data.cpu().numpy()[0], 
                    s_n_pred=s_n_pred.data.cpu().numpy()[0]))
        with open(analysis_recorder, 'a') as f:
            f.write(analysis_data_line+'\n')

        if i % 20 == 0:
            log_line = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  's_n_Prec@1 {s_n_top1.val:.3f} ({s_n_top1.avg:.3f})\t'
                  's_n_Prec@5 {s_n_top5.val:.3f} ({s_n_top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses, 
                   top1=top1, top5=top5, s_n_top1=s_n_top1, s_n_top5=s_n_top5))
            print(log_line)
            # eval_logger.write(log_line+'\n')
            with open(eval_logger, 'a') as f:
                f.write(log_line+'\n')

    # print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          # .format(top1=top1, top5=top5, loss=losses)))
    log_line = ('Testing Results: Loss {loss.avg:.5f}'
          .format(loss=losses))
    print(log_line)
    with open(eval_logger, 'a') as f:
        f.write(log_line+'\n\n')

    return


class FeatureMapModel(torch.nn.Module):
    def __init__(self, whole_model, consensus_type, modality):
        super(FeatureMapModel, self).__init__()
        self.modality = modality
        if self.modality == 'RGB':
            self.new_length = 1
        else:
            self.new_length = 10
        if consensus_type == 'bilinear_att' or consensus_type == 'conv_lstm':
            self.base_model = whole_model.module.base_model
        elif consensus_type == 'lstm' or consensus_type == 'ele_multi':
            removed = list(whole_model.module.base_model.children())[:-1]
            self.base_model = torch.nn.Sequential(*removed)
        elif consensus_type == 'avg' or consensus_type == 'max':
            removed = list(whole_model.module.base_model.children())[:-2]
            self.base_model = torch.nn.Sequential(*removed)
        else:
            ValueError(('Not supported consensus \
                        type {}.'.format(self.consensus_type)))
        print(self.base_model)

    def forward(self, inputs):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # print(input.size())
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())

        base_out = self.base_model(inputs.view((-1, 
                        sample_len) + inputs.size()[-2:]))
        base_out = base_out.view((-1, self.num_segments) + \
                        base_out.size()[-3:])
        base_out = base_out.mean(dim=1)
        return base_out.squeeze(1)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                "Get a model's sensitivity to temporal info.")
    parser.add_argument('first_model_path', type=str, 
                help='Path to a pretrained model.')
    parser.add_argument('second_model_path', type=str, 
                help='Path to a pretrained model.')
    parser.add_argument('dataset', type=str, 
                choices=['ucf101', 'something'])
    parser.add_argument('result_path', default='result', type=str,
                        metavar='LOG_PATH', help='results and log path')
    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--modality', type=str, default='RGB', 
                choices=['RGB', 'Flow'])
    parser.add_argument('--arch', type=str, default="resnet34")
    parser.add_argument('--consensus_type', type=str, default='TRN')

    # ====== Modified ======
    # parser.add_argument('--scale_size', default=256, type=int, 
                        # help='size to be scaled before crop (default 256)')
    # parser.add_argument('--crop_size', default=224, type=int, 
                        # help='size to be cropped to (default 224)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--compared_temp_transform', default='shuffle', 
                        type=str, help='temp transform to compare', 
                        choices=['shuffle', 'reverse'])
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_feature_dim', default=256, type=int, 
                        help="the feature dimension for each frame")

    
    args = parser.parse_args()
    args.first_model_path = os.path.join('model', args.first_model_path)
    args.second_model_path = os.path.join('model', args.second_model_path)
    if args.dataset == 'ucf101':
        num_class = 101
        args.train_list = '../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt'
        args.val_list = '../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt'
        args.root_path = '/'
        with open('video_datasets/ucf101/classInd.txt', 'r') as f:
            content = f.readlines()
        class_to_name = {int(line.strip().split(' ')[0])-1:line.strip().split(' ')[1] for line in content}
        prefix = 'image_{:05d}.jpg'
    else:
        categories, args.train_list, args.val_list, args.root_path, prefix = \
            datasets_video.return_dataset(args.dataset, args.modality)
        class_to_name = {i:name.replace(' ', '-') for i, name in enumerate(categories)}
        num_class = len(categories)
    
    print(class_to_name)
    # input('...')
    first_model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, 
                dropout=0.8, 
                img_feature_dim=args.img_feature_dim)
    second_model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, 
                dropout=0.8, 
                img_feature_dim=args.img_feature_dim)
    crop_size = first_model.crop_size
    scale_size = first_model.scale_size
    input_mean = first_model.input_mean
    input_std = first_model.input_std

    first_model = torch.nn.DataParallel(first_model, 
                device_ids=args.gpus).cuda()
    second_model = torch.nn.DataParallel(second_model, 
                device_ids=args.gpus).cuda()

    if os.path.isfile(args.first_model_path):
        print(("=> loading checkpoint '{}'".format(args.first_model_path)))
        checkpoint = torch.load(args.first_model_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        first_model.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint epoch {}"
              .format(checkpoint['epoch'])))
    else:
        ValueError(('No check point found at "{}"'.format(args.first_model_path)))

    if os.path.isfile(args.second_model_path):
        print(("=> loading checkpoint '{}'".format(args.second_model_path)))
        checkpoint = torch.load(args.second_model_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        second_model.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint epoch {}"
              .format(checkpoint['epoch'])))
    else:
        ValueError(('No check point found at "{}"'.format(args.second_model_path)))

    # f_model = FeatureMapModel(first_model, args.consensus_type, args.modality)
    # s_model = FeatureMapModel(second_model, args.consensus_type, args.modality)
    f_model = first_model
    # print(f_model)
    s_model = second_model
    # input('...')

    f_model = torch.nn.DataParallel(f_model, device_ids=args.gpus).cuda()
    s_model = torch.nn.DataParallel(s_model, device_ids=args.gpus).cuda()
    # f_model = torch.nn.DataParallel(f_model.cuda(devices[0]), device_ids=devices)

    # if args.compared_temp_transform == 'shuffle':
        # temp_transform = ShuffleFrames()
    # else:
        # temp_transform = ReverseFrames()
    normalize = GroupNormalize(input_mean, input_std)
    temp_transform  = IdentityTransform()
    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=1,
                   modality=args.modality,
                   image_tmpl=prefix,
                   temp_transform=temp_transform, 
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ]), 
                   score_inf_mode=True),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True

    # if args.measure_type == 'KL':
        # criterion = torch.nn.KLDivLoss().cuda()
    # else:
        # criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.MSELoss().cuda()
    softmax = torch.nn.Softmax(1).cuda()

    eval_logger = os.path.join(args.result_path, 'inf_log.log')
    with open(eval_logger, 'w') as f:
        f.write('')
    analysis_recorder = os.path.join(args.result_path, 'inf_analysis.txt')
    with open(analysis_recorder, 'w') as f:
        f.write('')
    evaluate(test_loader, f_model, s_model, criterion, eval_logger=eval_logger, 
            softmax=softmax, analysis_recorder=analysis_recorder)
    
