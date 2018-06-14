import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, 
                 temp_transform=None, 
                 score_sens_mode=False, 
                 score_inf_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.temp_transform = temp_transform
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.score_sens_mode = score_sens_mode
        self.score_inf_mode = score_inf_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx-1)*5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def _get_normal_plus_shuffle(self, record):
        segment_indices = self._get_val_indices(record)
        norm_images = list()
        abnorm_images = list()
        idx_list = list()
        for seg_ind in segment_indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                idx_list.append(p)
                seg_imgs = self._load_image(record.path, p)
                norm_images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        # print('before:', idx_list)
        ab_idx_list = self.temp_transform(idx_list)
        # print('after: ', ab_idx_list)
        # input('...')
        for p in ab_idx_list:
            seg_imgs = self._load_image(record.path, p)
            abnorm_images.extend(seg_imgs)

        trans_norm_images = self.transform(norm_images)
        trans_abnorm_images = self.transform(abnorm_images)
        # print('trans_norm_images.size():', trans_norm_images.size())
        # print('trans_abnorm_images.size():', trans_abnorm_images.size())
        # input('...')
        return [trans_norm_images, trans_abnorm_images, idx_list, ab_idx_list, \
                record.path], \
                    record.label

    def _get_normal_inf(self, record):
        images = list()
        idx_list = list()
        indices = self._get_val_indices(record)
        # print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                idx_list.append(p)
                # seg_imgs = self._load_image(record.path, p)
                # images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # print('before temp trans: ', idx_list)
        # print(record.path)
        process_idx_list = self.temp_transform(idx_list)
        # print(process_idx_list)
        # input('...')
        for p in process_idx_list:
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return [process_data, record.path], record.label

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, 
            self.image_tmpl.format(1))):
            # print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if self.score_sens_mode:
            return self._get_normal_plus_shuffle(record)
        elif self.score_inf_mode:
            return self._get_normal_inf(record)
        elif not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)
    
    def get(self, record, indices):

        images = list()
        idx_list = list()
        # print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                idx_list.append(p)
                # seg_imgs = self._load_image(record.path, p)
                # images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # print('before temp trans: ', idx_list)
        process_idx_list = self.temp_transform(idx_list)
        # print(process_idx_list)
        # input('...')
        for p in process_idx_list:
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    # def get(self, record, indices):

        # images = list()
        # for seg_ind in indices:
            # p = int(seg_ind)
            # for i in range(self.new_length):
                # seg_imgs = self._load_image(record.path, p)
                # images.extend(seg_imgs)
                # if p < record.num_frames:
                    # p += 1

        # process_data = self.transform(images)
        # return process_data, record.label

    def __len__(self):
        return len(self.video_list)
