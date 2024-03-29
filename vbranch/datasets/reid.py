import os
import json
import numpy as np
import cv2
from keras.applications import resnet50
from keras.applications import densenet
import math
import json

from ..utils import openpose as op

class DataGenerator(object):
    def __init__(self, dataset, split, pose_orientation=None, n_poses=2, verify=False,
            data_root='../data', keypoint_root='./openpose_output'):
        """
        Get data for Market-1501 and DukeMTMC datasets. Both use same file
        format. `files_dict` stores paths under each corresponding identity;
        `files_arr` stores paths as list of (path, idt, camera) tuples.
        Args:
            - dataset: 'market', 'duke'
            - split: 'train', 'test', 'query'
            - verify: if true, verify that each query has corresponding groud
            truth image in testing dataset; used for testing settting
            - data_root: path to folder containing market and duke folders
            - keypoint_root: path to folder containing keypoint data json files
        Raises:
            - AssertionError if invalid dataset or split"""

        assert dataset in ['market', 'duke']
        assert split in ['train', 'test', 'query']
        assert pose_orientation < n_poses, 'pose must be less than n_poses'

        self.dataset = dataset
        self.split = split
        self.files_dict, self.files_arr = self._get_data(dataset,
            split, data_root, keypoint_root, pose_orientation, n_poses)

        for idt in self.files_dict.keys():
            assert len(self.files_dict[idt]) > 0, f'idt {idt} has zero samples'

        if verify and split in ['test' , 'query']:
            self._verify(dataset)

    @staticmethod
    def _get_data(dataset, split, data_root, keypoint_root, pose_orientation, n_poses):
        files_dict = {}
        files_arr = []

        dir = os.path.join(data_root, dataset, split)
        keypoints_path = os.path.join(keypoint_root,dataset,split+'.json')
        with open(keypoints_path, 'r') as f:
            keypoint_data = json.load(f)

        for name in os.listdir(dir):
            if name[-4:] == '.jpg':
                idt = int(name[0:name.index('_')])
                # For testing, ignore irrelavent images
                if idt != -1:
                    if not any(idt == l for l in files_dict.keys()):
                        files_dict[idt] = []

                    path = os.path.join(dir, name)
                    camera = int(name[name.index('_')+2 : name.index('_')+3])
                    # if name in keypoint_data.keys():
                    #     pose = op.get_pose(keypoint_data[name], n_poses)
                    # else:
                    #     pose = -1
                    pose = op.get_pose_from_name(keypoint_data, name, n_poses)

                    if pose_orientation is None:
                        files_arr.append([path, idt, camera, pose])
                        files_dict[idt].append(path)
                    else:
                        if pose == pose_orientation:
                            files_arr.append([path, idt, camera, pose])
                            files_dict[idt].append(path)

        if pose_orientation is None:
            return files_dict, files_arr

        # Remove empty identities
        cleaned_files_dict = {}
        for idt in files_dict.keys():
            if len(files_dict[idt]) > 0:
                cleaned_files_dict[idt] = files_dict[idt]

        return cleaned_files_dict, files_arr

    @staticmethod
    def _verify(dataset):
        gallery_dict, gallery_arr = _get_data(dataset, 'test')
        query_dict, query_arr = _get_data(dataset, 'query')

        gallery_idts = np.array([p[1] for p in gallery_arr])
        gallery_cams = np.array([p[2] for p in gallery_arr])

        missing = 0
        for q in range(len(query_arr)):
            _, idt, camera = query_arr[q]

            b = np.logical_or(gallery_cams != camera, gallery_idts != idt)

            # Verify exists a valid instance in the gallery set
            i = 0
            for _, idt_t, cam_t in np.array(gallery_arr)[b]:
                if idt == int(idt_t) and camera != int(cam_t):
                    i += 1
            if i == 0:
                missing += 1

        if missing > 0:
            print("Warning: {} query samples missing ground-truth samples.".\
                format(missing))
            return

        print("Verification successful! All query have ground-truth samples.")

class TripletDataGenerator(DataGenerator):
    """Returns object that randomly samples triplets from the dataset"""

    def __init__(self, dataset, split, P, K,
            preprocess=True, img_dim=(128,64,3),
            crop=False, flip=True,
            flatten=True, labels=False, pose_orientation=None):
        super().__init__(dataset, split, pose_orientation)
        self.P = P
        self.K = K
        self.preprocess = preprocess
        self.img_dim = img_dim
        self.crop = crop
        self.flip = flip
        self.flatten = True
        self.labels = labels

        self.print_config()

    def __next__(self):
        return self.sample(self.P, self.K, self.preprocess, self.img_dim,
            self.crop, self.flip, self.flatten, self.labels)

    def sample(self, P, K,
            preprocess=True, img_dim=(256,128,3),
            crop=False, flip=True,
            flatten=True, labels=False):
        """
        Args:
            - P: number of identities
            - K: number of samples per identity
            - preprocess: if true, apply imagenet preprocessing (resnet)
            - img_dim: image dimension of batch output, resizes if necessary
            - crop: if true, apply random croppings of original image
            - flip: if true, apply random horizontal flips of image
            - flatten: if true, flatten batch to size (P*K, height, width, channels)
            must be true when generating batches for training
            - labels: if true, return corresponding identities of batch images
        Returns:
            - np array (batch)
            - np array (labels, if `labels` is true)
        """
        batch = []
        im_labels = []

        idt_choice = self._sample(list(self.files_dict.keys()), P, False)
        for p in range(P):
            k_choice = self._sample(len(self.files_dict[idt_choice[p]]),
                K, replace=K > len(self.files_dict[idt_choice[p]]))
            for k in k_choice:
                path = self.files_dict[idt_choice[p]][k]

                if crop:
                    im = self._imread_crop(path, img_dim, preprocess)
                else:
                    im = _imread_scale(path, img_dim, preprocess)

                if flip and np.random.random() < 0.5:
                    im = np.flip(im, axis=1)

                batch.append(im)
                if labels:
                    im_labels.append(list(self.files_dict.keys()).\
                        index(idt_choice[p]))

        batch = np.stack(batch)
        if not flatten:
            batch = np.reshape(batch, (P, K) + batch.shape[1:])
        if labels:
            return batch, np.array(im_labels)

        return batch

    @staticmethod
    def _sample(arr, n, replace=False):
        return np.random.choice(arr, n, replace=replace)

    @staticmethod
    def _imread_crop(path, shape, preprocess):
        """Read image from path and take random crops"""
        crop_x = np.random.randint(0.125 * shape[1])
        crop_y = np.random.randint(0.125 * shape[0])

        new_shape = (int(1.125 * shape[0]), int(1.125 * shape[1]))
        im = _imread_scale(path, new_shape, preprocess)
        return im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]

    def print_config(self):
        print('TripletDataGenerator CONFIG')
        print('Dataset:\t', self.dataset)
        print('Split:\t\t', self.split)
        print('Preprocess:\t', self.preprocess)
        print('Dimension:\t', self.img_dim)
        print('Crop:\t\t', self.crop)
        print('Flip:\t\t', self.flip)
        print('Flatten:\t', self.flatten)
        print('Labels:\t\t', self.labels)

class TestingDataGenerator(DataGenerator):
    def __init__(self, dataset, split,
            preprocess=None, img_dim=(128,64,3),
            crop=False, flip=True, buffer=1000,
            pose_orientation=None):
        """
        Creates iterable object over the specified split of the dataset
        (can be `train` split, despite name of class).
        Args (see TripletDataGenerator):
            - crop: if true, take central crop of image
            - buffer: approx size of batch to return at each iteration, more
            evenly distributed buffer is calculated to improve accuracy of
            batch norm statistics, but not greater than specified buffer"""

        super().__init__(dataset, split, pose_orientation)

        self.preprocess = preprocess
        self.img_dim = img_dim
        self.crop = crop
        self.flip = flip

        n_batches =  math.ceil(len(self.files_arr) / buffer)
        self.buffer = math.ceil(len(self.files_arr) / n_batches)
        self.current = 0
        self.high = len(self.files_arr)
        self.n = 2 if flip else 1

        self.print_config()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.high:
            raise StopIteration
        else:
            img_buffer = []
            if self.n > 1:
                img_buffer_flip = []

            next = self.current + self.buffer

            for i in range(self.current, min(next, self.high)):
                path = self.files_arr[i][0]

                if self.crop:
                    im = self._imread_crop(path, self.img_dim, self.preprocess)
                else:
                    im = _imread_scale(path, self.img_dim, self.preprocess)

                if self.flip:
                    im_flip = np.flip(im, axis=1)
                    img_buffer_flip.append(im_flip)

                img_buffer.append(im)

            self.current = next

            if self.n > 1:
                batch = [np.stack(img_buffer), np.stack(img_buffer_flip)]
            else:
                batch = np.stack(img_buffer)

            return batch

    @staticmethod
    def _imread_crop(path, shape, preprocess):
        """Read image and take central + corner crops"""
        crop_x = shape[1] // 16
        crop_y = shape[0] // 16

        new_shape = (int(1.125 * shape[0]), int(1.125 * shape[1]))
        im = _imread_scale(path, new_shape, preprocess)
        return im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]

    def print_config(self):
        print('TestingDataGenerator CONFIG')
        print('Dataset:\t', self.dataset)
        print('Split:\t\t', self.split)
        print('Preprocess:\t', self.preprocess)
        print('Dimension:\t', self.img_dim)
        print('Crop:\t\t', self.crop)
        print('Flip:\t\t', self.flip)
        print('Buffer:\t\t', self.buffer)

def _imread(img_path):
    """
    returns RGB image
    misc.imread is deprecated
    """
    im = cv2.imread(img_path)
    if len(im.shape) == 3 and im.shape[-1] == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def _imread_scale(img_path, shape, preprocess=True):
    """Read image and resize if necessary to match shape"""

    im = _imread(img_path)
    if im.shape[:2] != shape[:2]:
        # print('resize')
        im = cv2.resize(im, (shape[1], shape[0]))

    if preprocess == 'resnet':
        return resnet50.preprocess_input(im.astype('float32'))
    elif preprocess == 'densenet':
        return densenet.preprocess_input(im.astype('float32'))
    elif preprocess == True:
        im = im / 127.5 - 1

    return im
