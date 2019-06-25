import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DataGenerator(object):
    def __init__(self, dataset, split, verify=False, DATA_ROOT='../data'):
        self.dataset = dataset
        self.DATA_ROOT = DATA_ROOT
        # self.camera = camera

        self.files_dict, self.files_arr = self._get_data(dataset, split)

        if verify and split in ['test' , 'query']:
            self._verify(dataset)

    def _get_data(self, dataset, split):
        assert dataset in ['market', 'cuhk03', 'duke']
        assert split in ['train', 'test', 'query']

        files_dict = {}
        files_arr = []

        name_dict = {
            'train' : 'train', # 'bounding_box_train',
            'test'  : 'test', # 'bounding_box_test',
            'query' : 'query'
        }

        if dataset == 'market' or dataset == 'duke':
            dir = os.path.join(self.DATA_ROOT, dataset, name_dict[split])

        for f in os.listdir(dir):
            if f[-4:] == '.jpg':
                idt = int(f[0:f.index('_')])
                # For testing, ignore irrelavent images
                if idt != -1:
                    if not any(idt == l for l in files_dict.keys()):
                        files_dict[idt] = []

                    path = os.path.join(dir, f)
                    camera = f[f.index('_') + 2 : f.index('_') + 3]
                    files_arr.append([path, idt, int(camera)])
                    files_dict[idt].append(path)

        return files_dict, files_arr

    @staticmethod
    def _verify(dataset):
        gallery_dict, gallery_arr = _get_data(dataset, 'test')
        query_dict, query_arr = _get_data(dataset, 'query')

        gallery_idts = np.array([p[1] for p in gallery_arr])
        gallery_cams = np.array([p[2] for p in gallery_arr])

        missing = 0
        for q in range(len(query_arr)):
            idt, camera = int(query_arr[q][1]), int(query_arr[q][2])

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
    def __init__(self, dataset, split):
        super().__init__(dataset, split)

    def next(self, P, K, preprocess=True, img_dim=(256,128,3), crop=False,
            flip=False, flatten=True, labels=False):
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
                if flip:
                    if np.random.random() < 0.5:
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
        crop_x = np.random.randint(0.125 * shape[1])
        crop_y = np.random.randint(0.125 * shape[0])

        new_shape = (int(1.125 * shape[0]), int(1.125 * shape[1]))
        im = _imread_scale(path, new_shape, preprocess)
        return im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]

class TestingDataGenerator(DataGenerator):
    def __init__(self, dataset, split, preprocess=None, img_dim=(128,64,3),
            crop=False, flip=False, buffer=100):
        super().__init__(dataset, split)

        self.preprocess = preprocess
        self.img_dim = img_dim
        self.crop = crop
        self.flip = flip
        self.buffer = buffer

        self.current = 0
        self.high = len(self.files_arr)
        self.n = 2 if flip else 1

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

            batch = np.stack(img_buffer)
            return batch

    @staticmethod
    def _imread_crop(path, shape, preprocess):
        crop_x = shape[1] // 16
        crop_y = shape[0] // 16

        new_shape = (int(1.125 * shape[0]), int(1.125 * shape[1]))
        im = _imread_scale(path, new_shape, preprocess)
        return im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]

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
    im = _imread(img_path)
    if im.shape[:2] != shape[:2]:
        # print('resize')
        im = cv2.resize(im, (shape[1], shape[0]))

    if preprocess:
        # return _densenet_preprocess(im.astype('float64'))
        return _imagenet_preprocess(im)
    # elif preprocess == 'norm':
    #     return _preprocess_norm(im)

    return im

def _densenet_preprocess(x):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.

    https://github.com/titu1994/DenseNet
    """

    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x

def _imagenet_preprocess(x):
    # return _preprocess_numpy_input(x, 'channels_last', 'tf')
    return x / 127.5 - 1
