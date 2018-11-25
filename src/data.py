import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.backend as K

class DataGenerator():
    def __init__(self, dataset, split, camera=False):
        self.DATA_ROOT = '/home/gong/research/data'
        self.dataset = dataset
        self.camera = camera

        files_dict, files_arr = self._get_data(dataset, split, camera)
        self.files_dict = files_dict
        self.files_arr = files_arr

    def _get_data(self, dataset, split, camera):
        assert dataset in ['market', 'cuhk03', 'duke']

        files_dict = {}
        files_arr = []

        if dataset == 'market' or dataset == 'duke':
            if split in ['train', 'test', 'query']:
                name_dict = {
                    'train' : 'train', # 'bounding_box_train',
                    'test'  : 'test', # 'bounding_box_test',
                    'query' : 'query'
                }
                dir = os.path.join(self.DATA_ROOT, dataset,
                    name_dict.get(split, "invalid split for market dataset"))

        for f in os.listdir(dir):
            if f[-4:] == '.jpg':
                idt = int(f[0:f.index('_')])
                if idt != -1: # For testing, ignore irrelavent images
                    if not any(idt == l for l in files_dict.keys()):
                        if camera:
                            files_dict[idt] = {}
                        else:
                            files_dict[idt] = []

                    path = os.path.join(dir, f)

                    if camera:
                        if dataset == 'market' or dataset == 'duke':
                            camera = f[f.index('_') + 2 : f.index('_') + 3]
                        # elif dataset == 'cuhk03':
                        #     camera = f[len(f) - f[::-1].index('_'):-4]

                        files_arr.append([path, idt, int(camera)])
                        try:
                            files_dict[idt][int(camera)].append(path)
                        except:
                            files_dict[idt][int(camera)] = []
                            files_dict[idt][int(camera)].append(path)
                    else:
                        files_arr.append([path, idt])
                        files_dict[idt].append(path)

        return files_dict, files_arr

class TripletDataGenerator(DataGenerator):
    def __init__(self, dataset, split, preprocess=None, img_dim=(128,64,3),
            P=4, K=4, crop=False, flip=False):
        self.img_dim = img_dim
        self.P = P
        self.K = K
        self.crop = crop
        self.flip = flip
        self.preprocess = preprocess

        super().__init__(dataset, split)

    def __iter__(self):
        return self

    def __next__(self):
        input_imgs = []
        labels = []

        idt_choice = np.random.choice(list(self.files_dict.keys()), self.P, replace=False)
        for p in range(self.P):
            k_choice = np.random.choice(len(self.files_dict[idt_choice[p]]),
                self.K, replace=self.K > len(self.files_dict[idt_choice[p]]))

            for k in k_choice:
                path = self.files_dict[idt_choice[p]][k]

                crop_x = np.random.randint(0.125 * self.img_dim[1])
                crop_y = np.random.randint(0.125 * self.img_dim[0])

                if self.crop:
                    shape = (int(1.125 * self.img_dim[0]), int(1.125 * self.img_dim[1]))
                    im = _imread_scale(path, shape, self.preprocess)
                    im = im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]
                else:
                    im = _imread_scale(path, self.img_dim, self.preprocess)

                if self.flip:
                    if np.random.random() < 0.5:
                        im = np.flip(im, axis=1)

                input_imgs.append(im)
                labels.append(list(self.files_dict.keys()).index(idt_choice[p]))

        return np.stack(input_imgs), np.stack(labels)[:, np.newaxis]

    def sample(self, width=None, height=None, plot=True, figsize=(10,10)):
        sample_batch, sample_labels = next(self)

        if plot:
            if height:
                n_rows = min(self.P, height)
            else:
                n_rows = self.P

            if width:
                n_cols = min(self.K, width)
            else:
                n_cols = self.K

            plt.figure(figsize=figsize)
            for row in range(n_rows):
                for col in range(n_cols):
                    n = row * n_cols + col
                    plt.subplot(n_rows, n_cols, n + 1)
                    plt.imshow(sample_batch[n])
                    plt.title('idx : %d' % sample_labels[n])
                    plt.axis('off')
            plt.show()

        return sample_batch, sample_labels

class TestingDataGenerator(DataGenerator):
    def __init__(self, dataset, split, preprocess=None, img_dim=(128,64,3),
            crop=False, flip=False, buffer=100):
        super().__init__(dataset, split, True)

        self.preprocess = preprocess
        self.img_dim = img_dim
        self.crop = crop
        self.flip = flip
        self.buffer = buffer

        self.current = 0
        self.high = len(self.files_arr)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.high:
            raise StopIteration
        else:
            img_buffer = []
            next = self.current + self.buffer

            for i in range(self.current, min(next, self.high)):
                path = self.files_arr[i][0]

                crop_x = np.random.randint(0.125 * self.img_dim[1])
                crop_y = np.random.randint(0.125 * self.img_dim[0])

                if self.crop:
                    shape = (int(1.125 * self.img_dim[0]), int(1.125 * self.img_dim[1]))
                    im = _imread_scale(path, shape, self.preprocess)
                    im = im[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]
                else:
                    im = _imread_scale(path, self.img_dim, self.preprocess)

                if self.flip:
                    if np.random.random() < 0.5:
                        im = np.flip(im, axis=1)

                img_buffer.append(im)

            self.current = next
            return img_buffer


def _imread(img_path):
    """
    returns RGB image
    misc.imread is deprecated
    """
    im = cv2.imread(img_path)
    if len(im.shape) == 3 and im.shape[-1] == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        return im

def _imread_scale(img_path, shape, preprocess='densenet'):
    im = _imread(img_path)
    if im.shape[:2] != shape[:2]:
        # print('resize')
        im = cv2.resize(im, (shape[1], shape[0]))

    if preprocess == 'densenet':
        return _densenet_preprocess(im.astype('float64'))
    elif preprocess == 'norm':
        return _preprocess_norm(im)

    return im

def _preprocess_norm(x):
    return x / 127.5 - 1

def _densenet_preprocess(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.

    https://github.com/titu1994/DenseNet
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x
