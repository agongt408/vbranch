import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.backend as K

class DataGenerator():
    def __init__(self, dataset, split, verify=False):
        self.DATA_ROOT = '/home/gong/research/data'
        self.dataset = dataset
        # self.camera = camera

        files_dict, files_arr = self._get_data(dataset, split)
        self.files_dict = files_dict
        self.files_arr = files_arr

        if verify and split in ["test" , "query"]:
            self._verify(dataset)

    def _get_data(self, dataset, split):
        assert dataset in ['market', 'cuhk03', 'duke']

        files_dict = {}
        files_arr = []

        if dataset == 'market' or dataset == 'duke':
            name_dict = {
                'train' : 'train', # 'bounding_box_train',
                'test'  : 'test', # 'bounding_box_test',
                'query' : 'query'
            }

            if split in ['train', 'test', 'query']:
                try:
                    dir = os.path.join(self.DATA_ROOT, dataset, name_dict[split])
                except KeyError:
                    raise ValueError("invalid split for market dataset")

        for f in os.listdir(dir):
            if f[-4:] == '.jpg':
                idt = int(f[0:f.index('_')])
                if idt != -1: # For testing, ignore irrelavent images
                    if not any(idt == l for l in files_dict.keys()):
                        files_dict[idt] = []

                    path = os.path.join(dir, f)

                    camera = f[f.index('_') + 2 : f.index('_') + 3]
                    # elif dataset == 'cuhk03':
                    #     camera = f[len(f) - f[::-1].index('_'):-4]

                    files_arr.append([path, idt, int(camera)])
                    files_dict[idt].append(path)

        return files_dict, files_arr

    def _verify(self, dataset):
        gallery_dict, gallery_arr = self._get_data(dataset, 'test')
        query_dict, query_arr = self._get_data(dataset, 'query')

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
            print("Warning: {} query samples missing ground-truth samples.".format(missing))
        else:
            print("Verification successful! All query have ground-truth samples.")

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

                if self.crop:
                    crop_x = np.random.randint(0.125 * self.img_dim[1])
                    crop_y = np.random.randint(0.125 * self.img_dim[0])

                    shape = (int(1.125 * self.img_dim[0]), int(1.125 * self.img_dim[1]))
                    im = _imread_scale(path, shape, self.preprocess)
                    im = im[crop_y:crop_y+self.img_dim[0], crop_x:crop_x+self.img_dim[1]]
                else:
                    im = _imread_scale(path, self.img_dim, self.preprocess)

                if self.flip:
                    if np.random.random() < 0.5:
                        # print('flipped')
                        im = np.flip(im, axis=1)

                # print(im.shape)
                input_imgs.append(im)
                labels.append(list(self.files_dict.keys()).index(idt_choice[p]))

        return np.stack(input_imgs), np.stack(labels)[:, np.newaxis]

    def sample(self, width=None, height=None, plot=True, figsize=(10,10)):
        sample_batch, sample_labels = next(self)

        if sample_batch.dtype == np.float64:
            sample_batch = np.clip(sample_batch, 0, 1)

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
        super().__init__(dataset, split)

        self.preprocess = preprocess
        self.img_dim = img_dim
        self.crop = crop
        self.flip = (True if flip in ["mean", "concat"] else False)
        self.buffer = buffer

        self.current = 0
        self.high = len(self.files_arr)

        # Test augmentation
        self.n = 2 if self.flip else 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.high:
            raise StopIteration
        else:
            img_buffer = [[] for i in range(self.n)]
            next = self.current + self.buffer

            for i in range(self.current, min(next, self.high)):
                path = self.files_arr[i][0]

                if self.crop:
                    crop_x = self.img_dim[1] // 16
                    crop_y = self.img_dim[0] // 16

                    shape = (int(1.125 * self.img_dim[0]), int(1.125 * self.img_dim[1]))
                    im = _imread_scale(path, shape, self.preprocess)
                    im = im[crop_y:crop_y+self.img_dim[0], crop_x:crop_x+self.img_dim[1]]
                else:
                    im = _imread_scale(path, self.img_dim, self.preprocess)

                if self.flip:
                    im_flip = np.flip(im, axis=1)
                    img_buffer[1].append(im_flip)

                img_buffer[0].append(im)

            self.current = next
            return np.stack(img_buffer)

# https://github.com/Cysu/open-reid/blob/master/reid/datasets/cuhk03.py
# https://github.com/Cysu/open-reid/blob/master/reid/evaluators.py
class CUHK03():
    def __init__(self, labeled, split, idx):
        self.DATA_ROOT = '/home/gong/research/data/cuhk03'

        full_dict, full_arr = self._get_full_data(labeled)
        self.full_dict = full_dict
        self.full_arr = full_arr

        files_dict, files_arr = self._get_split(split, idx)
        self.files_dict = files_dict
        self.files_arr = files_arr

    def _get_full_data(self, labeled):
        with open(os.path.join(self.DATA_ROOT, "meta.json")) as f:
            meta = json.load(f)

        files_arr = []
        files_dict = {}

        for idt, names in enumerate(meta["identities"]):
            if labeled:
                cam0 = names[0][:5]
                cam1 = names[1][:5]
            else:
                cam0 = names[0][5:]
                cam1 = names[1][5:]

            files_dict[idt] = cam0 + cam1

            for f in cam0:
                path = os.path.join(self.DATA_ROOT, "images", f)
                files_arr.append([path, idt, 0])
            for f in cam0:
                path = os.path.join(self.DATA_ROOT, "images", f)
                files_arr.append([path, idt, 1])

        return files_dict , files_arr

    def _get_split(self, split, split_idx):
        assert split_idx >= 0 and split_idx < 20

        with open(os.path.join(self.DATA_ROOT, "split.json")) as f:
            split_info_arr = json.load(f)

        files_dict = {}
        files_arr = []

        split_info = split_info_arr[split_idx][split]
        for idt in split_info:
            files_dict[idt] = self.full_dict[idt]
        for i, x in self.full_arr:
            idt_x = x[1]
            if idt_x in split_info:
                files_arr.append(x)

        return files_dict , files_arr

# https://github.com/liangzheng06/MARS-evaluation/blob/master/test_mars.m
# https://github.com/jiyanggao/Video-Person-ReID/blob/master/data_manager.py
# => MARS loaded
# Dataset statistics:
#   ------------------------------
#   subset   | # ids | # tracklets
#   ------------------------------
#   train    |   625 |     8298
#   query    |   626 |     1980
#   gallery  |   622 |     9330
#   ------------------------------
#   total    |  1251 |    19608
#   number of images per tracklet: 2 ~ 920, average 59.5
#   ------------------------------
# Issue: Different number of query and gallery identities?
# Known problem:
# https://github.com/KaiyangZhou/deep-person-reid/issues/77
# https://github.com/KaiyangZhou/deep-person-reid/issues/59

class MARS():
    def __init__(self, split):
        self.DATA_ROOT = '/home/gong/research/data/mars'
        self.split = split

        self.files_arr = self._get_files()
        self.files_dict = self._get_dict()

    def _get_names(self, fpath):
        # fpath = os.path.join(self.DATA_ROOT, self.split)
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return np.array(names)

    def _get_files(self):
        assert self.split in ["train" , "test", "query"]

        info_dir = "/home/gong/research/ref/MARS-evaluation/info"
        split = "train" if self.split == "train" else "test"
        names = self._get_names(os.path.join(info_dir, "{}_name.txt".format(split)))

        if self.split != "train":
            from scipy.io import loadmat
            query_idx = loadmat(os.path.join(info_dir, "query_IDX.mat"))
            query_idx = query_idx["query_IDX"][0]
            query_idx -= 1 # MATLAB indexing starts at 1
            print(query_idx.shape)

            track_test = loadmat(os.path.join(info_dir, "tracks_test_info.mat"))
            track_test = track_test["track_test_info"]

            pid_list = []
            if self.split == "query":
                tmp_names = []
                for start, end, pid, cam in track_test[query_idx]:
                    tmp_names += names[start - 1 : end].tolist()
                    if not pid in pid_list:
                        pid_list.append(pid)
                names = tmp_names
            else:
                # test
                gallery_idx = np.array([i for i in range(track_test.shape[0]) if i not in query_idx])
                print(gallery_idx.shape)
                tmp_names = []
                for start, end, pid, cam in track_test[gallery_idx]:
                    tmp_names += names[start - 1 : end].tolist()
                    if not pid in pid_list:
                        pid_list.append(pid)
                names = tmp_names
            print(len(pid_list))

        files_arr = []
        for name in names:
            path = os.path.join(self.DATA_ROOT, "bbox_{}".format(split), name)
            # pid = int(name[:name.index('C')])
            pid = self._extract_pid(name[:name.index('C')])
            cam = int(name[name.index('C') + 1 : name.index('T')])
            track = int(name[name.index('T') + 1 : name.index('F')])
            frame = int(name[name.index('F') + 1 : name.index('.')])
            files_arr.append([path, pid, cam, track, frame])

        return files_arr

    # def _get_paths(self):
    #     files_arr = []
    #     split_dir = "bbox_train" if self.split == "train" else "bbox_test"
    #     for frame, track, cam, pid in self.track_info:
    #         name = "{:04d}C{}T{:04d}F{:04d}.jpg".format(pid, cam, track, frame)
    #         path = os.path.join(self.DATA_ROOT, split_dir, name)
    #         files_arr.append(path)
    #     return files_arr

    def _get_dict(self):
        files_dict = {}
        for i, (path, pid, cam, track, frame) in enumerate(self.files_arr):
            if not pid in files_dict.keys():
                files_dict[pid] = {}

            if self.split == "train":
                if not track in files_dict[pid].keys():
                    files_dict[pid][track] = []
                files_dict[pid][track].append(path)
            else:
                if not cam in files_dict[pid].keys():
                    files_dict[pid][cam] = {}

                if not track in files_dict[pid][cam].keys():
                    files_dict[pid][cam][track] = []

                files_dict[pid][cam][track].append(path)

        return files_dict

    def _extract_pid(self, s):
        if s == "00-1":
            return -1
        return int(s)

# https://github.com/Yu-Wu/DukeMTMC-VideoReID
class DukeVideo():
    def __init__(self, split, camera=False):
        self.DATA_ROOT = '/home/gong/research/data/duke-video'
        self.camera = camera

        files_dict, files_arr = self._get_data(split, camera)
        self.files_dict = files_dict
        self.files_arr = files_arr

    def _get_data(self, split, camera):
        files_dict = {}
        files_arr = []

        name_dict = {
            'train' : 'train', # 'bounding_box_train',
            'test'  : 'gallery', # 'bounding_box_test',
            'query' : 'query'
        }

        if split in ['train', 'test', 'query']:
            try:
                dir = os.path.join(self.DATA_ROOT, name_dict[split])
            except KeyError:
                raise ValueError("invalid split for DukeMTMC dataset")

        for idt_str in os.listdir(dir):
            # if f[-4:] == '.jpg':
            idt = int(idt_str)
            if idt != -1: # For testing, ignore irrelavent images
                if not idt in files_dict.keys():
                    files_dict[idt] = {}

                for track_str in os.listdir(os.path.join(dir, idt_str)):
                    track = int(track_str)
                    if not track in files_dict[idt].keys():
                        files_dict[idt][track] = []

                    for frame in os.listdir(os.path.join(dir, idt_str, track_str)):
                        assert idt == int(frame[:frame.index('_')])

                        C_idx = frame.index('C')
                        cam = int(frame[C_idx + 1 : C_idx + 2])

                        path = os.path.join(dir, idt_str, track_str, frame)
                        files_arr.append([path, idt, track, cam])
                        files_dict[idt][track].append(path)

        return files_dict, files_arr

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

def _imread_scale(img_path, shape, preprocess=True):
    im = _imread(img_path)
    if im.shape[:2] != shape[:2]:
        # print('resize')
        im = cv2.resize(im, (shape[1], shape[0]))

    if preprocess:
        return _densenet_preprocess(im.astype('float64'))
    # elif preprocess == 'norm':
    #     return _preprocess_norm(im)

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
