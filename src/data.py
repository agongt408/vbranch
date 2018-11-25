import os
import json
import numpy as np
import cv2

import sys
sys.path.insert(0, '../DenseNet/')
import densenet

from config import *

input_output_dim = 128
input_shape = (256,128)
input_preprocess = True
input_r = 0.3

BODY_PARTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar"
]

"""# DATA_ROOT = '/home/albert/github/tensorflow/data/'
DATA_ROOT = '/home/albert/research/vbranch/data/'
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = '/home/ubuntu/albert/data/'"""

def get_data(split, keypoints=None, dataset='market', cuhk03='detected'):
    files_dict = {}
    files_arr = []

    if dataset == 'market' or dataset == 'duke':
        if split in ['train', 'test', 'query']:
            try:
                img_dir = os.listdir(
                    os.path.join(DATA_ROOT, dataset, split))
            except OSError:
                img_dir = os.listdir(
                os.path.join(DATA_ROOT, dataset, split))
        else:
            raise ValueError, 'split must be either query, train, or test'

    elif dataset == 'cuhk03':
        if split in ['train', 'test']:
            try:
                img_dir = os.listdir(
                    os.path.join(DATA_ROOT, dataset, cuhk03, split))
            except OSError:
                img_dir = os.listdir(
                    os.path.join(DATA_ROOT, dataset, cuhk03, split))
        else:
            raise ValueError, 'split must be either train or test'

    else:
        raise ValueError, 'dataset must be either market or cuhk03'

    for f in img_dir:
        if f[-4:] == '.jpg':
            idt = int(f[0:f.index('_')])
            # if idt != 0 and idt != -1:
            if not any(idt == l for l in files_dict.keys()):
                files_dict[idt] = []

            if dataset == 'market' or dataset == 'duke':
                path = os.path.join(DATA_ROOT, dataset, split, f)
            else:
                path = os.path.join(DATA_ROOT, dataset, cuhk03, split, f)

            if keypoints is None:
                files_arr.append([path, idt])
                files_dict[idt].append(path)
            else:
                print 'a', keypoints
                if _exist_all_keypoints(path, keypoints):
                    files_arr.append([path, idt])
                    files_dict[idt].append(path)

    for idt in files_dict.keys():
        if len(files_dict[idt]) == 0:
            files_dict.pop(idt)

    return files_dict, files_arr


def get_data_orientation(split, orientation='front'):
    files_dict, files_arr = get_data(
        split, ['RShoulder', 'LShoulder', 'RHip', 'LHip'])

    new_files_dict = {}
    new_files_arr = []

    for path, idt in files_arr:
        if not any(idt == l for l in new_files_dict.keys()):
            new_files_dict[idt] = []

        theta = _orientation_angle(path)
        if orientation is 'front':
            if theta < np.pi / 3:
                new_files_arr.append([path, idt])
                new_files_dict[idt].append(path)
        elif orientation is 'side':
            if theta >= np.pi / 3 and theta < 2 * np.pi / 3:
                new_files_arr.append([path, idt])
                new_files_dict[idt].append(path)
        elif orientation is 'back':
            if theta >= 2 * np.pi / 3:
                new_files_arr.append([path, idt])
                new_files_dict[idt].append(path)
        else:
            raise ValueError, 'orientation must be either front, back, or side'
            return None

    for idt in new_files_dict.keys():
        if len(new_files_dict[idt]) == 0:
            new_files_dict.pop(idt)

    return new_files_dict, new_files_arr


def _exist_all_keypoints(img_path, keypoints='all'):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    try:
        x = img_path.index('train')
        keypoint_path = os.path.join(
            DATA_ROOT, 'market/train_openpose/train_keypoints',
            '%s_keypoints.json' % root)
    except:
        try:
            x = img_path.index('test')
            keypoint_path = os.path.join(
                DATA_ROOT, 'market/test_openpose/test_keypoints',
                '%s_keypoints.json' % root)
        except:
            keypoint_path = os.path.join(
                DATA_ROOT, 'market-1501/query_openpose/query_keypoints',
                '%s_keypoints.json' % root)

    with open(keypoint_path) as data_file:
        data = json.load(data_file)

    if keypoints == 'all':
        keypoints = BODY_PARTS
    elif type(keypoints) == list:
        pass
    else:
        raise ValueError, 'invalid keypoints argument, ' + \
                            'must be "all" or list of strings'

    keypoints_map = np.zeros((3 * len(BODY_PARTS)), dtype=np.int64)
    for k in keypoints:
        n = BODY_PARTS.index(k)
        keypoints_map[3 * n] = keypoints_map[3 * n + 1] = \
                                keypoints_map[3 * n + 2] = 1

    for person in range(len(data['people'])):
        where_keypoints = np.multiply(keypoints_map,
            np.array(data['people'][person]['pose_keypoints']))
        if np.where(where_keypoints > 0.0)[0].shape[0] == 3 * len(keypoints):
            return True
        else:
            return False

def _dist(x0,y0,x1,y1):
    return np.hypot(x1 - x0, y1 - y0)

def _orientation_angle(img_path):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    try:
        x = img_path.index('train')
        keypoint_path = os.path.join(
            DATA_ROOT, 'market/train_openpose/train_keypoints',
            '%s_keypoints.json' % root)
    except:
        try:
            x = img_path.index('test')
            keypoint_path = os.path.join(
                DATA_ROOT, 'market/test_openpose/test_keypoints' ,
                '%s_keypoints.json' % root)
        except:
            keypoint_path = os.path.join(
                DATA_ROOT, 'market/query_openpose/query_keypoints/' ,
                '%s_keypoints.json' % root)

    with open(keypoint_path) as data_file:
        data = json.load(data_file)

    if len(data['people']) > 0:
        idx_rs = BODY_PARTS.index('RShoulder')
        RShoulder = data['people'][0]['pose_keypoints'][3*idx_rs:3*idx_rs+2]

        idx_ls = BODY_PARTS.index('LShoulder')
        LShoulder = data['people'][0]['pose_keypoints'][3*idx_ls:3*idx_ls+2]

        idx_rh = BODY_PARTS.index('RHip')
        RHip = data['people'][0]['pose_keypoints'][3*idx_rh:3*idx_rh+2]

        idx_lh = BODY_PARTS.index('LHip')
        LHip = data['people'][0]['pose_keypoints'][3*idx_lh:3*idx_lh+2]

        mean_torso_height = 0.5 * (_dist(RShoulder[0], RShoulder[1], RHip[0], RHip[1]) \
                            + _dist(LShoulder[0], LShoulder[1], LHip[0], LHip[1]))

        sign = 1 if RShoulder[0] < LShoulder[0] else -1

        r = sign * _dist(RShoulder[0], RShoulder[1], LShoulder[0], LShoulder[1]) \
                    / (mean_torso_height + 1e-6)

        r_bound = np.minimum(np.maximum(-1.0, r), 1.0)

        return np.arccos(r_bound)
    else:
        raise ValueError, 'image must contain at least one person'
        return None


def _imread(img_path):
    """
    returns RGB image
    misc.imread is deprecated
    """
    im = cv2.imread(img_path)
    if len(im.shape) == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        return im


def _imread_scale(img_path, shape=input_shape, preprocess=input_preprocess):
    if preprocess:
        return densenet.preprocess_input(cv2.resize(_imread(img_path),
                (shape[1], shape[0])).astype(np.float64))
    else:
        return cv2.resize(_imread(img_path),
                    (shape[1], shape[0])).astype(np.float64)


def _make_gaussian(shape, r=input_r, center=None, inverse=False):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    shape = shape of numpy array (y-axis, x-axis)
    """

    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)[:,np.newaxis]

    if center is None:
        x0 = y0 = np.array(shape).min() // 2
    else:
        x0 = center[0]
        y0 = center[1]

    radius = np.array(shape).min() * r
    result = np.exp(-4*np.log(2) * (np.power(x-x0,2) + np.power(y-y0,2))
                / radius**2)[0:shape[0], 0:shape[1]]

    if inverse:
        result = 1 - result

    return result


def _make_rect(shape, r=input_r, center=None, inverse=False, rand=True):
    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)[:,np.newaxis]

    if center is None:
        x0 = shape[1] / 2
        y0 = shape[0] / 2
    else:
        x0 = int(center[0])
        y0 = int(center[1])

    radius = int(np.array(shape).min() * r)

    if rand:
        top_left = (np.maximum(0, x0-radius), np.maximum(0, y0-radius))
        bottom_right = (np.minimum(shape[1], x0+radius + 1),
                            np.minimum(shape[0], y0+radius + 1))

        if inverse:
            result = np.ones(shape, dtype=np.float64)
            result[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                np.random.rand(bottom_right[1] - top_left[1],
                                bottom_right[0] - top_left[0])

        else:
            result = np.random.rand(shape[0], shape[1])
            result[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                np.ones((bottom_right[1] - top_left[1],
                            bottom_right[0] - top_left[0]))

        return result
    else:
        if inverse:
            result = cv2.rectangle(np.ones(shape, dtype=np.float64),
                                    (x0-radius,y0-radius),
                                    (x0+radius,y0+radius), 0.0, -1)
        else:
            result = cv2.rectangle(np.zeros(shape, dtype=np.float64),
                                    (x0-radius,y0-radius),
                                    (x0+radius,y0+radius), 1.0, -1)
        return result


def _create_keypoints(img_path, shape=input_shape,
                    preprocess=input_preprocess,
                    r=input_r, keypoints=None,
                    rect=False, inverse=False, rand=False):
    if img_path.find('/') > -1:
        root = img_path[len(img_path) - img_path[::-1].index('/'):-4:]
    else:
        root = img_path[0:img_path.index('.')]

    htmp = np.zeros(shape).astype(np.float64)

    try:
        x = img_path.index('train')
        keypoint_path = os.path.join(
            DATA_ROOT, 'market/train_openpose/train_keypoints',
            '%s_keypoints.json' % root)
    except:
        try:
            x = img_path.index('test')
            keypoint_path = os.path.join(
                DATA_ROOT, 'market/test_openpose/test_keypoints' ,
                '%s_keypoints.json' % root)
        except:
            keypoint_path = os.path.join(
                DATA_ROOT, 'market/query_openpose/query_keypoints' ,
                '%s_keypoints.json' % root)

    if keypoints == 'all':
        keypoints = BODY_PARTS
    elif type(keypoints) == list:
        pass
    elif keypoints == None:
        pass
    else:
        raise ValueError, 'invalid keypoints argument, ' + \
                            'must be "all", list or None'

    with open(keypoint_path) as data_file:
        data = json.load(data_file)

    if keypoints is not None:
        for person in range(len(data['people'])):
            for k in keypoints:
                idx = BODY_PARTS.index(k)
                x_key = data['people'][person]['pose_keypoints'][idx*3] * shape[1] / 64.0
                y_key = data['people'][person]['pose_keypoints'][idx*3+1] * shape[0] / 128.0
                # c_key = data['people'][person]['pose_keypoints'][BODY_PARTS.index(k) * 3 + 2]

                if not (x_key == 0 and y_key == 0):
                    if rect:
                        htmp = np.maximum(
                                    htmp,_make_rect(shape, r,
                                        (x_key, y_key),inverse, rand))
                    else:
                        htmp = np.maximum(
                                    htmp,_make_gaussian(shape, r,
                                        (x_key, y_key), inverse))

    # Make sure type is float
    return htmp


def batch_generator(files_dict, P=4, K=4, shape=input_shape, preprocess=input_preprocess,
                    r=input_r, keypoints=None, cam_output_dim=(16,8),
                    crop=False, flip=False, rect=False, cam_rect=False, cam_input=True,
                    inverse=False, rand=False):
    while True:
        input_batch = [[]]
        target_batch = [np.zeros((1,P*K))]

        if keypoints is not None:
            for k in range(len(keypoints)):
                if cam_input:
                    input_batch.append([])
                else:
                    target_batch.append([])

        idt_choice = np.random.choice(files_dict.keys(), P, replace=False)
        for p in range(len(idt_choice)):
            if K > len(files_dict[idt_choice[p]]):
                k_choice = np.random.choice(
                    range(len(files_dict[idt_choice[p]])),
                    K, replace=True)

            else:
                k_choice = np.random.choice(
                    range(len(files_dict[idt_choice[p]])),
                    K, replace=False)

            for k in k_choice:
                path = files_dict[idt_choice[p]][k]

                crop_x = np.random.randint(0.125 * shape[1])
                crop_y = np.random.randint(0.125 * shape[0])

                if keypoints is not None:
                    for k in range(len(keypoints)):
                        if cam_rect:
                            htmp = _create_keypoints(
                                path, cam_output_dim, preprocess,
                                r, keypoints[k],
                                cam_rect, inverse, rand).\
                                reshape((1,) + cam_output_dim + (1,))
                        else:
                            htmp = _create_keypoints(
                                path, cam_output_dim, preprocess,
                                r, keypoints[k],
                                cam_rect, inverse).\
                                reshape((1,) + cam_output_dim + (1,))

                        if cam_input:
                            input_batch[k + 1].append(htmp)
                        else:
                            target_batch[k + 1].append(htmp)

                im = None
                if crop:
                    im = _imread_scale(
                        path, (int(1.125 * shape[0]), int(1.125 * shape[1])),
                        preprocess)[crop_y:crop_y+shape[0], crop_x:crop_x+shape[1]]
                else:
                    im = _imread_scale(path, shape, preprocess)

                    if rect:
                        im *= np.tile(
                            _create_keypoints(path, shape, preprocess, 0.3, ['Neck']).\
                            reshape((shape[0], shape[1], 1)), [1,1,3])

                if flip:
                    if np.random.randint(2) == 1:
                        im = np.flip(im, axis=1)

                input_batch[0].append(im.reshape((1,) + shape + (3,)))

        for i in range(len(input_batch)):
            input_batch[i] = np.concatenate(input_batch[i], axis=0)

        for i in range(len(target_batch)):
            target_batch[i] = np.concatenate(target_batch[i], axis=0)

        yield (input_batch, target_batch)

def multi_batch_generator(files_dict=[], P=4, K=4, shape=input_shape,
                            preprocess=input_preprocess,
                            crop=False, flip=False):

    generator_arr = [batch_generator(f_d, P, K, shape, preprocess,
                        crop=crop, flip=flip)
                        for f_d in files_dict]

    while True:
        yield ([gen.next()[0][0] for gen in generator_arr], np.zeros((P*K,)))
