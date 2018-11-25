import numpy as np
import time
import matplotlib.pyplot as plt

import os
import sys

import data

input_shape = (256,128)
input_preprocess = True

"""# DATA_ROOT = '/home/albert/github/tensorflow/data/'
DATA_ROOT = '/home/albert/research/vbranch/data/'
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = '/home/ubuntu/albert/data/'"""

DATA_ROOT = '/home/gong/research/data'

# http://www.liangzheng.org/Project/project_reid.html
# http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
def _get_data(split, keypoints=None, dataset='market', cuhk03='detected'):
    files_dict = {}
    files_arr = []

    if dataset == 'market' or dataset == 'duke':
        if split in ['train', 'test', 'query']:
            img_dir = os.listdir(os.path.join(DATA_ROOT, dataset, split))
        else:
            raise ValueError('split must be either query, train, or test')

    elif dataset == 'cuhk03':
        if split in ['train', 'test']:
            img_dir = os.listdir(
                os.path.join(DATA_ROOT, dataset, cuhk03, split))
        else:
            raise ValueError('split must be either train or test')

    else:
        raise ValueError('dataset must be either market or cuhk03')

    for f in img_dir:
        if f[-4:] == '.jpg':
            idt = int(f[0:f.index('_')])
            if idt != -1:
                if not any(idt == l for l in files_dict.keys()):
                    files_dict[idt] = {}

                if dataset == 'market' or dataset == 'duke':
                    path = os.path.join(DATA_ROOT, dataset, split, f)
                else:
                    path = os.path.join(DATA_ROOT, dataset, cuhk03, split, f)

                if dataset == 'market' or dataset == 'duke':
                    camera = f[f.index('_') + 2 : f.index('_') + 3]
                elif dataset == 'cuhk03':
                    camera = f[len(f) - f[::-1].index('_'):-4]
                else:
                    raise ValueError('dataset must be either market or cuhk03')

                if keypoints is None:
                    files_arr.append([path, idt, int(camera)])
                    try:
                        files_dict[idt][int(camera)].append(path)
                    except:
                        files_dict[idt][int(camera)] = []
                        files_dict[idt][int(camera)].append(path)
                else:
                    if data._exist_all_keypoints(path, keypoints, DATA_ROOT):
                        files_arr.append([path, idt, int(camera)])
                        try:
                            files_dict[idt][int(camera)].append(path)
                        except:
                            files_dict[idt][int(camera)] = []
                            files_dict[idt][int(camera)].append(path)

    return files_dict, files_arr


def _get_data_orientation(split, orientation='front'):
    files_dict, files_arr = _get_data(
        split, keypoints=['RShoulder', 'LShoulder', 'RHip', 'LHip'])

    new_files_dict = {}
    new_files_arr = []

    for path, idt, camera in files_arr:
        if not any(idt == l for l in new_files_dict.keys()):
            new_files_dict[idt] = {}

        theta = data._orientation_angle(path)
        if orientation is 'front':
            if theta < np.pi / 3:
                new_files_arr.append([path, idt, camera])
                try:
                    new_files_dict[idt][int(camera)].append(path)
                except:
                    new_files_dict[idt][int(camera)] = []
                    new_files_dict[idt][int(camera)].append(path)
        elif orientation is 'side':
            if theta >= np.pi / 3 and theta < 2 * np.pi / 3:
                new_files_arr.append([path, idt, camera])
                try:
                    new_files_dict[idt][int(camera)].append(path)
                except:
                    new_files_dict[idt][int(camera)] = []
                    new_files_dict[idt][int(camera)].append(path)
        elif orientation is 'back':
            if theta >= 2 * np.pi / 3:
                new_files_arr.append([path, idt, camera])
                try:
                    new_files_dict[idt][int(camera)].append(path)
                except:
                    new_files_dict[idt][int(camera)] = []
                    new_files_dict[idt][int(camera)].append(path)
        else:
            raise ValueError('orientation must be either front, back, or side')
            return None

    return new_files_dict, new_files_arr


def _get_embeddings(model, test_files=None, query_files=None,
                    shape=input_shape, preprocess=input_preprocess,
                    crop=False, flip=False, inputs=None):
    if test_files is None or query_files is None:
        _, test_files = _get_data('test')
        _, query_files = _get_data('query')

    all_embs = []
    query_embs = []

    for files_arr, emb_arr in [(test_files, all_embs), (query_files, query_embs)]:
        s, z = time.time(), 0

        for f, _, _ in files_arr:
            # np.random.randint(0.125 * shape[1])
            crop_x = int(0.125 * shape[1] / 2)
            # np.random.randint(0.125 * shape[0])
            crop_y = int(0.125 * shape[0] / 2)

            if crop:
                img = data._imread_scale(f, (int(1.125 * shape[0]),
                                            int(1.125 * shape[1])),
                                            preprocess)[crop_y:crop_y+shape[0],
                                                        crop_x:crop_x+shape[1]]
            else:
                img = data._imread_scale(f, shape, preprocess)

            # img = data._imread_scale(f, shape, preprocess)

            predict = []

            aug = 2 if flip else 1

            for i in range(aug):
                input_batch = []
                for ip in range(inputs if inputs else len(model.inputs)):
                    input_batch.append(np.flip(img, axis=1).\
                                        reshape(1,shape[0],shape[1],3) if i else \
                                        reshape(1,shape[0],shape[1],3) if i else \
                                        img.reshape(1,shape[0],shape[1],3))

                if len(model.outputs) == 1:
                    predict.append(model.predict(input_batch)[0])
                else:
                    predict.append(model.predict(input_batch)[0][0])

            emb_arr.append(np.array(predict).reshape((1,aug,-1)).mean(axis=1))

            z += 1
            if z % 1000 == 0:
                print(z, time.time() - s)

    return np.concatenate(all_embs, axis=0), np.concatenate(query_embs, axis=0)

# https://cysu.github.io/open-reid/notes/evaluation_metrics.html
def _evaluate_metrics(all_embs, query_embs,
                    test_dict=None, test_files=None, query_files=None,
                    rank=[1,5], mAP=True, dataset='market'):

    if test_dict is None or test_files is None or query_files is None:
        test_dict, test_files = _get_data('test')
        _, query_files = _get_data('query')

    all_identities = np.array([p[1] for p in test_files])
    all_camera = np.array([p[2] for p in test_files])

    if rank is not None:
        correct = np.array([0] * len(rank))
        test_iter = np.array([0] * len(rank))

    if mAP:
        AP = []

    for q in range(len(query_files)):
        idt, camera = int(query_files[q][1]), int(query_files[q][2])

        b = np.logical_or(all_camera != camera, all_identities != idt)

        i = 0
        for _, idt_t, cam_t in np.array(test_files)[b]:
            if idt == int(idt_t) and camera != int(cam_t):
                i += 1
        if i == 0:
            print('missing')
            continue

        if len(test_dict[idt].keys()) > 1:
            query_emb = query_embs[q]
            distance_vectors = np.power(np.squeeze(np.abs(all_embs[b] - query_emb)), 2)
            distance = np.sqrt(np.sum(distance_vectors, axis=1))

            top_inds = distance.argsort()
            output_classes = all_identities[b][top_inds]

            # Calculate rank
            for r in range(len(rank)):
                r_top_inds = top_inds[:rank[r]]
                r_output_classes = all_identities[b][r_top_inds]

                if np.where(r_output_classes == idt)[0].shape[0] > 0:
                    correct[r] += 1
                test_iter[r] += 1

            # Calculate mAP
            if mAP:
                precision = []
                correct_old = 0

                for t in range(distance.shape[0]):
                    if idt == output_classes[t]:
                        precision.append(float(correct_old + 1) / (t + 1))
                        correct_old += 1

                AP.append(np.mean(np.array(precision)))

    metrics = {}
    if rank is not None:
        metrics['rank'] = correct.astype(np.float32) / test_iter
    if mAP:
        metrics['mAP'] = np.array(AP).mean()

    return metrics


def _get_embeddings_cuhk03(model, test_files=None,
                    shape=input_shape, preprocess=input_preprocess, cover=False,
                    cuhk03='detected'):
    if test_files is None:
        _, test_files = _get_data('test', dataset='cuhk03', cuhk03=cuhk03)

    all_embs = {}

    s, z = time.time(), 0

    for f, idt, cam in test_files:
        if not any(idt == l for l in all_embs.keys()):
            all_embs[idt] = {}

        img = data._imread_scale(f, shape, preprocess)

        if cover:
            img *= np.tile(data.create_keypoints(f, shape, preprocess,
                                                0.1, ['Neck'], rect=True).\
                            reshape((shape[0], shape[1], 1)), [1,1,3])

        input_batch = []
        for i in range(len(model.inputs)):
            input_batch.append(img.reshape(1,shape[0],shape[1],3))
        # input_batch = [img.reshape(1,shape[0],shape[1],3)]
        '''
        for i in range(len(model.inputs) - 1):
            input_batch.append(np.ones((1,) + model.input_shape[1][-3:]))
        '''
        if len(model.outputs) == 1:
            predict = model.predict(input_batch)[0]
        else:
            predict = model.predict(input_batch)[0][0]

        try:
            all_embs[idt][int(cam)].append(predict.reshape((1,-1)))
        except:
            all_embs[idt][int(cam)] = []
            all_embs[idt][int(cam)].append(predict.reshape((1,-1)))

        z += 1
        if z % 500 == 0:
            print(z, time.time() - s)

    return all_embs


def _evaluate_metrics_cuhk03(all_embs, rank=[1,5], mAP=True, n=1):
    metrics = {}
    if mAP:
        metrics['mAP'] = 0
    if rank is not None:
        metrics['rank'] = np.zeros((len(rank)))

    for i in range(n):
        if rank is not None:
            correct = np.array([0] * len(rank))
            test_iter = np.array([0] * len(rank))

        if mAP:
            AP = []

        for idt_q in all_embs.keys():
            for cam_q in all_embs[idt_q].keys():
                for query_emb in all_embs[idt_q][cam_q]:
                    gallery_embs = []
                    gallery_idts = []

                    for idt_g in all_embs.keys():
                        g_choice = np.random.choice(
                            range(len(all_embs[idt_g][1 - int(cam_q)])), 1)[0]
                        gallery_embs.append(all_embs[idt_g][1 - int(cam_q)][g_choice])
                        gallery_idts.append(idt_g)

                    gallery_embs = np.array(gallery_embs)
                    gallery_idts = np.array(gallery_idts)

                    distance_vectors = np.squeeze(np.abs(gallery_embs - query_emb))
                    distance = np.sum(distance_vectors, axis=1)

                    top_inds = distance.argsort()
                    output_classes = gallery_idts[top_inds]

                    # Calculate rank
                    for r in range(len(rank)):
                        r_top_inds = top_inds[:rank[r]]
                        r_output_classes = gallery_idts[r_top_inds]

                        if np.where(r_output_classes == idt_q)[0].shape[0] > 0:
                            correct[r] += 1
                        test_iter[r] += 1

                    # Calculate mAP
                    if mAP:
                        precision = []
                        correct_old = 0

                        for t in range(distance.shape[0]):
                            if idt_q == output_classes[t]:
                                precision.append(float(correct_old + 1) / (t + 1))
                                correct_old += 1

                        AP.append(np.mean(np.array(precision)))

        if rank is not None:
            metrics['rank'] += np.divide(correct.astype(np.float32), test_iter) / n
        if mAP:
            metrics['mAP'] += np.mean(np.array(AP)) / n

    return metrics


def get_score(model, hist=None, keypoints=None, dataset='market', cuhk03='detected',
                preprocess=input_preprocess, shape=input_shape, crop=False, flip=False,
                test_dict=None, test_files=None, query_files=None, rank=[1,5], mAP=True,
                inputs=None, n=1):
    score = {
        'rank' : {},
        'mAP' : 0,
        'loss' : []
    }

    if test_dict is None and test_files is None:
        test_dict, test_files = _get_data('test', keypoints, dataset, cuhk03)

    if query_files is None:
        _, query_files = _get_data('query', keypoints, dataset)

    if dataset == 'market' or dataset == 'duke':
        all_embs, query_embs = _get_embeddings(model, test_files, query_files,
                                            shape, input_preprocess, crop, flip, inputs)

        start = time.time()
        metrics = _evaluate_metrics(all_embs, query_embs, test_dict,
                                        test_files, query_files, rank, mAP, dataset)
        print('metric time: %f' % (time.time() - start))

    elif dataset == 'cuhk03':
        all_embs = _get_embeddings_cuhk03(model, test_files, shape,
                                            input_preprocess, flip)

        start = time.time()
        metrics = _evaluate_metrics_cuhk03(all_embs, rank, mAP, n)
        print('metric time: %f' % (time.time() - start))

    else:
        raise ValueError('dataset must be either market or cuhk03')

    if rank is not None:
        for r in range(len(rank)):
            score['rank']['r' + str(rank[r])] = np.around(
                metrics['rank'], decimals=4).tolist()[r]
    if mAP:
        score['mAP'] = np.around(metrics['mAP'], decimals=4)

    try:
        score['loss'] += hist.history['loss']
    except AttributeError:
        pass

    return score
