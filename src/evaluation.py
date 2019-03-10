import numpy as np
import time
import matplotlib.pyplot as plt

import os
import sys

from data import TestingDataGenerator

from keras.utils import Progbar

def get_score(model, dataset='market', cuhk03='detected', preprocess=True,
        img_dim=(128,64,3), crop=False, flip=False, rank=[1,5], compute_mAP=True):

    gallery_data = TestingDataGenerator(dataset, 'test', preprocess, img_dim, crop, flip)
    query_data = TestingDataGenerator(dataset, 'query', preprocess, img_dim, crop, flip)

    if dataset == 'market' or dataset == 'duke':
        if flip == "concat":
            gallery_embs, query_embs = _get_embeddings_concat(model, gallery_data, query_data)
        elif flip == "mean":
            gallery_embs, query_embs = _get_embeddings_mean(model, gallery_data, query_data)
        else:
            gallery_embs, query_embs = _get_embeddings(model, gallery_data, query_data)

        rank_score, mAP_score = _evaluate_metrics(gallery_embs, query_embs, gallery_data,
            query_data, rank, compute_mAP)

    # elif dataset == 'cuhk03':
    #     all_embs = _get_embeddings_cuhk03(model, test_files, shape,
    #                                         input_preprocess, flip)
    #
    #     start = time.time()
    #     metrics = _evaluate_metrics_cuhk03(all_embs, rank, mAP, n)
    #     print('metric time: %f' % (time.time() - start))
    #
    # else:
    #     raise ValueError('dataset must be either market or cuhk03')

    for i in range(len(rank)):
        print("rank-{}: {:.2f}".format(rank[i], 100 * rank_score[i]))
    print("mAP: {:.2f}".format(100 * mAP_score))

    return {'rank' : rank_score, 'mAP' : mAP_score}

# def _get_embeddings(model, gallery_iter, query_iter):
#     n_inputs = len(model.inputs) # Add multiple inputs for ensemble networks
#     if n_inputs > 1:
#         print("Warning: this model has {} inputs.".format(len(model.inputs)))
#
#     gallery_embs = []
#     query_embs = []
#
#     print('Computing gallery embeddings...')
#     gallery_progbar = Progbar(len(gallery_iter.files_arr))
#
#     for it in gallery_iter:
#         e = model.predict([it[0],] * n_inputs)
#         gallery_embs.append(e)
#         gallery_progbar.add(len(e))
#
#     print('Computing query embeddings...')
#     query_progbar = Progbar(len(query_iter.files_arr))
#
#     for it in query_iter:
#         e = model.predict([it[0],] * n_inputs)
#         query_embs.append(e)
#         query_progbar.add(len(e))
#
#     gallery_embs = np.concatenate(gallery_embs, axis=0)
#     query_embs = np.concatenate(query_embs, axis=0)
#     return gallery_embs , query_embs

def _get_embeddings(model, gallery_iter, query_iter):
    n_inputs = len(model.inputs) # Add multiple inputs for ensemble networks
    if n_inputs > 1:
        print("Warning: this model has {} inputs.".format(len(model.inputs)))

    gallery_embs = []
    query_embs = []

    for name, emb_list, data_iter in [('gallery', gallery_embs, gallery_iter),
                                      ('query', query_embs, query_iter)]:
        print("Computing {} embeddings...".format(name))
        progbar = Progbar(len(data_iter.files_arr))

        for it in data_iter:
            e = model.predict([it[0],] * n_inputs)
            emb_list.append(e)
            progbar.add(len(e))

    gallery_embs = np.concatenate(gallery_embs, axis=0)
    query_embs = np.concatenate(query_embs, axis=0)
    return gallery_embs , query_embs

def _get_embeddings_mean(model, gallery_iter, query_iter):
    # print("mean")
    n_inputs = len(model.inputs) # Add multiple inputs for ensemble networks
    if n_inputs > 1:
        print("Warning: this model has {} inputs.".format(len(model.inputs)))

    gallery_embs = []
    query_embs = []

    for name, emb_list, data_iter in [('gallery', gallery_embs, gallery_iter),
                                      ('query', query_embs, query_iter)]:
        print("Computing {} embeddings...".format(name))
        progbar = Progbar(len(data_iter.files_arr))

        for batch in data_iter:
            # Test augmentation
            e = []
            for it in batch:
                e.append(model.predict([it,] * n_inputs))
            if len(batch) == 1:
                e = e[0]
            else:
                e = np.mean(e, axis=0)

            emb_list.append(e)
            progbar.add(len(e))

    gallery_embs = np.concatenate(gallery_embs, axis=0)
    query_embs = np.concatenate(query_embs, axis=0)
    return gallery_embs , query_embs

def _get_embeddings_concat(model, gallery_iter, query_iter):
    # print("concat")
    n_inputs = len(model.inputs) // 2 # Add multiple inputs for ensemble networks
    if n_inputs > 1:
        print("Warning: this model has {} inputs.".format(len(model.inputs)))

    gallery_embs = []
    query_embs = []

    for name, emb_list, data_iter in [('gallery', gallery_embs, gallery_iter),
                                      ('query', query_embs, query_iter)]:
        print("Computing {} embeddings...".format(name))
        progbar = Progbar(len(data_iter.files_arr))

        for batch in data_iter:
            e = model.predict([batch[0],batch[1]] * n_inputs)
            emb_list.append(e)
            progbar.add(len(e))

    gallery_embs = np.concatenate(gallery_embs, axis=0)
    query_embs = np.concatenate(query_embs, axis=0)
    return gallery_embs , query_embs

# https://cysu.github.io/open-reid/notes/evaluation_metrics.html
def _evaluate_metrics(gallery_embs, query_embs, gallery_data, query_data, rank, compute_mAP):
    gallery_idts = np.array([p[1] for p in gallery_data.files_arr])
    gallery_cams = np.array([p[2] for p in gallery_data.files_arr])

    if rank is not None:
        correct = np.array([0] * len(rank))
        test_iter = np.array([0] * len(rank))

    AP = []

    print("Computing rank ({}) and mAP ({}) scores...".format(rank, compute_mAP))
    progbar = Progbar(len(query_data.files_arr))

    for q in range(len(query_data.files_arr)):
        idt, camera = int(query_data.files_arr[q][1]), int(query_data.files_arr[q][2])

        b = np.logical_or(gallery_cams != camera, gallery_idts != idt)
        # b = (gallery_idts != idt)
        # print(b)

        # Verify exists a valid instance in the gallery set
        # Commment out for performance imprvmts (not needed for market dataset)
        # i = 0
        # for _, idt_t, cam_t in np.array(gallery_data.files_arr)[b]:
        #     if idt == int(idt_t) and camera != int(cam_t):
        #         i += 1
        # if i == 0:
        #     print('missing')
        #     continue

        # if len(gallery_data.files_dict[idt].keys()) > 1:
        q_emb = query_embs[q]
        distance_vectors = np.power(np.squeeze(np.abs(gallery_embs[b] - q_emb)), 2)
        distance = np.sqrt(np.sum(distance_vectors, axis=1))

        top_inds = distance.argsort()
        output_classes = gallery_idts[b][top_inds]

        # Calculate rank
        for r in range(len(rank)):
            r_top_inds = top_inds[:rank[r]]
            r_output_classes = gallery_idts[b][r_top_inds]

            if np.where(r_output_classes == idt)[0].shape[0] > 0:
                correct[r] += 1
            test_iter[r] += 1

        if compute_mAP:
            precision = []
            correct_old = 0

            for t in range(distance.shape[0]):
                if idt == output_classes[t]:
                    precision.append(float(correct_old + 1) / (t + 1))
                    correct_old += 1

            AP.append(np.mean(np.array(precision)))

        if (q + 1) % 10 == 0 or (q + 1) == len(query_data.files_arr):
            progbar.update(q + 1)

    rank_score = correct.astype(np.float32) / test_iter
    mAP = np.mean(AP)
    return rank_score, mAP


# def _get_embeddings_cuhk03(model, test_files=None,
#                     shape=input_shape, preprocess=input_preprocess, cover=False,
#                     cuhk03='detected'):
#     if test_files is None:
#         _, test_files = _get_data('test', dataset='cuhk03', cuhk03=cuhk03)
#
#     all_embs = {}
#
#     s, z = time.time(), 0
#
#     for f, idt, cam in test_files:
#         if not any(idt == l for l in all_embs.keys()):
#             all_embs[idt] = {}
#
#         img = data._imread_scale(f, shape, preprocess)
#
#         if cover:
#             img *= np.tile(data.create_keypoints(f, shape, preprocess,
#                                                 0.1, ['Neck'], rect=True).\
#                             reshape((shape[0], shape[1], 1)), [1,1,3])
#
#         input_batch = []
#         for i in range(len(model.inputs)):
#             input_batch.append(img.reshape(1,shape[0],shape[1],3))
#         # input_batch = [img.reshape(1,shape[0],shape[1],3)]
#         '''
#         for i in range(len(model.inputs) - 1):
#             input_batch.append(np.ones((1,) + model.input_shape[1][-3:]))
#         '''
#         if len(model.outputs) == 1:
#             predict = model.predict(input_batch)[0]
#         else:
#             predict = model.predict(input_batch)[0][0]
#
#         try:
#             all_embs[idt][int(cam)].append(predict.reshape((1,-1)))
#         except:
#             all_embs[idt][int(cam)] = []
#             all_embs[idt][int(cam)].append(predict.reshape((1,-1)))
#
#         z += 1
#         if z % 500 == 0:
#             print(z, time.time() - s)
#
#     return all_embs
#
#
# def _evaluate_metrics_cuhk03(all_embs, rank=[1,5], mAP=True, n=1):
#     metrics = {}
#     if mAP:
#         metrics['mAP'] = 0
#     if rank is not None:
#         metrics['rank'] = np.zeros((len(rank)))
#
#     for i in range(n):
#         if rank is not None:
#             correct = np.array([0] * len(rank))
#             test_iter = np.array([0] * len(rank))
#
#         if mAP:
#             AP = []
#
#         for idt_q in all_embs.keys():
#             for cam_q in all_embs[idt_q].keys():
#                 for query_emb in all_embs[idt_q][cam_q]:
#                     gallery_embs = []
#                     gallery_idts = []
#
#                     for idt_g in all_embs.keys():
#                         g_choice = np.random.choice(
#                             range(len(all_embs[idt_g][1 - int(cam_q)])), 1)[0]
#                         gallery_embs.append(all_embs[idt_g][1 - int(cam_q)][g_choice])
#                         gallery_idts.append(idt_g)
#
#                     gallery_embs = np.array(gallery_embs)
#                     gallery_idts = np.array(gallery_idts)
#
#                     distance_vectors = np.squeeze(np.abs(gallery_embs - query_emb))
#                     distance = np.sum(distance_vectors, axis=1)
#
#                     top_inds = distance.argsort()
#                     output_classes = gallery_idts[top_inds]
#
#                     # Calculate rank
#                     for r in range(len(rank)):
#                         r_top_inds = top_inds[:rank[r]]
#                         r_output_classes = gallery_idts[r_top_inds]
#
#                         if np.where(r_output_classes == idt_q)[0].shape[0] > 0:
#                             correct[r] += 1
#                         test_iter[r] += 1
#
#                     # Calculate mAP
#                     if mAP:
#                         precision = []
#                         correct_old = 0
#
#                         for t in range(distance.shape[0]):
#                             if idt_q == output_classes[t]:
#                                 precision.append(float(correct_old + 1) / (t + 1))
#                                 correct_old += 1
#
#                         AP.append(np.mean(np.array(precision)))
#
#         if rank is not None:
#             metrics['rank'] += np.divide(correct.astype(np.float32), test_iter) / n
#         if mAP:
#             metrics['mAP'] += np.mean(np.array(AP)) / n
#
#     return metrics
