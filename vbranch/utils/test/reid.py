from ... datasets.reid import TestingDataGenerator as TestGen

import numpy as np
from tensorflow.keras.utils import Progbar
import multiprocessing
from time import time

def get_score(sess, dataset='market', rank=[1,5], n_branches=1, **kwargs):
    """
    Returns the rank scores (multiple) and mAP
    Args:
        - sess: tf session
        - dataset: 'market', 'duke'
        - preprocess: if true, preprocess using resnet imagenet
        - img_dim: image dimensions
        - crop: if true, concatenate embeddings from crops for each image (future)
        - flip: if true, take mean of horizontal flips for each image/crop (future)
        - rank: number of nearest neighbors to consider when calculating rank
        - n_branches: number of branches
    Returns:
        - dict of rank scores and mAP"""

    assert dataset in ['market', 'duke']

    # Get image data iterators
    gallery_data = TestGen(dataset, 'test', **kwargs)
    query_data = TestGen(dataset, 'query', **kwargs)
    gallery_embs,query_embs = _get_emb(sess, gallery_data, query_data, n_branches)

    if n_branches == 1:
        rank_score, mAP_score = _evaluate_metrics(gallery_embs, query_embs,
            gallery_data, query_data, rank)

        results = {'mAP': mAP_score}
        for i, r in enumerate(rank):
            results['rank'+str(r)] = rank_score[i]
    else:
        results = {}
        for branch in range(n_branches):
            rank_score, mAP_score = _evaluate_metrics(gallery_embs[branch],
                query_embs[branch], gallery_data, query_data, rank)

            results['mAP_{}'.format(branch+1)] = mAP_score
            for i, r in enumerate(rank):
                results['rank{}_{}'.format(r, branch+1)] = rank_score[i]

        # Ensemble performance
        gallery_concat = np.concatenate(gallery_embs, axis=-1)
        query_concat = np.concatenate(query_embs, axis=-1)
        rank_score, mAP_score = _evaluate_metrics(gallery_concat,
            query_concat, gallery_data, query_data, rank)

        results['mAP_ensemble'] = mAP_score
        for i, r in enumerate(rank):
            results['rank{}_ensemble'.format(r)] = rank_score[i]

    return results

def _get_emb(sess, gallery_iter, query_iter, n_branches=1):
    """Get embeddings for gallery and query sets"""

    def compute_emb(data_iter, n_branches):
        embs = []
        progbar = Progbar(len(data_iter.files_arr))

        if n_branches == 1:
            test_init_op = 'test_init_op'
            output = 'model/output/output:0'
            # output = 'output/BiasAdd:0'
        else:
            test_init_op = []
            output = []
            for i in range(n_branches):
                test_init_op.append('test_init_op_'+str(i+1))
                output.append('model/output/vb{}/output:0'.format(i+1))

        for i, batch in enumerate(data_iter):
            if isinstance(batch, np.ndarray):
                sess.run(test_init_op, feed_dict={'x:0':batch,
                    'batch_size:0':len(batch)})
                e = sess.run(output)
            else:
                # Test augmentation
                aug_outputs = []
                for i, aug_batch in enumerate(batch):
                    sess.run(test_init_op, feed_dict={'x:0':aug_batch,
                        'batch_size:0':len(aug_batch)})
                    aug_outputs.append(sess.run(output))
                # Mean has better performance vs. concatenate
                e = np.mean(aug_outputs, axis=0)

            # if n_branches > 1:
            #     e = np.concatenate(e, axis=-1)

            # print(e.shape)
            embs.append(e)
            progbar.add(e.shape[-2])

        return embs

    # Concatenate along batch dim
    print('Computing gallery embeddings...')
    gallery_embs = np.concatenate(compute_emb(gallery_iter, n_branches), axis=-2)
    print('Computing query embeddings...')
    query_embs = np.concatenate(compute_emb(query_iter, n_branches), axis=-2)

    return gallery_embs , query_embs

def _evaluate_metrics(gallery_embs, query_embs, gallery_data, query_data, rank):
    # print("Computing rank {} and mAP...".format(rank))
    # progbar = Progbar(len(query_data.files_arr))

    gallery_idts = np.array([p[1] for p in gallery_data.files_arr])
    gallery_cams = np.array([p[2] for p in gallery_data.files_arr])

    def evaluate(query_e, query_f, mem_correct, mem_AP):
        # print('Started process...')
        correct = [0 for _ in rank]
        AP = []

        for q, (_, idt, camera) in enumerate(query_f):
            b = np.logical_or(gallery_cams != camera, gallery_idts != idt)
            distance_vectors = np.power(np.squeeze(np.abs(gallery_embs[b] - query_e[q])), 2)
            distance = np.sqrt(np.sum(distance_vectors, axis=1))

            top_inds = distance.argsort()
            output_classes = gallery_idts[b][top_inds]

            # Calculate rank
            for i, r in enumerate(rank):
                if np.sum(gallery_idts[b][top_inds[:r]] == idt) > 0:
                    correct[i] += 1

            precision = []
            correct_old = 0

            # Compute mAP
            for t in range(distance.shape[0]):
                if idt == output_classes[t]:
                    precision.append(float(correct_old + 1) / (t + 1))
                    correct_old += 1

            mean_precision = np.mean(precision) if len(precision) > 0 else 0
            AP.append(mean_precision)

        for i in range(len(rank)):
            mem_correct[i] += correct[i]

        mem_AP += AP

    with multiprocessing.Manager() as manager:
        start = time()
        num_processes = multiprocessing.cpu_count()
        processes = []

        mem_correct = manager.list([0 for _ in rank])
        mem_AP = manager.list([])

        n_query = len(query_data.files_arr)
        for i in range(num_processes):
            q_begin = i * n_query // num_processes
            if i == num_processes - 1:
                q_end = n_query
            else:
                q_end = (i + 1) * n_query // num_processes

            query_e = query_embs[q_begin:q_end]
            query_f = query_data.files_arr[q_begin:q_end]

            process = multiprocessing.Process(target=evaluate,
                        args=(query_e, query_f, mem_correct, mem_AP))
            processes.append(process)

        for i in range(num_processes):
            processes[i].start()

        for i in range(num_processes):
            processes[i].join()

        correct = np.array(mem_correct[:], dtype='float32')
        rank_score = correct / n_query
        mAP = np.mean(mem_AP[:])

        print('Elapsed time:', time() - start)

    return rank_score, mAP

# def _evaluate_metrics(gallery_embs, query_embs, gallery_data, query_data, rank):
#     gallery_idts = np.array([p[1] for p in gallery_data.files_arr])
#     gallery_cams = np.array([p[2] for p in gallery_data.files_arr])
#
#     correct = np.array([0] * len(rank))
#     test_iter = np.array([0] * len(rank))
#     AP = []
#
#     print("Computing rank {} and mAP...".format(rank))
#     progbar = Progbar(len(query_data.files_arr))
#
#     for q in range(len(query_data.files_arr)):
#         _, idt, camera = query_data.files_arr[q]
#
#         b = np.logical_or(gallery_cams != camera, gallery_idts != idt)
#         q_emb = query_embs[q]
#         distance_vectors = np.power(np.squeeze(np.abs(gallery_embs[b] - q_emb)), 2)
#         distance = np.sqrt(np.sum(distance_vectors, axis=1))
#
#         top_inds = distance.argsort()
#         output_classes = gallery_idts[b][top_inds]
#
#         # Calculate rank
#         for r in range(len(rank)):
#             r_top_inds = top_inds[:rank[r]]
#             r_output_classes = gallery_idts[b][r_top_inds]
#
#             if np.where(r_output_classes == idt)[0].shape[0] > 0:
#                 correct[r] += 1
#             test_iter[r] += 1
#
#         precision = []
#         correct_old = 0
#
#         # Compute mAP
#         for t in range(distance.shape[0]):
#             if idt == output_classes[t]:
#                 precision.append(float(correct_old + 1) / (t + 1))
#                 correct_old += 1
#
#         AP.append(np.mean(np.array(precision)))
#
#         if (q + 1) % 100 == 0 or (q + 1) == len(query_data.files_arr):
#             progbar.update(q + 1)
#
#     rank_score = correct.astype(np.float32) / test_iter
#     mAP = np.mean(AP)
#     return rank_score, mAP
