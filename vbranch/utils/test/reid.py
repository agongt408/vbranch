from ... datasets.reid import TestingDataGenerator as TestGen

import numpy as np
from tensorflow.keras.utils import Progbar

def get_score(sess, dataset='market', preprocess=True, img_dim=(128,64,3),
        crop=False, flip=False, rank=[1,5]):

    assert dataset in ['market', 'duke']

    gallery_data = TestGen(dataset, 'test', preprocess, img_dim, crop, flip)
    query_data = TestGen(dataset, 'query', preprocess, img_dim, crop, flip)

    # if flip == 'concat':
    #     gallery_embs, query_embs = _get_emb_concat(model,
    #         gallery_data, query_data)
    # elif flip == 'mean':
    #     gallery_embs, query_embs = _get_emb_mean(model,
    #         gallery_data, query_data)
    # else:
    gallery_embs,query_embs = _get_emb(sess, gallery_data, query_data)

    rank_score, mAP_score = _evaluate_metrics(gallery_embs, query_embs,
        gallery_data, query_data, rank, compute_mAP=True)

    results = {'mAP': mAP_score}
    for i, r in enumerate(rank):
        results['rank'+str(r)] = rank_score[i]

    return results

def _get_emb(sess, gallery_iter, query_iter, n_branches=1):
    def compute_emb(data_iter, n_branches):
        embs = []
        progbar = Progbar(len(data_iter.files_arr))

        if n_branches == 1:
            test_init_op = 'test_init_op'
            output = 'model/output/output:0'
        else:
            test_init_op = []
            output = []
            for i in range(n_branches):
                test_init_op.append('test_init_op_'+str(i+1))
                output.append('model/output/vb{}_output:0'.format(i+1))

        for it in data_iter:
            sess.run(test_init_op, feed_dict={'x:0':it,'batch_size:0':len(it)})
            e = sess.run(output)

            if n_branches > 1:
                e = np.concatenate(e, axis=-1)

            embs.append(e)
            progbar.add(len(e))

        return embs

    print('Computing gallery embeddings...')
    gallery_embs = np.concatenate(compute_emb(gallery_iter, n_branches), axis=0)
    print('Computing query embeddings...')
    query_embs = np.concatenate(compute_emb(query_iter, n_branches), axis=0)

    return gallery_embs , query_embs

def _get_emb_mean(model, gallery_iter, query_iter):
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

def _get_emb_concat(model, gallery_iter, query_iter):
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

        if (q + 1) % 100 == 0 or (q + 1) == len(query_data.files_arr):
            progbar.update(q + 1)

    rank_score = correct.astype(np.float32) / test_iter
    mAP = np.mean(AP)
    return rank_score, mAP
