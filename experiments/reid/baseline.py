import sys
sys.path.insert(0, '.')

from vbranch.datasets.reid import TripletDataGenerator, TestingDataGenerator
from vbranch.applications import SimpleCNNLarge, ResNet50, DenseNet121
from vbranch.losses import triplet
from vbranch.callbacks import reid_acc
from vbranch.utils import *

import tensorflow as tf
import numpy as np
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', default='market',
                    nargs='?', choices=['market'], help='dataset')
parser.add_argument('--architecture', action='store', default='simple',
                    nargs='?', choices=['simple', 'resnet', 'densenet'],
                    help='model architecture, i.e., simple, resnet, densenet')
parser.add_argument('--P',action='store',default=18,nargs='?',type=int,help='P')
parser.add_argument('--K',action='store',default=4,nargs='?',type=int,help='K')

parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--trials', action='store', default=1, nargs='?', type=int,
                    help='number of trials to perform, if 1, then model_id used')
parser.add_argument('--epochs', action='store', default=250, nargs='?',
                    type=int, help='number of epochs to train model')
parser.add_argument('--model_id',action='store',nargs='*',type=int,default=[1],
                    help='list of checkpoint model ids')
parser.add_argument('--steps_per_epoch', action='store', default=100, nargs='?',
                    type=int, help='number of training steps per epoch')

IMG_DIM = (128, 64, 3)
OUTPUT_DIM = 128
T_0 = 150

def build_model(dataset, arch, train_generator, lr_scheduler, P, K):
    inputs, train_init_op, test_init_op = get_data_iterator_from_generator(
        train_generator, (None,)+IMG_DIM, n=1)

    lr = tf.placeholder('float32', name='lr')
    name = 'model'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if arch == 'simple':
            model = SimpleCNNLarge(inputs, OUTPUT_DIM, name=name)
        elif arch == 'resnet':
            model, assign_ops = ResNet50(inputs, OUTPUT_DIM, name=name,
                                         weights='imagenet')
        elif arch == 'densenet':
            model, assign_ops = DenseNet121(inputs, OUTPUT_DIM, name=name,
                                         weights='imagenet')

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Compile model
        callable_acc = reid_acc(dataset, n_branches=1, preprocess=arch,
                        buffer=1000, img_dim=IMG_DIM)
        model.compile(optimizer, triplet(P,K, margin='soft'),
                      train_init_op, test_init_op,
                      callbacks={'acc' : callable_acc},
                      schedulers={'lr:0': lr_scheduler},
                      assign_ops=assign_ops)

    return model

def train(dataset, arch, model_id, epochs, steps_per_epoch, P, K):
    dirpath = get_dir_path(dataset, arch)
    model_path = os.path.join('models', dirpath, 'model_{}'.format(model_id))
    os.system('mkdir -p ' + model_path)

    p_console('Save model path: '+ model_path)

    train_generator = TripletDataGenerator(dataset, 'train',
                                           P=P, K=K,
                                           preprocess=arch,
                                           img_dim=IMG_DIM)

    tf.reset_default_graph()
    lr_scheduler = lr_exp_decay_scheduler(0.0003, T_0, epochs, 0.001)
    model = build_model(dataset, arch, train_generator, lr_scheduler, P, K)
    model.summary()

    history = model.fit(epochs, steps_per_epoch, log_path=model_path, call_step=10)
    save_results(history, dirpath, 'train_%d.json' % model_id)

def test(dataset, arch, model_id_list):
    dirpath = get_dir_path(dataset, arch)

    gallery_data = TestGen(dataset, 'test', img_dim=IMG_DIM, preprocess=arch)
    query_data = TestGen(dataset, 'query', img_dim=IMG_DIM, preprocess=arch)

    gallery_embs = []
    query_embs = []
    for model_id in model_id_list:
        model_path = os.path.join('models', dirpath, 'model_{}'.format(model_id))
        tf.reset_default_graph()
        with TFSessionGrow() as sess:
            restore_sess(sess, model_path)
            gallery, query = get_emb(sess, gallery_data, query_data)
            gallery_embs.append(gallery)
            query_embs.append(query)

    results = get_score(sess=None, dataset=dataset,
        n_branches=len(model_id_list), embeddings=(gallery_embs, query_embs),
        img_dim=IMG_DIM, preprocess=arch)

    print(results)
    save_results(results, dirpath, 'test.csv', mode='a')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        p_console('MODE: TEST')

        if args.trials == 1:
            # args.model_id is a list of model ids
            test(args.dataset, args.architecture, args.model_id)
        else:
            avail_runs = glob('models/{}-{}_*'.format(args.dataset,
                args.architecture))
            avail_ids = [int(path[path.index('_')+1:]) for path in avail_runs]

            for i in range(args.trials):
                # Choose models for ensemble
                model_ids = np.random.choice(avail_ids, len(args.model_id),
                    replace=False)
                model_ids.sort()
                test(args.dataset, args.architecture, model_ids)
    else:
        p_console('MODE: TRAIN')

        if args.trials == 1:
            for id in args.model_id:
                train(args.dataset,args.architecture, id,
                    args.epochs, args.steps_per_epoch, args.P, args.K)
        else:
            for i in range(args.trials):
                train(args.dataset, args.architecture, i+1,
                    args.epochs,args.steps_per_epoch, args.P, args.K)

    print('Finished!')
