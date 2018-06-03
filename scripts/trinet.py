import argparse
import json
import os
import pprint as pp
import numpy as np
import sys

if os.path.exists('/home/ubuntu/'):
    sys.path.insert(0, '/home/ubuntu/keras-2.0.8/')
    sys.path.append('/home/ubuntu/albert/scripts/')
    sys.path.append('/home/ubuntu/albert/src/')
else:    
    sys.path.append('/home/albert/github/tensorflow/scripts/')
    sys.path.append('/home/albert/github/tensorflow/src/')
    
from utils import get_parser as gp
from utils import line_break
import evaluation,data, training, models

from keras.callbacks import LearningRateScheduler, History

# https://docs.python.org/2/library/argparse.html#action

MODELS_ROOT = '/home/albert/github/tensorflow/models/'
if not os.path.exists(MODELS_ROOT):
    MODELS_ROOT = '/home/ubuntu/albert/models/'

'''Gather training parameters'''

def get_training_params_from_parser_args(args):
    if args.no_weights:
        if args.weights_file is not None:
            raise ValueError, 'either --no_weights or --weights_file ' + \
                'must be True or not None but not both'
        else:
            weights = None
    else:
        if args.weights_file is not None:
            weights = args.weights_file
        else:
            weights = 'imagenet'

    #Determine file root
    file_root = args.file_root + '_' + 'P%dK%d' % (args.P_param, args.K_param)

    # if model name with 'file_root' already exists,
    # append version number to end
    max_n = 0
    for model_name in os.listdir(MODELS_ROOT):
        print model_name
        if model_name == file_root:
            n = 1
            if n > max_n:
                max_n = n

        elif model_name.find(file_root) > -1:
            try:
                n = int(model_name[-(model_name[::-1].index('_')):])
                if n > max_n:
                    max_n = n
            except ValueError:
                continue
    if max_n >= 1:
        file_root += '_' + str(max_n + 1)

    params = {
        'P' : args.P_param,
        'K' : args.K_param,
        'era' : args.era,
        't1' : args.t1,
        'dataset' : args.dataset,
        'weights' : weights,
        'eval_era' : args.eval_era,
        'batch_norm_diagnostic' : args.diagnostic,
        'epochs_per_era' : 10,
        'steps_per_epoch' : 100,
        'file_root' : file_root,
        'memory_fraction' : args.memory_fraction,
        'start' : 1,
        'blocks' : args.blocks,
    }

    return params


def get_training_report_from_json(file_root):
    path = os.path.join(MODELS_ROOT, file_root, file_root + '_report.json')

    if not os.path.exists(path):
        raise ValueError, \
            'json file with root %s does not exist' % file_root

    with open(path, 'r') as f:
        report_dict = json.load(f)
        params = report_dict['params']
        rank = report_dict['rank']
        mAP = report_dict['mAP']
        loss = report_dict['loss']

    max_it = 0
    for weights_file in os.listdir(
        os.path.join(MODELS_ROOT, params['file_root'])):
        print weights_file
        try:
            it = int(weights_file[-(weights_file[::-1].index('_')):\
                    weights_file.index('.')])
            if it > max_it:
                max_it = it
        except ValueError:
            continue

    print max_it
    last_era = max_it / (params['steps_per_epoch'] * params['epochs_per_era'])

    params['start'] = last_era + 1

    return params, rank, mAP, loss


args = gp().parse_args()

if args.memory_fraction < 1.0:
    training.set_gpu_memory_fraction(args.memory_fraction)
elif args.memory_fraction > 1.0:
    raise ValueError, 'memory_fraction must be between 0.0 and 1.0'


if args.test is not None:
    '''Test model'''
    
    if args.continue_from_root is None:
        rank, mAP, loss = {}, {}, []
        
        params = {
            'file_root' : args.test[0],
            'dataset' : args.dataset,
            'blocks' : args.blocks,
        }
    else:
        params, rank, mAP, loss = get_training_report_from_json(
            args.continue_from_root)
            
    print '#' * 64
    pp.pprint(params)
    print '#' * 64
    
    training_report = {
        'rank' : rank,
        'mAP' : mAP,
        'loss' : loss,
        'params' : params
    }
    
    print training_report
    
    model = models.TriNet(blocks=params['blocks'])
    
    if params['batch_norm_diagnostic']:
        print model.summary()
    
    for it in args.test[1:]:
        path = os.path.join(MODELS_ROOT, params['file_root'], 
                    params['file_root'] + '_' + str(it) + '.npy')
        if os.path.exists(path):
            print path
            
            model.set_weights(np.load(path))
            s = evaluation.get_score(model, dataset=params['dataset'])
            print s
            
            training_report['rank'][it] = s['rank']
            training_report['mAP'][it] = s['mAP']
            training_report['loss'] += s['loss']
            
            with open(os.path.join(MODELS_ROOT, params['file_root'], \
                    '%s_report.json' % params['file_root']), 'w') as fp:
                json.dump(training_report, fp)
        else:
            print 'file with path %s does not exist' % path
        
else:
    '''Train model'''
    
    if args.continue_from_root is None:
        params = get_training_params_from_parser_args(args)
    else:
        params, rank, mAP, loss = get_training_report_from_json(
            args.continue_from_root)
            
    line_break()
    pp.pprint(params)
    line_break()

    # Load training data
    train_dict, train_files = data.get_data(
                                    'train', dataset=params['dataset'])
    print 'Training data loaded successfully'

    if args.continue_from_root is None:
        model = models.TriNet(params['P'], params['K'], params['weights'],
            diagnostic=params['batch_norm_diagnostic'], blocks=params['blocks'])
    else:
        model = models.TriNet(params['P'], params['K'], None,
            diagnostic=params['batch_norm_diagnostic'], blocks=params['blocks'])

        last_it = (params['start'] - 1) * params['steps_per_epoch'] \
            * params['epochs_per_era']
        weights_path = os.path.join(MODELS_ROOT, params['file_root'], \
            params['file_root'] + '_' + str(last_it) + '.npy')

        model.set_weights(np.load(weights_path))

        print 'Weights from previous training session loaded ' + \
            'successfully: ' + weights_path
            
    if params['batch_norm_diagnostic']:
        print model.summary()

    history = History()

    if args.continue_from_root is None:
        training_report = {
            'rank' : {}, 'mAP' : {}, 'loss' : [], 'params' : params
        }
    else:
        training_report = {
            'rank' : rank,
            'mAP' : mAP,
            'loss' : loss,
            'params' : params
        }

    # model.set_weights()

    for era in range(params['start'],params['era'] + 1):
        iterations = era * params['steps_per_epoch'] * \
                        params['epochs_per_era']
        lrate = LearningRateScheduler(
                    training.step_decay_cont(params['epochs_per_era'], era))

        line_break()
        print 'era, ' + str(era)

        model.fit_generator(data.batch_generator(
                                train_dict, P=params['P'], K=params['K'],
                                preprocess=True, shape=(256,128)),
                             steps_per_epoch=params['steps_per_epoch'],
                             epochs=params['epochs_per_era'],
                             callbacks=[lrate,history])
        
        if not os.path.exists(os.path.join(MODELS_ROOT, params['file_root'])):
            os.system('mkdir ' + os.path.join(MODELS_ROOT, params['file_root']))
        
        path = training.save_weights(
                        model, it=iterations, root=params['file_root'])
        print 'Weights saved at', path

        if params['eval_era'] > 0:
            if era % params['eval_era'] == 0 or era == params['eval_era']:
                s = evaluation.get_score(
                            model, hist=history, dataset=params['dataset'])
                print s
                training_report['rank'][iterations] = s['rank']
                training_report['mAP'][iterations] = s['mAP']
                training_report['loss'] += s['loss']

                with open(os.path.join(MODELS_ROOT, params['file_root'],
                            '%s_report.json' % params['file_root']), 'w') as fp:
                    json.dump(training_report, fp)
        else:
            with open(os.path.join(MODELS_ROOT, params['file_root'],
                        '%s_report.json' % params['file_root']), 'w') as fp:
                json.dump(training_report, fp)
