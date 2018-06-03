from keras.layers.core import Dense, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

import numpy as np
import tensorflow as tf

import sys
sys.path.append('/home/albert/github/DenseNet/')
sys.path.append('/home/ubuntu/DenseNet/')
import densenet

sys.path.append('/home/albert/github/tensorflow/')
sys.path.append('/home/ubuntu/albert/')
from dense_bk import __dense_block
from src import ModelConfig, DenseNetBlockImageNet121, ResNet50
import losses


def TriNet(P_param=1, K_param=1, weights=None, shape=(256,128),
            base='densenet', output_dim=128, dropout_rate=0.0,
            diagnostic=False, comp=True):
    '''
    Instantiates TriNet model (https://arxiv.org/pdf/1703.07737.pdf)
    # Arguments:
        P_param: P (see paper)
        K_param: K (see paper)
        weights: None, 'imagenet', or path to weights saved as npy file
        shape: input shape
        base: 'densenet' or 'resnet'
        output_dim: number of units in final embedding (FC) layer
        dropout_rate: fraction of units set to 0 in each layer
    # Returns:
        TriNet model compiled with triplet loss function
    # Raises:
        ValueError if base is not supported
    '''

    if base is 'densenet':
        if weights is None or weights is 'imagenet':
            trinet = DenseNetBlockImageNet121(
                shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None,
                weights=weights, output_dim=output_dim,
                dropout_rate=dropout_rate, diagnostic=diagnostic)
        else:
            trinet = DenseNetBlockImageNet121(
                shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None,
                weights=None, output_dim=output_dim,
                dropout_rate=dropout_rate, diagnostic=diagnostic)

            trinet.set_weights(np.load(weights))

    elif base is 'resnet':
        if weights is None or weights is 'imagenet':
            base = ResNet50(include_top=False, weights=weights,
                            input_shape=(shape + (3,)))

            pred = GlobalAveragePooling2D()(base.layers[-2].output)
            pred = Dense(1024)(pred)
            pred = BatchNormalization(diagnostic=diagnostic)(pred)
            pred = Activation('relu')(pred)
            pred = Dense(128)(pred)

            trinet = Model(inputs=base.input, outputs=pred)
        else:
            base = ResNet50(include_top=False, weights=None,
                            input_shape=(shape + (3,)))

            pred = GlobalAveragePooling2D()(base.layers[-2].output)
            pred = Dense(1024)(pred)
            pred = BatchNormalization(diagnostic=diagnostic)(pred)
            pred = Activation('relu')(pred)
            pred = Dense(128)(pred)

            trinet = Model(inputs=base.input, outputs=pred)

            trinet.set_weights(np.load(weights))
    else:
        raise ValueError, 'base must either be densenet or resnet'

    if comp:
        _compile(trinet, 'triplet', P_param, K_param)

    return trinet


def PoseNet(P_param=1, K_param=1, weights=None, shape=(256,128),
            cam_dim=(16,8), cam_weight=0.2):
    if weights is None or weights is 'imagenet':
        posenet = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3],
            cam_dim=cam_dim, weights=weights)
    else:
        posenet = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3],
            cam_dim=cam_dim, weights=None)
        posenet.set_weights(np.load(weights))

    posenet.compile(loss=[losses.triplet(P_param=P_param,
                        K_param=K_param, output_dim=128), losses.cam_loss],
                        loss_weights=[1.0, cam_weight],
                        optimizer=Adam(lr=0.0003, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, decay=0.0))
    return posenet


def MergeNet(P_param=1, K_param=1, weights=None, shape=(256,128),
                cam_dim=None, keypoints=['neck', 'hip'], branch_block=3,
                cam_weight=0.2):
    '''Instantiates MergeNet with each branch "localized" around a
    specific body region (i.e., neck, hip) or simple ensemble (no CAM).
    Loss function implementation using CAMs.

    # Arguments:
        cam_dim: 2D dimension of CAM; feature map is resized using bilinear interpolation
            if cam_dim is greater in spatial dimension than initial feature map
        keypoints: array of body regions, also name of branches
        branch_block: block number in DenseNet to branch
        cam_weight: loss weight of cam in each branch

    # Returns:
        MergeNet model compiled with triplet loss and CAM loss
    '''

    if weights is 'imagenet':
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None, weights=weights)
    else:
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None, weights=None)

    base.layers[-1].name = 'base_output'

    input_arr = [base.input]
    cam_arr = []
    pred_arr = [base.output]

    for k in keypoints:
        if weights is 'imagenet':
            branch = DenseNetBlockImageNet121(
                shape + (3,), blocks=[3], cam_dim=cam_dim, weights=weights)
        else:
            branch = DenseNetBlockImageNet121(
                shape + (3,), blocks=[3], cam_dim=cam_dim, weights=None)

        print branch.input_shape
        branch.name = k
        branch_output = branch(base.layers[310].output)

        if cam_dim is None:
            pred_arr.append(branch_output)
        else:
            pred_arr.append(branch_output[0])
            cam_arr.append(branch_output[1])

    if len(keypoints) > 0:
        pred = concatenate(pred_arr)
    else:
        pred = trinet.output

    model = Model(inputs=base.input, outputs=([pred] + cam_arr))

    if weights is not None and weights is not 'imagenet':
        model.set_weights(np.load(weights))

    if cam_dim is None:
        model.compile(loss=losses.triplet(P_param=P_param,
                                        K_param=K_param,
                                        output_dim=128),
                    optimizer=Adam(lr=0.0003,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0))
    else:
        model.compile(loss=([losses.triplet(P_param=P_param,
                                            K_param=K_param,
                                            output_dim=128)] \
                            + len(keypoints) * [losses.cam_loss]),
                loss_weights=([1.0] + len(keypoints) * [cam_weight]),
                optimizer=Adam(lr=0.0003, beta_1=0.9,
                                beta_2=0.999, epsilon=1e-08,
                                decay=0.0))

    return model


def MergeNet_Or(P_param=1, K_param=1, weights=None, shape=(256,128)):
    if weights is 'imagenet':
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], weights=weights)
    else:
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], weights=None)

    shared_blocks = Model(inputs=base.input, outputs=base.layers[310].output)

    input_arr = []
    pred_arr = []

    for i in range(4):
        branch_input = Input(shape=(shape + (3,)))

        if weights is 'imagenet':
            branch = DenseNetBlockImageNet121(shape + (3,),
                        blocks=[3], weights=weights)
        else:
            branch = DenseNetBlockImageNet121(shape + (3,),
                        blocks=[3], weights=None)

        branch.name = 'branch_' + str(i + 1)

        input_arr.append(branch_input)
        pred_arr.append(branch(shared_blocks(branch_input)))

    pred = concatenate(pred_arr)

    model = Model(inputs=input_arr, outputs=pred)

    if weights is not None and weights is not 'imagenet':
        model.set_weights(np.load(weights))

    model.compile(loss=losses.triplet(P_param=P_param,
                                        K_param=K_param,
                                        output_dim=128),
                    optimizer=Adam(lr=0.0003, beta_1=0.9,
                                    beta_2=0.999, epsilon=1e-08,
                                    decay=0.0))

    return model


def MergeNet_Dense(P_param=1, K_param=1, weights=None, shape=(256,128),
                    branches=3):
    if weights is 'imagenet':
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None,
            weights=weights, include_top=False)
    else:
        base = DenseNetBlockImageNet121(
            shape + (3,), blocks=[-1,0,1,2,3], cam_dim=None,
            weights=None, include_top=False)

    pred_arr = []

    for i in range(branches):
        pred = Dense(1024)(base.output)
        pred = BatchNormalization()(pred)
        pred = Activation('relu')(pred)
        pred = Dense(128)(pred)
        pred_arr.append(pred)

    if branches == 1:
        final_output = pred
    else:
        final_output = concatenate(pred_arr)

    model = Model(inputs=base.input, outputs=final_output)

    if weights is not None and weights is not 'imagenet':
        model.set_weights(np.load(weights))

    model.compile(loss=losses.triplet(P_param=P_param,
                                        K_param=K_param,
                                        output_dim=128),
                optimizer=Adam(lr=0.0003, beta_1=0.9,
                                beta_2=0.999, epsilon=1e-08,
                                decay=0.0))

    return model


def MergeNet_Drop(base, masks, P_param=1, K_param=1, weights=None,
                tile=False, comp=False):
    '''
    Generates compiled MergeNet-Drop model given a base model and
    list of binary masks.

    Args:
        base: base model
        masks: list of binary masks
        ...
        weights: path to model weights file (np array)
        dense_only: if true, only apply masks to final dense block,
            i.e., after global average pooling layer

    Returns:
        compiled model
    '''

    def __get_nth_pool_layer(n, config):
        n_pool = 0
        pool_layer = 0
        for l in range(len(config.layers) - 1):
            if config.layers[l + 1].class_name.find('Pooling') > -1:
                if n_pool == n:
                    pool_layer = l + 1
                    break
                n_pool += 1

        return pool_layer

    def __branch(l, n_add, first_branch, branches, amend_inbound_nodes=True):
        inbound_name = config.layers[l + n_add].name
        outbound_layer = config.layers[l + 1 + n_add]

        remove = True

        for b in range(branches):
            if first_branch < branches:
                n_add = __add_mask_layer(l, n_add, masks, b)
                n_add = __add_drop_layer(l, n_add, inbound_name, 0)
                first_branch += 1
            else:
                n_add = __add_mask_layer(l, n_add, masks, b)
                n_add = __add_drop_layer(l, n_add, inbound_name, b)

            if amend_inbound_nodes:
                remove = __amend_inbound_nodes(l, n_add, outbound_layer,
                                                remove)
        return n_add, first_branch

    def __add_mask_layer(l, n_add, masks, b):
        ip_op = Input(tensor=K.constant(masks[l][b]))
        config.add_input(ip_op, l + n_add)
        n_add += 1
        return n_add

    def __add_drop_layer(l, n_add, inbound_name, ib_n):
        drop_op = Lambda(lambda x : tf.multiply(x[0], x[1]))
        config.add_layer('Lambda', drop_op, l + n_add,
                        [[[inbound_name, ib_n],
                        [config.layers[l + n_add].name, 0]]], [])
        n_add += 1
        return n_add

    def __amend_inbound_nodes(l, n_add, outbound_layer, remove):
        l_ib_n = outbound_layer.inbound_nodes.tolist()

        if remove:
            l_ib_n.pop(0)
            remove = False

        l_ib_n.append([[config.layers[l + n_add].name, 0]])
        outbound_layer.inbound_nodes = np.array(l_ib_n)
        # print outbound_layer.inbound_nodes

        return remove

    def __gather_final(l, n_add, branches, masks):
        inbound_name = config.layers[l + n_add].name

        for b in range(branches):
            indices = []
            where = np.where(masks[-1][b] > 0)[0]
            for i in range(P_param * K_param):
                idx = where.reshape((where.shape[0],1))
                idx = np.concatenate([np.ones((where.shape[0],1)) \
                                    * i, idx], axis=1).\
                                    astype(np.int32).tolist()
                indices += idx

            ip_op = Input(tensor=K.constant(indices, dtype=tf.int32))
            config.add_input(ip_op, l + n_add)
            n_add += 1

            reshape_op = Input(
                tensor=K.variable([where.shape[0]]))
            config.add_input(reshape_op, l + n_add)
            n_add += 1

            gather_op = Lambda(lambda x : gather(x[0], x[1], x[2]))

            config.add_layer('Lambda', gather_op, l + n_add,
                            [[[inbound_name, b],
                            [config.layers[l + n_add - 1].name, 0],
                            [config.layers[l + n_add].name, 0]]], [])
            n_add += 1
        return n_add

    init_config = ModelConfig()
    init_config.from_model(base)
    _, x_list = init_config.reconstruct_model()

    # Reset config so that each layer starts with one output node
    config = ModelConfig()
    config.from_model(base)

    n_add = 0
    first_branch = 0
    branches = len(masks[-1])

    for l in range(len(masks) - 1):
        if len(masks[l]) > 0:
            outbound_layer = config.layers[l + 1 + n_add]
            remove = True

            if outbound_layer.class_name == 'Concatenate':
                inbound_name = config.layers[l + n_add].name # Conv2D layer
                concat_ib_n = None

                for b in range(branches):
                    n_add = __add_mask_layer(l, n_add, masks, b)
                    n_add = __add_drop_layer(l, n_add, inbound_name, b)

                    l_ib_n = outbound_layer.inbound_nodes.tolist()

                    if remove:
                        concat_ib_n = l_ib_n.pop(0)
                        remove = False

                    if first_branch <= 2 * branches:
                        l_ib_n.append([concat_ib_n[0],
                            [config.layers[l + n_add].name, 0]])
                        first_branch += 1
                    else:
                        l_ib_n.append([[concat_ib_n[0][0], b],
                            [config.layers[l + n_add].name, 0]])
                    outbound_layer.inbound_nodes = np.array(l_ib_n)
            else:
                if config.layers[l + n_add].class_name != 'Concatenate':
                    n_add, first_branch = __branch(
                        l, n_add, first_branch, branches)
                else:
                    remove = True
                    for b in range(branches):
                        l_ib_n = outbound_layer.inbound_nodes.tolist()

                        if remove:
                            l_ib_n.pop(0)
                            remove = False

                        l_ib_n.append([[config.layers[l + n_add].name, b]])
                        outbound_layer.inbound_nodes = np.array(l_ib_n)

    # Gather embeddings from final layer
    n_add = __gather_final(
        len(init_config.layers) - 1, n_add, branches, masks)

    if tile:
        config.layers[360].inbound_nodes[0][0][0] = 'l_313'
        config.layers[360].inbound_nodes[1][0][0] = 'l_315'
        config.layers[360].inbound_nodes[2][0][0] = 'l_317'

    # return config

    output_idx = []
    for b in range(branches - 1, -1, -1):
        output_idx.append(-(3*b + 1))

    model_rec, _ = config.reconstruct_model(
                    output_idx=output_idx, name='MergeNet_Drop')

    if len(model_rec.outputs) == 1:
        model = Model(model_rec.input, model_rec.outputs)
    else:
        print model_rec.outputs
        model = Model(model_rec.input, concatenate(model_rec.outputs, axis=1))

    if weights is not None:
        model.set_weights(np.load(weights))

    if comp:
        _compile(model, 'triplet_drop', P_param, K_param, masks[-1])

    return model


def DenseNet121Drop(P_param=1, K_param=1, branches=3, overlap=0,
                    shape=(256,128), weights=None, nth_pool=3,
                    diagnostic=False):

    base = TriNet(P_param, K_param, None, shape,
        output_dim=128*branches, diagnostic=diagnostic, comp=False)

    l_start = _get_nth_pool_layer(nth_pool, base)

    masks = generate_model_masks(base, branches, l_start, overlap)

    regularizers = _get_regularizers(base, masks)
    batch_norm_list = _get_batch_norm_list(base, masks)

    nb_layers_per_block = [6, 12, 24, 16]
    nb_filter_per_block = [64, 128, 256, 512]

    x, _ = __dense_block(base.layers[l_start].output, nb_layers=16,
                        nb_filter=512, growth_rate=32, bottleneck=True,
                        weight_decay=0.0001, regularizers=regularizers,
                        batch_norm_list=batch_norm_list[:-2] \
                            if batch_norm_list else None)

    x = batch_norm_list[-2](x) if batch_norm_list else BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024)(x)
    x = batch_norm_list[-1](x) if batch_norm_list else BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(np.minimum(128*branches, 128*branches))(x)

    model = Model(base.input, x)

    if weights is not None:
        weights_model = TriNet(P_param, K_param,
            weights if weights is not None else None, shape,
            output_dim=128, diagnostic=diagnostic, comp=False)

        for l in range(len(weights_model.layers) - 1):
            model.layers[l].set_weights(
                weights_model.layers[l].get_weights())
        model.layers[-1].set_weights([
            np.tile(weights_model.layers[-1].get_weights()[0],
                    (1,np.minimum(branches, branches))),
            np.tile(weights_model.layers[-1].get_weights()[1],
                    (np.minimum(branches, branches)))])

    model_drop = MergeNet_Drop(model, masks, P_param, K_param, comp=True)

    # _compile(model_drop, 'triplet_drop', P_param, K_param, [masks[-1][0]])

    if diagnostic:
        return model_drop, weights_model, masks

    return model_drop


def StackedDenseNet121Drop(P_param=1, K_param=1, branches=3,
                        shape=(256,128), weights=None, dense_only=False,
                        diagnostic=False):
    '''
    Notes:
    Kernel regularizer (Conv2D):
        - Declare regularizers externally to have reference handles
        to regularizer objects and include indices (idx) arg

        - keras/regularizers.py
            __init__()
                mod: Added indices property to only apply regularizer to
                parameters being trained (line 35)
            __call__()
                mod: gather elements corresponding to indices (line 48)
            get_config()
                mod: update config dict to show indices (line 61)

    Batch normalization:
        - Do not apply moving average update to masked parameters:
        estimated mean, estimated variance

        - keras/layers/normalization.py
            __init__()
                mod: added mask_list and n_calls args (lines 70, 71)
            call()
                mod: added mask to update op (lines 191, 195)
            get_config():
                mod: added mask_list and n_calls items (lines 220, 221)

        - keras/backend/tensorflow_backend.py
            moving_average_update()
                mod: added mask argument (line 915)

        - tensorflow/python/training/moving_averages.py
            assign_moving_average()
                mod: added mask arg, multiply update_delta
                by mask (line 78)

    Args:
        See MergeNet_Drop()

    Returns:
        Compiled MergeNet-Drop of Stacked Dense model
    '''

    model = _stacked_dense_block(P_param, K_param, None, shape, branches)

    masks = generate_model_masks(model, branches, 311, 0)

    regularizers = _get_regularizers(model, masks)
    batch_norm_list = _get_batch_norm_list(model, masks)

    model, base = _stacked_dense_block(P_param, K_param, weights, shape,
                        branches, regularizers, batch_norm_list, True)

    model_drop = MergeNet_Drop(model, masks, P_param, K_param,
                dense_only=dense_only, tile=True)

    if diagnostic:
        return model_drop, base

    return model_drop


def _get_regularizers(model, masks):
    branches = len(masks[-1])
    regularizers = []

    for l in range(len(masks)):
        if len(masks[l]) > 0:
            try:
                if model.layers[l].kernel_regularizer is not None:
                    w = np.ones((model.layers[l].get_weights()[0].shape))

                    idx_list = []
                    for b in range(branches):
                        w = np.ones((model.layers[l].get_weights()[0].shape))

                        for i in np.nditer(np.where(masks[l - 1][b] == 1)[0]):
                            for j in np.nditer(np.where(masks[l][b] == 1)[0]):
                                w[:, :, i, j] = 0

                        idx_list.append(np.concatenate(
                            np.where(w == 0)).reshape((4, -1)).transpose())

                    idx_unique = np.unique(
                        np.concatenate(idx_list, axis=0), axis=0)
                    regularizers.append(l2(0.0001, idx=idx_unique))
            except:
                continue
    return regularizers

def _get_batch_norm_list(model, masks):
    batch_norm_list = []
    for l in range(len(masks)):
        if len(masks[l]) > 0:
            if model.layers[l].name.find('batch_norm') > -1:
                batch_norm_list.append(BatchNormalization(
                    axis=-1, epsilon=1.1e-5,
                    mask_list=masks[l] if masks else None))
    return batch_norm_list


def _stacked_dense_block(P_param=1, K_param=1, weights=None,
                            shape=(256,128), branches=3,
                            regularizers=None, batch_norm_list=None,
                            diagnostic=False):
    '''
    Replaces last dense block in DenseNet121 with stacked version, i.e.,
    3 x 'growth_rate' and 3 x FC layer units

    Args:
        weights: None or 'imagenet'
            if 'imagenet', weights are tiled according to model
            architecture
        shape: dimensions of input image, e.g., (256,128)
        branches: number of blocks to stack
        regularizers: list of regularizers, e.g., l2
        batch_norm_list: list of batch norm objects

    Returns:
        Compiled model with stacked last dense and FC blocks
    '''

    if weights not in {None, 'imagenet'}:
        raise ValueError, "weights must either be none or 'imagenet'"

    base = DenseNetBlockImageNet121(shape + (3,), blocks=[-1,0,1,2,3],
                                    weights=weights, output_dim=128,
                                    fc1=1024)

    third_pool_layer = _get_nth_pool_layer(3, base)

    ip = Lambda(lambda x : K.tile(x, [1,1,1,3]))(base.\
            layers[third_pool_layer].output)

    x, _ = __dense_block(ip, nb_layers=16, nb_filter=512,
                        growth_rate=branches*32, bottleneck=True,
                        weight_decay=0.0001,
                        regularizers=regularizers,
                        batch_norm_list=batch_norm_list[:-2] \
                            if batch_norm_list else None)

    x = batch_norm_list[-2](x) if batch_norm_list else BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024 * branches)(x)
    x = batch_norm_list[-1](x) if batch_norm_list else BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128 * branches)(x)

    model = Model(base.input, x)

    _compile(model, loss='triplet', P_param=P_param, K_param=K_param)

    if weights == 'imagenet':
        transfer_stacked_weights(base, model, third_pool_layer + 1)

    if diagnostic:
        return model, base

    return model


def generate_model_masks(model, branches, l_start=311, overlap=0):
    '''
    Generate masks for MergeNet_Drop

    Args:
        branches: number of branches > 0
        l_start: nth layer to start branching, all layers before l_start
            have empty lists as masks, i.e., no branching applied
        overlap: degree of overlap (0 to 1)

    Returns:
        list containing masks for each layer
    '''

    masks = []
    concat_mask = [[] for b in range(branches)]
    l_add_conv = 0

    for l in range(len(model.layers)):
        if l == l_start:
            m = gen_layer_mask(model.layers[l].output_shape[-1],
                                branches, overlap, value=1)
            for b in range(branches):
                concat_mask[b].append(m[b])
            masks.append(m)

        elif l > l_start:
            if model.layers[l].name.find('conv') > -1:
                masks.append(gen_layer_mask(model.layers[l].output_shape[-1],
                                            branches, overlap, value=1))
                l_add_conv = l

            elif model.layers[l].name.find('batch') > -1:
                if model.layers[l - 1].name.find('concatenate') > -1:
                    for b in range(branches):
                        concat_mask[b].append(masks[l_add_conv][b])

                    masks.append([np.concatenate(concat_mask[b], axis=0) \
                                for b in range(branches)])
                else:
                    masks.append(gen_layer_mask(model.layers[l].output_shape[-1],
                                                branches, overlap, value=1))

            elif model.layers[l].name.find('act') > -1:
                if model.layers[l - 2].name.find('concatenate') > -1:
                    masks.append([np.concatenate(concat_mask[b], axis=0) \
                                for b in range(branches)])
                else:
                    masks.append(gen_layer_mask(model.layers[l].output_shape[-1],
                                branches, overlap, value=1))

            elif model.layers[l].name.find('global') > -1:
                masks.append([np.concatenate(concat_mask[b], axis=0) \
                            for b in range(branches)])

            elif model.layers[l].name.find('dense') > -1:
                masks.append(gen_layer_mask(model.layers[l].output_shape[-1],
                                            branches, overlap, value=1))

            else:
                masks.append(gen_layer_mask(model.layers[l].output_shape[-1],
                                            branches, 1, value=1))
        else:
            masks.append([])

    return masks


def transfer_stacked_weights(donor, recipient, l_start=311, l_end=None,
                            align_from_end=True):
    '''
    Intelligently tranfer weights from donor model to recipient model
    while preserving donor weights structure.

    Args:
        donor: model to transfer from
        recipient: model to transfer to
        l_start: index of donor model to start tranfer (0 inclusive)
        l_end: index of donor model to stop transfer (0 inclusive),
            must be greater than l_start
        align_from_end: if true, takes care of added layers in recipient
            model
    '''

    if l_end:
        assert l_end >= l_start, 'l_end must be >= l_start'
    else:
        l_end = len(donor.layers)

    if align_from_end:
        print recipient.layers[l_end].output_shape[-1]
        print donor.layers[l_end - 1].output_shape[-1]
        n_tile = int(recipient.layers[l_end].output_shape[-1] / \
                    donor.layers[l_end - 1].output_shape[-1])
    else:
        n_tile = int(recipient.layers[l_start].output_shape[-1] / \
                    donor.layers[l_end].output_shape[-1])

    print n_tile

    for l in range(l_start, l_end):
        weights = []
        for w in range(len(donor.layers[l].get_weights())):
            if len(donor.layers[l].get_weights()[w].shape) == 1:
                if (donor.layers[l].name.find('batch') > -1 and \
                        donor.layers[l - 1].name.find('concat') > -1) \
                        or (donor.layers[l].name.find('act') > -1 and \
                        donor.layers[l - 2].name.find('concat') > -1):
                    wt = []
                    prev = 0
                    for i in range(512, donor.layers[l].get_weights()[w]\
                                            .shape[0] + 32, 32):
                        wt += n_tile * donor.layers[l].\
                                get_weights()[w][prev:i].tolist()
                        prev = i
                    weights.append(np.array(wt))
                else:
                    weights.append(
                        np.tile(donor.layers[l].get_weights()[w], (n_tile)))

            elif len(donor.layers[l].get_weights()[w].shape) == 2: # Dense
                print donor.layers[l].name
                if donor.layers[l - 1].name.find('pool') > -1:
                    print 'success'
                    wt = []
                    prev = 0
                    for i in range(512, donor.layers[l].\
                            get_weights()[w].shape[0] + 32, 32):
                        for b in range(n_tile):
                            wt.append(donor.layers[l].get_weights()[w][prev:i, :])
                        prev = i
                    weights.append(np.tile(
                        np.concatenate(wt, axis=0), (1,n_tile)))
                else:
                    weights.append(np.tile(
                        donor.layers[l].get_weights()[w], (n_tile,n_tile)))

            elif len(donor.layers[l].get_weights()[w].shape) == 4: # Conv2D
                if donor.layers[l - 3].name.find('concat') > -1:
                    wt = []
                    prev = 0
                    for i in range(512, donor.layers[l].\
                            get_weights()[w].shape[2] + 32, 32):
                        for b in range(n_tile):
                            wt.append(donor.layers[l].\
                                get_weights()[w][:, :, prev:i, :])
                        prev = i
                    weights.append(np.tile(
                        np.concatenate(wt, axis=2), (1,1,1,n_tile)))
                else:
                    weights.append(np.tile(
                        donor.layers[l].get_weights()[w], (1,1,n_tile,n_tile)))

        recipient.layers[l + len(recipient.layers) - len(donor.layers)].\
            set_weights(weights)

    print 'Stacked weights loaded successfully'


def gather(tensor, indices, l):
    '''
    Gather elements of 'tensor' according to 'indices'

    Args:
        tensor: Tensor to extract from
        indices: list of indices with last dimension equal to the
            rank of 'tensor'
        l: value of last dimension with which to reshape indexed tensor
    '''
    sl = tf.gather_nd(tensor, tf.cast(indices, tf.int32))
    sl = tf.reshape(sl, (-1, tf.cast(l, tf.int32)[0]))

    return sl


def partition(dim, branches=1, overlap=0):
    '''
    Calculates number of neurons in a layer that are shared/unique

    Args:
        dim: number of neurons in the layer
        branches: number of "partitions" > 0
        overlap: degree of overlap (0 to 1)

    Returns:
    Tuple (a, b):
        a: number of neurons to share
        b: number of neurons unique to each branch
    '''

    if branches <= 0:
        raise ValueError, 'b must greater than 0'

    if overlap == 0:
        return (0, int(round(float(dim) / branches)))
    else:
        x = float(dim) / (1 + branches * (1/overlap - 1))
        n = x * (1 / overlap - 1)
        return (int(round(x)), int(round(n)))


def gen_layer_mask(dim, branches=1, overlap=0, value=1):
    '''
    Generate binary mask based on neuron partition

    Args:
        See partition()
        value: number with which to multiply each non-masked neuron

    Returns:
        1D numpy array with length 'dim'
    '''

    x, n = partition(dim, branches, overlap)
    masks = []
    for b in range(0, branches - 1):
        m = np.zeros((dim,))
        m[:x] = 1
        m[x + b * n : x + (b + 1) * n] = value
        masks.append(m)

    m = np.zeros((dim,))
    m[:x] = 1
    m[x + (branches - 1) * n :] = value
    masks.append(m)

    return masks


def _get_nth_pool_layer(n, model):
    '''
    Args:
        n: nth pool layer (0 exclusive)
        model: model to search
    Returns:
        index of the nth pool layer
    '''

    n_pool = 0
    pool_layer = 0
    for l in range(len(model.layers) - 1):
        if model.layers[l + 1].name.find('pool') > -1:
            if n_pool == n:
                pool_layer = l + 1
                break
            n_pool += 1

    return pool_layer


def _compile(model, loss='triplet', P_param=1, K_param=1, mask_final=None):
    optimizer_op = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999,
                        epsilon=1e-08, decay=0.0)

    if loss == 'triplet':
        model.compile(loss=losses.triplet(
                        P_param=P_param, K_param=K_param),
                        optimizer=optimizer_op)
    elif loss == 'triplet_drop':
        model.compile(loss=losses.triplet_drop(
                        P_param=P_param, K_param=K_param,
                        masks=mask_final),
                        optimizer=optimizer_op)
    else:
        raise ValueError, "loss must either be 'triplet' or 'triplet_drop'"
