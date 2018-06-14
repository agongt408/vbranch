# Virtual Branching (vbranch)

vbranch for person re-identification (re-ID) [arXiv paper](https://arxiv.org/pdf/1803.05872.pdf)

Implemented in Keras. Using the `ModelConfig` class (see `ModelConfig.py`), a Keras model is converted into a list of Kears layers so that new layers can be added to construct the virtual branches.

Models constructed in `models.py`.

Implementation for training on Market1501, CUHK03, and DukeMTMC-reID provided (`data.py`).

[Batch-hard]((https://arxiv.org/pdf/1703.07737.pdf)) variant of triplet loss used as loss function (`losses.py`).

Only DenseNet implemenation so far (`dense.py`).

## Requirements
Python 2.7 <br>
Tensorflow 1.3.0 <br>
Keras 2.0.8 <br>

### Notes
Make the following changes to the Keras and Tensorflow source code:

Kernel regularizer (Conv2D): Declare regularizers externally to have reference handles to regularizer objects and include indices (idx) arg

    - keras/regularizers.py
        __init__()
            mod: Added indices property to only apply regularizer to
            parameters being trained (line 35)
        __call__()
            mod: gather elements corresponding to indices (line 48)
        get_config()
            mod: update config dict to show indices (line 61)

Batch normalization: Do not apply moving average update to masked parameters: estimated mean, estimated variance

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
