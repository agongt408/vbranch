# Virtual Branching (vbranch)

vbranch for person re-identification (re-ID)

Implemented in Keras. Using the `ModelConfig` class (see `ListModel.py`), a Keras model is converted into a list of Kears layers so that new layers can be added to construct the virtual branches.

Models constructed in `models.py`.

Implementation for training on Market1501, CUHK03, and DukeMTMC-reID provided (`data.py`).

[Batch-hard]((https://arxiv.org/pdf/1703.07737.pdf)) variant of triplet loss used as loss function (`losses.py`).

Only DenseNet implemenation so far (`dense.py`).
