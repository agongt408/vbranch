# Virtual Branching

- Simple ensemble
- Multiple branches (same data)
- Multiple branches (different data, equally sized partitions)
- Multiple branches (different data, differently sized partitions)

Datasets:
- MNIST
- Toy classification dataset
- Omniglot
- Person Re-ID (future)

## Results

### MNIST

![fcn-results](figs/fcn-2-results.png)
![fcn-results](figs/fcn-3-results.png)
![fcn-results](figs/fcn-4-results.png)

FCN = 784 (input) -> 512 -> 10 (output); batch norm, relu; softmax; 15 epochs;
learning rate = 0.001; test = before mean acc

![fcn2-results](figs/fcn2-2-results.png)
![fcn2-results](figs/fcn2-3-results.png)
![fcn2-results](figs/fcn2-4-results.png)

FCN = 784 (input) -> 512 -> 256 -> 10 (output); batch norm, relu; softmax; 15 epochs;
learning rate = 0.001; test = before mean acc

** No batch norm or relu for final FC layer (output layer); improves performance for shared_frac=1; for figures with batch norm/relu after final FC layer, see [old-fcn](old/figs)

![cnn-results](old/figs/cnn-2-results.png)
![cnn-results](old/figs/cnn-3-results.png)
![cnn-results](old/figs/cnn-4-results.png)

CNN = 1 (input) -> 16 -> 16 -> 32 -> 32 filters; batch norm, relu; softmax; 30 epochs;
learning rate = 0.001; test = before mean acc

Training graph:
![mnist-val-acc](figs/mnist-val-acc.png)

CNN = 1 (input) -> 16 -> 16 -> 32 -> 32 filters; batch norm, relu; softmax; 30 epochs;
learning rate = 0.001; test = before mean acc

Converges by around epoch 30

### Omniglot

![omniglot-results](figs/omniglot-simple-2-concat-results.png)
![omniglot-results](figs/omniglot-simple-3-concat-results.png)
![omniglot-results](figs/omniglot-simple-4-concat-results.png)

CNN = 1 (input) -> 32 -> 32 -> 64 -> 64 -> 128 -> 128 -> 256 -> 256 filters; batch norm, relu; softmax; 90 epochs;
learning rate = 0.001; concatenate embeddings

### Toy

[toy-classification-results](toy-classification.ipynb)

## Notes

Theory:
- Hypothesis: reduce variance (see Bagging, Random Forest)

<!-- Why does vbranch result in worse performance for FCN and shared fraction equals 1?
- Hypothesis: too large of a batch size can sometimes lead to worse performance, especially for small models (i.e., FCN) -->

Graph performance vs. number of parameters (direct instead of using shared_frac)

Can we reduce batch size of individual branches when using vbranch and achieve same performance improvements with lower computation?

### Related Papers
http://openaccess.thecvf.com/content_cvpr_2018/papers/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.pdf

https://arxiv.org/pdf/1511.02954.pdf

http://leemon.com/papers/2009scsb.pdf

Net2Net:
https://arxiv.org/pdf/1511.05641.pdf

TF QueueRunners:
https://adventuresinmachinelearning.com/introduction-tensorflow-queuing/

https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0

TF Dataset API:
https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

Bagging (with derivations):
https://www.stat.berkeley.edu/~breiman/bagging.pdf

Random Forest:
https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
