import os
import cv2
import numpy as np

class Omniglot(object):

    def __init__(self, dir_path, A, P, K, flatten=True):
        self.dir_path = dir_path
        self.A = A # number of alphabets per batch
        self.P = P # number of characters per alphabet
        self.K = K # numer of samples per character

        # if true, return batch with shape [None, ...]
        # if false, return triplet with shape [A, P, K, ...]
        self.flatten = flatten

        self.files = {}
        for alpha in os.listdir(dir_path):
            self.files[alpha] = {}
            characters = os.listdir(os.path.join(dir_path, alpha))
            for char in characters:
                im_dir = os.path.join(dir_path, alpha, char)
                self.files[alpha][char] = os.listdir(im_dir)

    def __iter__(self):
        return self

    def __next__(self):
        def sample(arr, n, replace=False):
            return np.random.choice(arr, n, replace=replace)

        batch = []

        a_choice = sample(list(self.files.keys()), self.A, False)
        for alpha in a_choice:
            p_choice = sample(list(self.files[alpha].keys()), self.P, False)
            for p in p_choice:
                k_choice = sample(list(self.files[alpha][p]), self.K, False)
                for k in k_choice:
                    im_path = os.path.join(self.dir_path, alpha, p, k)
                    # Only use first channel since in grayscale
                    batch.append(cv2.imread(im_path)[..., 0])

        batch = np.stack(batch)

        if not self.flatten:
            batch = np.reshape(batch, (self.A, self.P, self.K) + batch.shape[1:])

        return batch
