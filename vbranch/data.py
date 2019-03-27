import os
import cv2
import numpy as np

class Omniglot(object):
    def __init__(self, dir_path, flatten=True):
        self.dir_path = dir_path

        self.files = {}
        for alpha in os.listdir(dir_path):
            self.files[alpha] = {}
            characters = os.listdir(os.path.join(dir_path, alpha))
            for char in characters:
                im_dir = os.path.join(dir_path, alpha, char)
                self.files[alpha][char] = os.listdir(im_dir)

    def next(self, A, P, K, flatten=True):
        """
        Args:
            - A: number of alphabets per batch
            - P: number of characters per alphabet
            - K: numer of samples per character
            - flatten: if true, return batch with shape [None, ...],
            else return triplet with shape [A, P, K, ...]
        Returns:
            numpy array"""

        def sample(arr, n, replace=False):
            return np.random.choice(arr, n, replace=replace)

        batch = []

        a_choice = sample(list(self.files.keys()), A, False)
        for alpha in a_choice:
            p_choice = sample(list(self.files[alpha].keys()), P, False)
            for p in p_choice:
                k_choice = sample(list(self.files[alpha][p]), K, False)
                for k in k_choice:
                    im_path = os.path.join(self.dir_path, alpha, p, k)
                    # Only use first channel since in grayscale
                    batch.append(cv2.imread(im_path)[..., 0])

        batch = np.stack(batch)[..., np.newaxis]

        if not flatten:
            batch = np.reshape(batch, (A, P, K) + batch.shape[1:])

        return batch

    def get_flattened_files(self):
        index_list = []
        files_list = []

        counter = 0
        for alpha in self.files.keys():
            for char in self.files[alpha].keys():
                for name in self.files[alpha][char]:
                    index_list.append([counter, alpha, char, name])
                    files_list.append(os.path.join(self.dir_path, alpha, char, name))
                    counter += 1

        return index_list, files_list