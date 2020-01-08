import tensorflow as tf
import numpy as np

def load_data(one_hot=True, preprocess=False, 
        random_crop=False, random_flip=True, multiple=3):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # print('Mean:', np.mean(X_train/255. - 1, axis=(0,1,2)))
    # print('Std:', np.std(X_train/255. - 1, axis=(0,1,2)))

    if preprocess:
        X_train = preprocess_im(X_train)
        X_test = preprocess_im(X_test)

    if random_crop:
        augmented_images = []
        augmented_labels = []
        padding, imsize = 2, 32
        
        pad_spec = ((0,0), (padding, padding), (padding, padding), (0,0))
        padded_images = np.pad(X_train, pad_spec, 'constant', constant_values=0)

        for im, lb in zip(padded_images, y_train):
            for _ in range(multiple):
                crop_x = np.random.randint(2*padding)
                crop_y = np.random.randint(2*padding)
                cropped_im = im[crop_y:imsize+crop_y, crop_x:imsize+crop_x]
                
                if np.random.random() < 0.5:
                    cropped_im = np.flip(cropped_im, axis=1)
                    
                augmented_images.append(cropped_im)
                augmented_labels.append(lb)

        X_train = augmented_images
        y_train = augmented_labels

    if random_flip:
        for i in range(len(X_train)):
            if np.random.random() < 0.5:
                X_train[i] = np.flip(X_train[i], axis=1)
                
    if one_hot:
        num_classes = 10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)

def preprocess_im(im):
    im = im / 255

    im[..., 0] -= 0.4914
    im[..., 1] -= 0.4822
    im[..., 2] -= 0.4465

    im[..., 0] /= 0.2023
    im[..., 1] /= 0.1994
    im[..., 2] /= 0.2010
    return im

class DataGeneratorTrain(object):
    def __init__(self, batch_size=32,
            one_hot=True, preprocess=False,
            flip=False, padding=0, im_size=32):
        (self.X_train, self.y_train), _ = load_data(one_hot, preprocess)

        pad_spec = ((0,0), (padding, padding), (padding, padding), (0,0))
        self.X_train = np.pad(self.X_train, pad_spec, 'constant', constant_values=0)

        self.batch_size = batch_size
        self.flip = flip
        self.padding = padding
        self.im_size = im_size
        self.steps_per_epoch = len(self.X_train) // batch_size
        self.it = 0

    def __next__(self):
        if self.it == 0:
            total_samples = len(self.X_train)
            shuffle_choice = np.random.choice(total_samples, total_samples, replace=False)
            self.X_train = self.X_train[shuffle_choice]
            self.y_train = self.y_train[shuffle_choice]

        padded_batch = self.X_train[self.it*self.batch_size:(self.it+1)*self.batch_size]
        labels = self.y_train[self.it*self.batch_size:(self.it+1)*self.batch_size]

        if self.padding > 0:
            batch = []
            for im in padded_batch:
                crop_x = np.random.randint(2*self.padding)
                crop_y = np.random.randint(2*self.padding)
                im = im[crop_y:self.im_size+crop_y, crop_x:self.im_size+crop_x]
                if self.flip and np.random.random() < 0.5:
                    im = np.flip(im, axis=1)
                batch.append(im)
        else:
            batch = padded_batch

        self.it = (self.it + 1) % self.steps_per_epoch
        return np.stack(batch), np.array(labels)

    def get_steps_per_epoch(self):
        return self.steps_per_epoch
