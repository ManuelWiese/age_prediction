import cv2
import os
import random
import numpy as np
from collections import deque
import threading
import time

class Dataset:
    def __init__(self, path, size, batch_size, resampling=True, train_size=None, test_size=None, validate_size=None):
        self.path = path
        self.size = size
        self.batch_size = batch_size
        self.num_classes = 100
        self.resampling = resampling

        self.num_threads = 8
        self.next_batches = deque()
        self.max_next_batches_length = 100

        self.index = 0
        self.images_seen = 0
        self.epoch = 0

        self.train_filenames = self.get_image_filenames("train", train_size)
        
        self.test_filenames = self.get_image_filenames("test", test_size)
        self.validate_filenames = self.get_image_filenames("validate", validate_size)

        self.test = self.load_images(self.test_filenames)
        self.validate = self.load_images(self.validate_filenames)

    def get_image_filenames(self, subfolder, max_size=None):
        filenames = os.listdir("{}/{}".format(self.path, subfolder))

        if max_size is not None:
            filenames = filenames[:max_size]

        filenames = ["{}/{}/{}".format(self.path, subfolder, filename) for filename in filenames]

        return filenames

    def load_images(self, filenames):
        x = []
        y = []

        for filename in filenames:
            try:
                age = int(filename.split('-')[1].strip(".jpg"))
                tmp_y = self.create_one_hot(age)

                tmp_x = self.read_image(filename)
            except:
                continue
            x.append(tmp_x)
            y.append(tmp_y)

        return list(zip(x, y))

    def create_one_hot(self, age):
        one_hot = np.zeros(self.num_classes)
        one_hot[age] = 1.0

        return one_hot

    def read_image(self, filename):
        image = cv2.imread(filename, 0)
        image = cv2.resize(image, (self.size, self.size))
        image = image / 255
        image = image.reshape((self.size, self.size, 1))
        return image

    def get_next_batch(self):
        if len(self.next_batches) == 0:
            self.load_next_batches()
            time.sleep(1)
            return self.get_next_batch()
        else:
            batch = self.next_batches.popleft()
            self.load_next_batches()

            self.images_seen += len(batch)
            self.epoch = self.images_seen // len(self.train_filenames)

            return list(zip(*batch))

    def load_next_batches(self):
        num_threads = min(self.max_next_batches_length - len(self.next_batches), self.num_threads)
        num_threads -= threading.active_count() - 1
    
        if num_threads <= 0:
            return

        for _ in range(num_threads):        
            thread = threading.Thread(target=self.load_batch, args=(self.train_filenames[self.index : self.index + self.batch_size], ))
            thread.daemon = True
            thread.start()

            self.index += self.batch_size
            if self.index >= len(self.train_filenames):
                self.index = 0
                random.shuffle(self.train_filenames)
        
    def load_batch(self, filenames):
        batch = self.load_images(filenames)

        if self.resampling:
            batch = self.resample_batch(batch)

        self.next_batches.append(batch)

    def get_test_set(self):
        return list(zip(*self.test))

    def get_validate_set(self):
        return list(zip(*self.validate))

    def resample_batch(self, batch):
        images, ages = list(zip(*batch))

        images = list(images)
        ages = list(ages)

        for i in range(len(images)):
            images[i] = images[i] + np.random.normal(0.1, 0.1, images[i].shape)

            rows, cols, _ = images[i].shape

            M = cv2.getRotationMatrix2D((cols/2, rows/2), 20*random.random() - 10, 1)
            images[i] = cv2.warpAffine(images[i], M, (cols, rows)).reshape((self.size, self.size, 1))

        return list(zip(images, ages))


if __name__ == "__main__":
    dataset = Dataset("./cleaned_data", 256)
    print(dataset.get_next_batch(12))