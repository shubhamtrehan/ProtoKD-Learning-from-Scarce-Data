import numpy as np
import json, os, sys, pickle, cv2, random
from random import shuffle
import tensorflow as tf
import imgaug.augmenters as ia
from itertools import product
from copy import deepcopy
import cv2
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class OvaDataLoader():
    """
    DataLoader for the Ova dataset, designed for few-shot learning tasks.

    This class handles loading, preprocessing, and augmentation of image data,
    and provides methods to generate episodic data for prototypical networks.
    """
    def __init__(self, imgPath, img_size=128, numLabeled=None, randomTrain=False):
        """
        Initializes the OvaDataLoader.

        Args:
            imgPath (str): Path to the directory containing image data.
            img_size (int, optional): Size to which images will be resized. Defaults to 128.
            numLabeled (int, optional): Number of labeled samples per class for training.
                If None, all samples are considered labeled. Defaults to None.
            randomTrain (bool, optional): Whether to shuffle the training data. Defaults to False.
        """
        self.numLabeled = numLabeled
        self.imgPath = imgPath
        self.img_size = img_size

        # New Augmentation Scheme
        self.augment = tf.keras.Sequential(
            [
                layers.RandomContrast(0.3),
                layers.RandomFlip("vertical", input_shape=(img_size, img_size, 3)),
                layers.RandomRotation(0.8),
                layers.RandomZoom(0.4),
                layers.RandomContrast(0.5),
            ]
        )

        self.train_labeled = {}
        self.train_unlabeled = {}
        self.test = {}
        self.classes = self.loadClasses("./Files/classes.txt")

        files = os.listdir(imgPath)
        files.remove("Background")
        data = {}

        for className in files:
            if className.lower() not in data:
                data[self.classes.index(className.lower())] = []
            filePath = os.path.join(imgPath, className)
            img_files = os.listdir(filePath)
            for imgFile in img_files:
                if '.png' not in imgFile:
                    continue
                img = cv2.imread(os.path.join(filePath, imgFile))
                img = cv2.resize(img, (img_size, img_size))
                img = tf.keras.applications.resnet50.preprocess_input(img)
                img = img.astype(np.float32) / 255.0
                data[self.classes.index(className.lower())].append(img)

        for class_ind, imgList in data.items():
            if randomTrain:
                shuffle(imgList)
            self.train_labeled[class_ind] = imgList[:self.numLabeled]
            self.train_unlabeled[class_ind] = []
            self.test[class_ind] = imgList[self.numLabeled:]

        print("Dataset Statistics:")
        print("Class ID \t Num Labeled \t Num Unlabeled \t Num Test")
        for class_ind in self.train_labeled.keys():
            print(class_ind, "\t \t", len(self.train_labeled[class_ind]), "\t \t",
                  len(self.train_unlabeled[class_ind]), "\t \t", len(self.test[class_ind]))

    def loadClasses(self, fileName):
        """
        Loads class names from a file.

        Args:
            fileName (str): Path to the file containing class names, one per line.

        Returns:
            list: List of class names.
        """
        classes = []
        with open(fileName, 'r') as inFile:
            for line in inFile:
                line = line.replace('\n', '').lower()
                classes.append(line)
        return classes

    def getProtoEpisode(self, n_support=5, n_query=5):
        """
        Generates a prototypical network episode.

        This method creates support and query sets for a few-shot learning episode,
        with optional augmentation applied to the query set.

        Args:
            n_support (int, optional): Number of support samples per class. Defaults to 5.
            n_query (int, optional): Number of query samples per class. Defaults to 5.

        Returns:
            tuple: A tuple containing:
                - support (tf.Tensor): Support set tensor. Shape: (n_way, n_support, img_size, img_size, 3).
                - query (tf.Tensor): Query set tensor. Shape: (n_way, n_query, img_size, img_size, 3).
                - sup_labels (np.ndarray): Support set labels. Shape: (n_way * n_support,).
                - qry_labels (np.ndarray): Query set labels. Shape: (n_way * n_query,).
                - qry_rel_labels (np.ndarray): Query set relative labels. Shape: (n_way * n_query,).
        """
        qry_labels = []
        sup_labels = []
        qry_rel_labels = []

        support = []
        query = []

        for cnt, idx in enumerate(sorted(self.train_labeled.keys())):
            idx = int(idx)
            imgs = deepcopy(self.train_labeled[idx])
            if len(imgs) < n_support + n_query:
                n_repeat = np.ceil((n_support + n_query) / len(imgs)).astype(int)
                imgs *= n_repeat
            shuffle(imgs)
            sup_imgs = imgs[:n_support]
            qry_imgs = imgs[n_support:n_query + n_support]
            qry_labels += [idx] * n_query
            sup_labels += [idx] * n_support
            qry_rel_labels += [cnt] * n_query

            # Convert to Tensor and Apply Augmentation
            sup_imgs = tf.convert_to_tensor(np.stack(sup_imgs))
            qry_imgs = tf.convert_to_tensor(np.stack(qry_imgs))
            qry_imgs = self.augment(qry_imgs)#.numpy()  # Apply augmentation on the query set

            support.append(sup_imgs)#.numpy())
            query.append(qry_imgs)

        support = tf.stack(support)
        query = tf.stack(query)

        return support, query, np.array(sup_labels).astype(np.int32), np.array(qry_labels).astype(np.int32), np.array(qry_rel_labels).astype(np.int32)


