'''
This file implements a class for preprocessesing the raw image data in SFEW
And three data partition protocols (PPI, SPI, SPS)
'''

# import libraries
import os
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import torch

class IMG_Preprocessor:
    def __init__(self, path, label_info):
        # store the label of image
        self.labels = []
        # store the file name of image
        self.files = []
        # store the movie name of image
        self.movies = []
        # store the person name of image
        self.people = []
        # store the image
        self.images = []
        for label in os.listdir(path):
            if label != ".DS_Store":
                path_f = os.path.join(path, label)
                label = label_info.index(label)
                for file in os.listdir(path_f):
                    self.labels.append(label)
                    self.files.append(file)
                    name = file.split('.')[0].split('_')
                    person = int(name[-1])
                    movie = name[0]
                    self.people.append(person)
                    self.movies.append(movie)
                    img = os.path.join(path_f, file)
                    if file.endswith(".png"):
                        im = imread(img).T
                        # im = im.flatten('K')
                        self.images.append(im)

        self.labels = np.array(self.labels)
        self.files = np.array(self.files)
        self.movies = np.array(self.movies)
        self.people = np.array(self.people)
        self.images = np.array(self.images)
        self.data_num = len(self.labels)

    # Apply PPI (Partial Person Independent) protocol
    def PPI_divide(self, msk):
        train_data = self.images[msk]
        train_target = self.labels[msk]
        test_data = self.images[~msk]
        test_target = self.labels[~msk]
        return train_data, train_target, test_data, test_target

    # Apply SPI (Strictly Person Independent) protocol
    def SPI_divide(self, training_p = 0.8):
        characters = list(self.people)
        data = self.images

        character_map = {}
        for num in set(characters):
            character_map[num] = characters.count(num)

        # print(character_map)

        # select train data from person no.1, as person no.1 contains all the emotions to ensure the train data
        # include all emotion labels
        select_threshold = int(training_p * self.data_num)
        selected = 0
        select_index = []
        for key, value in character_map.items():
            if selected < select_threshold:
                selected += value
                select_index.append(key)
            else:
                break
        for p in range(len(characters)):
            if characters[p] not in select_index:
                characters[p] = False
            else:
                characters[p] = True
        characters = np.array(characters)
        train_data = data[characters]
        test_data = data[~characters]
        train_target = self.labels[characters]
        test_target = self.labels[~characters]

        return train_data, train_target, test_data, test_target

    # Apply SPS (strictly person specific) protocol
    def SPS_divide(self, training_p=0.8):
        characters = list(self.people)
        data = np.concatenate((self.labels.reshape((-1,1)), self.images.reshape((-1,1244160))),axis=1)

        character_map = {}
        for num in set(characters):
            character_map[num] = []
        for i in range(len(characters)):
            key = characters[i]
            character_map[key].append(data[i])
        train_data = []
        test_data = []
        for key, value in character_map.items():
            num_value = len(value)
            value = np.concatenate(value).reshape(-1, 1244161)
            msk = np.random.rand(num_value) < training_p
            train_data.append(value[msk])
            test_data.append(value[~msk])

        train_data = np.concatenate(train_data).reshape(-1, 1244161)
        test_data = np.concatenate(test_data).reshape(-1, 1244161)
        train_target = train_data[:, 0].astype('long')
        test_target = test_data[:, 0].astype('long')
        train_data = train_data[:,1:].reshape(-1,3,720,576)
        test_data = test_data[:,1:].reshape(-1,3,720,576)

        return train_data, train_target, test_data, test_target
