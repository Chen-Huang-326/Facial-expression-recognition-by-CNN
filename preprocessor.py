'''
This file aims at preprocessing the facial emotion data; The facial emotion data has 7 classes with 675 lines
and the image information is extracted with the first 5 PCA components by LQP and PHOG descriptor
The label of class is the column 2 and means:
   1 Angry
   2 Disgust
   3 Fear
   4 Happy
   5 Neutral
   6 Sad
   7 Surprise
'''

# import libraries
import pandas as pd
import numpy as np
import torch.utils.data


# Hyper parameters

# training_p = 0.8
# validation_p = 0
# testing_p = 0.5

# name of column index
# col_index = ['overview', 'emotion_label', 'LQP_00', 'LQP_01', 'LQP_02', 'LQP_03', 'LQP_04', 'LQP_05', 'PHOG_01', 'PHOG_02', 'PHOG_03', 'PHOG_04', 'PHOG_05']

'''
This class is a preprocessor to load data and split data in SFEW.xlsx (the CPA of images)
There are thress protocols:
    SPI: Strictly Person Independent protocol
    PPI: Partial Person Independent protocol
    SPS: Strictly Person Specific protocol
to partition data into training dataset and testing dataset
'''
class Preprocessor:
    def __init__(self):
        # load all data from SFEW.xlsx
        self.data = pd.read_excel('data/SFEW.xlsx', sheet_name='SFEW.csv')
        # The number of data points
        self.data_num = self.data.shape[0]

    # method to get the person information
    def GetPerson(self, names, training_p):
        person_list = []
        for name in names:
            info = name.split("_")
            person_info = info[-1]
            person = int(person_info.split(".")[0])
            person_list.append(person)

        person_map = {}
        for num in set(person_list):
            person_map[num] = person_list.count(num)

        # select train data from person no.1, as person no.1 contains all the emotions to ensure the train data
        # include all emotion labels
        select_threshold = int(training_p * self.data_num)
        selected = 0
        select_index = []
        for key, value in person_map.items():
            if selected < select_threshold:
                selected += value
                select_index.append(key)
            else:
                break
        for p in range(len(person_list)):
            if person_list[p] not in select_index:
                person_list[p] = False
            else:
                person_list[p] = True
        person_list = np.array(person_list)

        return person_list


    # Apply SPI (Strictly Person Independent) protocol
    def SPI_partition(self, data, training_p = 0.8):
        '''
        Transform the value of first column to categorical value
        i.e. 'Airheads_000519240_00000005.mat', the last information 00000005 is the unique identifier of the person in the image
        So, the corresponding numerical value of it is 5
        '''
        names = data['Unnamed: 0']
        data = self.Normalization(data)
        person_list = []
        for name in names:
            info = name.split("_")
            person_info = info[-1]
            person = int(person_info.split(".")[0])
            person_list.append(person)

        person_map = {}
        for num in set(person_list):
            person_map[num] = person_list.count(num)

        # select train data from person no.1, as person no.1 contains all the emotions to ensure the train data
        # include all emotion labels
        select_threshold = int(training_p * self.data_num)
        selected = 0
        select_index = []
        for key, value in person_map.items():
            if selected < select_threshold:
                selected += value
                select_index.append(key)
            else:
                break
        for p in range(len(person_list)):
            if person_list[p] not in select_index:
                person_list[p] = False
            else:
                person_list[p] = True
        person_list = np.array(person_list)
        train_data = data[person_list]
        test_data = data[~person_list]

        return train_data, test_data


    # Apply PPI (Partial Person Independent) protocol
    def PPI_partition(self, data, msk):
        data = self.Normalization(data)
        train_data = data[msk]
        test_data = data[~msk]
        return train_data, test_data

    # Apply SPS (strictly person specific) protocol
    def SPS_partition(self, data, training_p=0.8):
        names = data['Unnamed: 0']
        data = self.Normalization(data)
        person_list = []
        for name in names:
            info = name.split("_")
            person_info = info[-1]
            person = int(person_info.split(".")[0])
            person_list.append(person)

        person_map = {}
        for num in set(person_list):
            person_map[num] = []
        for i in range(len(person_list)):
            key = person_list[i]
            person_map[key].append(data.iloc[i].values)
        train_data = []
        test_data = []
        for key, value in person_map.items():
            num_value = len(value)
            value = pd.DataFrame(np.concatenate(value).reshape(-1, 11))
            msk = np.random.rand(num_value) < training_p
            train_data.append(value[msk])
            test_data.append(value[~msk])

        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)

        return train_data, test_data


    # Normalize the data
    def Normalization (self, data):
        # drop the first column
        data.drop(data.columns[0], axis=1, inplace=True)
        # convert na/nan value to 0
        data = data.fillna(0)
        # normalize the data except the target (the first column is target)
        for column in data:
            if column != 'label':
                data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

        return data



# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][1:]
        target = self.data_tensor[index][0] - 1

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n
