import learning_data
import os
import random as rn
import keras
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.keras import backend as K
import keras
from abc import ABC, abstractmethod
import cnn_vanilla
from sklearn.metrics import accuracy_score
from constants import LEARNING_RATE_TF, BATCH_SIZE_TF

class Transferlearning(ABC):
    """
    The TransferLearning class serves as an abstract base class (ABC) for implementing various transfer learning strategies. 
    It provides a foundation for fine-tuning, freezing, and progressive learning by handling model paths, dataset information, and training configurations.
    """
 
    
    def __init__(self, path_to_model:str, save_path:str, data_path:str, new_labels:list, user:str, episodes:int, batch_size: int, learning_rate = None, augment:bool = True):
        self.path_to_model = path_to_model
        self.save_path = save_path
        self.new_labels = new_labels
        self.user = user
        self.data_path = data_path
        self.episodes = episodes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.data_parameters = None
        self.training_parameters = None
        self.loss_type = None
        self.activation_type = None
        self.swimming_data = None
        self.model = None
        self.augment = augment
        
        self._init_variables()
        self._load_data()
     
     
    @abstractmethod   
    def load_model(self):
        pass	
    
    @ abstractmethod         
    def train(self):
        pass


    def _init_variables(self):
        # Get the training parameters used for loading
        defualt_training_parameters = cnn_vanilla.get_default_training_parameters()
        
        episodes = self.episodes if self.episodes is not None else defualt_training_parameters["max_epochs"]
        with open(os.path.join(self.path_to_model, 'data_parameters.pkl'), 'rb') as f:
            self.data_parameters = pickle.load(f)[0]

        # Get the data parameters used for loading
        with open(os.path.join(self.path_to_model, 'training_parameters.pkl'), 'rb') as f:
            self.training_parameters = pickle.load(f)[0]
            self.training_parameters['max_epochs'] = episodes
            self.training_parameters["epochs"] = episodes
            self.training_parameters["steps_per_epoch"] = episodes
            self.training_parameters['batch_size'] = self.batch_size if self.batch_size is not None else defualt_training_parameters["batch_size"]
            self.training_parameters['lr'] = self.learning_rate if self.learning_rate is not None else defualt_training_parameters["lr"]

        print(f"Learning rate: {self.training_parameters['lr']}")
        print(f"Batch size: {self.training_parameters['batch_size']}")
        
        # Update the data parameters
        for label in self.new_labels:
            if label not in self.data_parameters['labels']:
                self.data_parameters['labels'].append(label)
            
        self.data_parameters["users"] = [self.user]                     
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
    def _load_data(self):
        # Load the recordings
        self.swimming_data = learning_data.LearningData()
        self.swimming_data.load_data(data_path=self.data_path,
                                data_columns=self.data_parameters['data_columns'],
                                labels=self.data_parameters['labels'], 
                                users=[self.user]
                                )

        # Combine labels
        if self.data_parameters['combine_labels'] is not None:
            for label in self.data_parameters['combine_labels'].keys():
                self.swimming_data.combine_labels(labels=self.data_parameters['combine_labels'][label], new_label=label)


        # Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
        # Stored under swimming_data['time_scaled_1.1'][user_name]...
        if self.augment:
            self.swimming_data.augment_recordings(time_scale_factors=self.data_parameters['time_scale_factors'])

        self.swimming_data.sliding_window_locs(win_len=self.data_parameters['win_len'], slide_len=self.data_parameters['slide_len'])

        #Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'label']
        # Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
        self.swimming_data.compile_windows(norm_type=self.data_parameters['window_normalization'],
                                    label_type=self.data_parameters['label_type'],
                                    majority_thresh=self.data_parameters['majority_thresh'])

        # Set loss_type  and activation function
        if len(self.swimming_data.labels) == 2:
            self.loss_type = 'binary_crossentropy'
            self.activation_type = 'sigmoid'
        else:
            self.loss_type = 'categorical_crossentropy'
            self.activation_type = 'softmax'

