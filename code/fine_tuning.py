import os
import random as rn
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras import backend as K
from transfer_learning import Transferlearning
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam
from constants import BATCH_SIZE_TF, EPISODES_COMB, LEARNING_RATE_TF

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

class FineTunnig(Transferlearning):
    """
    FineTuning class for transfer learning.

    This class fine-tunes a pre-trained model on a new dataset.

    Parameters:
    - path_to_model (str): Path to the pre-trained model.
    - save_path (str): Directory where the fine-tuned model will be saved.
    - data_path (str): Path to the dataset used for fine-tuning.
    - new_labels (list): List of new class labels for the model.
    - user (str): Identifier for the user initiating the fine-tuning process.
    - episodes (int): Number of training iterations (epochs).
    - batch_size (int): Number of samples per training batch.
    - learning_rate (int): Learning rate for optimization.
    """
    
    def __init__(self, 
                 path_to_model:str, 
                 save_path:str, 
                 data_path:str, 
                 new_labels:list, 
                 user:str, 
                 episodes:int,
                 batch_size:int,
                 learning_rate:int,
                ):
        
      

        
        super().__init__(path_to_model, save_path, data_path, new_labels, user, episodes, batch_size, learning_rate)
        
    
    def load_model(self):
        # Fetch the model
        self.model = load_model(os.path.join(self.path_to_model, 'model_best.h5'))
        print(f"Loading model from: {self.path_to_model}")
                
        # Change the output layer
        self.model.pop()  # Remove the last layer
        self.model.add(Dense(len(self.swimming_data.labels), activation=self.activation_type, name="last_dense"))  # Add new output layer 
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])
        
        print(f"Total trainable parameters: {total_params}")
        print(f"Last layer changed to: {len(self.swimming_data.labels)} outputs")

    def train(self):
        history = None
        if self.model is None:
            raise ValueError("Model is not loaded, Do you forget to call load_model() method?")
        
        # Random seed stuff. Maybe this is overkill
        os.environ['PYTHONHASHSEED'] = '0'
        rn.seed(1337)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        np.random.seed(1337)
        tf.random.set_seed(1337)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        K.set_session(sess)

        # Path for saving results
        print("Running experiment: %s" % self.user)

        # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
        # lists of names for each label
        train_dict, val_dict = self.swimming_data.draw_train_val_dicts(self.swimming_data.users,validation=False)

        # The generator used to draw mini-batches
        gen = self.swimming_data.batch_generator_dicts(train_dict=train_dict,
                                                    batch_size=self.training_parameters['batch_size'],
                                                    noise_std=self.training_parameters['noise_std'],
                                                    mirror_prob=self.training_parameters['mirror_prob'],
                                                    random_rot_deg=self.training_parameters['random_rot_deg'])

        # Optimizer
        optimizer = Adam(learning_rate=self.training_parameters['lr'], beta_1=self.training_parameters['beta_1'],
                                            beta_2=self.training_parameters['beta_2'])

        # Path to the "best" model w.r.t. the validation accuracy
        best_path = os.path.join(self.save_path, 'model_best.h5')

        # Which model is the best model and where we save it
        checkpoint = ModelCheckpoint(best_path, verbose=1, monitor='val_weighted_acc',
                                                        save_best_only=True, mode='max')
        self.model.compile(optimizer=optimizer, loss=self.loss_type, metrics=['acc'], weighted_metrics=['acc'])

        # Train the model
        history = self.model.fit(gen, 
            epochs=self.training_parameters['max_epochs'],
            steps_per_epoch=self.training_parameters['steps_per_epoch'],
            callbacks=[checkpoint])
            
                
        self.model.save(os.path.join(self.save_path, 'model_best.h5'))
            
        # Saving the history and parameters
        with open(os.path.join(self.save_path, 'train_val_dicts.pkl'), 'wb') as f:
            pickle.dump([train_dict, val_dict], f)
        if history: 
            with open(os.path.join(self.save_path, 'history.pkl'), 'wb') as f:
                pickle.dump([history.history], f) 
        with open(os.path.join(self.save_path, 'data_parameters.pkl'), 'wb') as f:
            pickle.dump([self.data_parameters], f)
        with open(os.path.join(self.save_path, 'training_parameters.pkl'), 'wb') as f:
            pickle.dump([self.training_parameters], f)

        # Related to seed stuff
        K.clear_session()
        
        


if __name__ == '__main__':
  
    # Change this variable to define the new labels to train
    new_labels = [10] 

    # A path to re-sampled recordings which are organized into folders by user name.
    data_path = r'data_to_use'

    # Path of the model base model
    path_to_model = r"original"
    
    user = "circle_train"

    transfer = FineTunnig(
        path_to_model=path_to_model, 
        save_path=rf"Master\save_path\finetuning_{user}", 
        data_path = data_path, 
        new_labels= new_labels, 
        user=user,
        episodes=35, 
        batch_size=64,
        learning_rate=0.0005
        )
    
    transfer.load_model()
    transfer.train()

   