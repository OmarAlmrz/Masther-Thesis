
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
from constants import LEARNING_RATE_TF, BATCH_SIZE_TF, EPISODES_COMB

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

class Freezing(Transferlearning):
    
    """
        The Freezing class is used for transfer learning with selective freezing of layers. 
        It allows freezing a specified number of layers in a pre-trained model and optionally enables gradual freezing over multiple episodes.

        Parameters:
        -path_to_model (str): Path to the pre-trained model.
        -save_path (str): Directory where the fine-tuned model will be saved.
        -data_path (str): Path to the dataset used for training.
        -new_labels (list): List of new class labels.
        -user (str): Identifier for the user performing the freezing operation.
        -freezing_layers (int, default=0): Number of layers to freeze in the model.
        -episodes (int, default=20): Number of training iterations (epochs).
        -gradual_freezing (dict, default=None): A dictionary specifying gradual freezing strategy over episodes.
        -batch_size (int, default=64): Number of samples per training batch.
    """
    
    def __init__(self, 
                 path_to_model:str, 
                 save_path:str, 
                 data_path:str, 
                 new_labels:list, 
                 user:str, 
                 freezing_layers:int = 0, 
                 episodes:int = 20,
                 gradual_freezing: dict = None,
                 batch_size:int = 64,
                ):
        
        
        self.freezing_layers = freezing_layers
        self.gradual_freezing  = gradual_freezing
        self.gradual_f_episodes = self.calculate_gradual_freezing_ep(gradual_freezing, episodes)
        
        super().__init__(path_to_model, save_path, data_path, new_labels, user, episodes, batch_size)
        
    
    def calculate_gradual_freezing_ep(self, gradual_freezing:dict, total_episodes):
        if gradual_freezing is None : return None 
        gradual_ep = {}
        for key, value in gradual_freezing.items():
            gradual_ep[key] = round(total_episodes*value)
        return gradual_ep
        
        
    def load_model(self):
        # Fetch the model
        self.model = load_model(os.path.join(self.path_to_model, 'model_best.h5'))
        print(f"Loading model from: {self.path_to_model}")
        # Freeze the layers
        if self.freezing_layers and self.gradual_freezing == None:
            print("Freezing the first %d layers" % self.freezing_layers)
            for i in range(self.freezing_layers):
                self.model.layers[i].trainable = False
        
        # Freeze all layers
        elif self.gradual_freezing: 
            for layer in self.model.layers:
                layer.trainable = False
            
            # Unfreeze the last layer
            self.model.layers[-1].trainable = True
                
        # Change the output layer
        self.model.pop()  # Remove the last layer
        self.model.add(Dense(len(self.swimming_data.labels), activation=self.activation_type, name="last_dense"))  # Add new output layer 
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])
        print(f"Total trainable parameters: {total_params}")
        print(f"Last layer changed to: {len(self.swimming_data.labels)} outputs")

    # Function to unfreeze layers gradually
    def unfreeze_layers(self, freeze_layers:int ):
        if freeze_layers <= 0 : return
        layers_to_unfreeze = self.model.layers[freeze_layers:]
        for layer in layers_to_unfreeze:
            layer.trainable = True
        
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])           
        print(f"Total trainable parameters: {total_params}\n")
        
        
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

        
        if self.gradual_freezing is None: 
            # Train the model
            history = self.model.fit(gen, 
                                    epochs=self.training_parameters['max_epochs'],
                                    steps_per_epoch=self.training_parameters['steps_per_epoch'],
                                    callbacks=[checkpoint])
            
        # Train model with gradula freezing
        else: 
            print("Training with gradual freezing....\n")
            for epoch in range(self.training_parameters['max_epochs']):
                freezing = 0
                print(f"Epoch {epoch + 1}/{self.training_parameters['max_epochs']}")
                # Train the model
                self.model.fit(gen,steps_per_epoch=self.training_parameters['steps_per_epoch'],callbacks=[checkpoint], epochs=1)
                
                for key, val in self.gradual_f_episodes.items():
                    if epoch > val: freezing = key
                
                self.unfreeze_layers(freezing)
                
                
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
    freezing_comb = [18,13,9]
    
    # Change this variable to define the new labels to train
    new_labels = [6,7,8,9] 

    # A path to re-sampled recordings which are organized into folders by user name.
    data_path = 'data_to_use'

    # Path of the model base model
    path_to_model = "original"
    
    user = "train_all_replay"

    ##############################################
    # Best configuration
    # ep = 35
    # gradual_freezing = {13: .5, 9 : .75}
    # for idx, user in enumerate(styles_conb):
    #     save_path = os.path.join("Master\save_path", user)
    #     transfer = Freezing(
    #             path_to_model=r"Master\save_path\original",
    #             save_path=save_path, 
    #             data_path = data_path, 
    #             new_labels= new_labels, 
    #             user=user,
    #             freezing_layers=None,
    #             gradual_freezing=gradual_freezing,
    #             episodes=ep,
    #             )
            
    #     transfer.load_model()
    #     transfer.train()
    ##############################################

    #Gradual freezing consist of "freeezing layers": "progress"
    gradual_freezing = [{13: .75, 9 : .875}]
    
    for gradual_conf in gradual_freezing:
        for batch in BATCH_SIZE_TF:
            for ep in EPISODES_COMB:
                # Convert the dictionary to a string and replace or remove invalid characters
                dict_string = str(gradual_conf).replace(" ", "").replace("{", "").replace("}", "").replace(":", "-").replace(",", "_")
                save_path = rf"Master\save_path\freezing_{user}\{dict_string}_{ep}ep_{batch}bt"
        
                transfer = Freezing(
                    path_to_model=path_to_model, 
                    save_path=save_path, 
                    data_path = data_path, 
                    new_labels= new_labels, 
                    user=user,
                    freezing_layers=None,
                    gradual_freezing=gradual_conf,
                    episodes=ep,
                    batch_size=batch
                    )
                
                transfer.load_model()
                transfer.train()


        