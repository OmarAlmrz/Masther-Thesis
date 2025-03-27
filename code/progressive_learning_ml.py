
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from transfer_learning import Transferlearning
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

class Progressive_ML(Transferlearning):
    
    """
        The Progressive_ML class extends transfer learning by progressively modifying the model structure. 
        It loads a pre-trained model, freezes initial layers, removes the last layer and trains a classifier on the latent space.
        
        Parameters:
            -path_to_model (str): Path to the pre-trained model.
            -save_path (str): Directory where the modified model will be saved.
            -data_path (str): Path to the dataset used for training.
            -new_labels (list): List of new class labels.
            -user (str): Identifier for the user modifying the model.
            -episodes (int, default=20): Number of training iterations (epochs).
            -classifier (object): Classifier to train on the latent space.
            -classifier_name (str): Name of the classifier
    """
    
    def __init__(self, 
                 path_to_model:str, 
                 save_path:str, 
                 data_path:str, 
                 new_labels:list, 
                 user:str, 
                 episodes:int = 20,
                 classifier = None,
                 classifier_name = None,
                
                ):
        
        self.classifier = classifier
        self.classifier_name = classifier_name
        
        super().__init__(path_to_model, save_path, data_path, new_labels, user, episodes, batch_size=64, augment=False)
        
        
    def load_model(self):
        # Fetch the model
        self.model = load_model(os.path.join(self.path_to_model, 'model_best.h5'))
        
        # Freeze all layers
        for layer in self.model.layers:
            layer.trainable = False
          
        # Remove the last layer 3 layers
        for _ in range(3):
            self.model.pop()
        
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables])           

    def train(self):
        if self.classifier is None:
            raise ValueError("Classifier is not defined")
        
        # Extract latent space and labels
        latent_space, y_labels = self.get_latent_space()
        
        # Convert to one hot encoding
        y_labels = np.argmax(y_labels, axis=1)

        # Train the classifier
        self.classifier.fit(latent_space, y_labels)
        
        # Save the model
        model_save_path = os.path.join(self.save_path, f"{self.classifier_name}.pkl")
        joblib.dump(self.classifier, model_save_path)
    
       
    def get_latent_space(self):
        if self.model is None:
            raise ValueError("Model is not loaded, Do you forget to call load_model() method?")
        
        latent_space  = []
        y_labels = []
        
        # Get the latent space of the augmented data
        if self.augment:
            print("Getting latent space of augmented data")
            groups = self.swimming_data.data_windows.keys()
            for group in groups:
                x, y = self.swimming_data.get_windows([self.user], group=group)
                x_re = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
                latent_space.append(self.model.predict(x_re))
                y_labels.append(y)
        
        # Get the latent space of only "original" data
        else: 
            print("Getting latent space of original data")
            x, y = self.swimming_data.get_windows([self.user])
            x_re = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
            latent_space.append(self.model.predict(x_re))
            y_labels.append(y)
        
        latent_space = np.vstack(latent_space)
        y_labels = np.vstack(y_labels)
        
        print(f"Latent space shape: {latent_space.shape}")
        print(f"y_labels shape: {y_labels.shape}")  
        
        # Save the latent space and y labels on save_path
        np.save(os.path.join(self.save_path, 'latent_space.npy'), latent_space)
        np.save(os.path.join(self.save_path, 'y_labels.npy'), y_labels)
        
        return latent_space, y_labels
        
          
if __name__ == '__main__':
    # Change this variable to define the new labels to train
    new_labels = [6,7,8,9] 
    
    # A path to re-sampled recordings which are organized into folders by user name.
    data_path = r'data_to_use'
    
    # Path of the model
    path_to_model = r"original"

    user = "train_all_replay"

    # List of classifiers to evaluate
    classifiers = {
        "logistic_regression": LogisticRegression(),
        "svm": SVC(),
        "rf": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(),
        "gradient_boosting": GradientBoostingClassifier()
    }

    for name, clf in classifiers.items():
        save_path = rf"Master\save_path\progressiveML_{user}\{name}"
        
        transfer = Progressive_ML(
            path_to_model=path_to_model,
            save_path=save_path, 
            data_path = data_path, 
            new_labels= new_labels, 
            user=user,
            classifier_name=name,
            classifier=clf
        
            )
        
        transfer.load_model()
        transfer.train()