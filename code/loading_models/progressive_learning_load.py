
"""
Script Name: progressive_learning_load.py
Description: This is script is used to load and use the models trained in the progressive learning ml approach.
Author: Omar Almaraz Alonso 
Version: 1.0
"""
import os
import numpy as np
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
import learning_data
import glob 
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import utils
from constants import LABEL_NAMES_CM

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset


# Load the recordings
def load_data(results_path, user, data_path, new_labels = []):
    # Get the data parameters used for loading
    with open(os.path.join(results_path, 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0]
    
    for label in new_labels:
        if label not in data_parameters['labels']:
            data_parameters['labels'].append(label)
                
    swimming_data = learning_data.LearningData()
    swimming_data.load_data(data_path=data_path,
                            data_columns=data_parameters['data_columns'],
                            users=[user],
                            labels=data_parameters['labels'])
    if data_parameters['combine_labels'] is not None:
        for label in data_parameters['combine_labels'].keys():
            swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)
            
    swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])
    swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                                        label_type=data_parameters['label_type'],
                                        majority_thresh=data_parameters['majority_thresh'])
    
    return swimming_data

def load_base_model(path_to_model):
    # Fetch the model
    model = load_model(os.path.join(path_to_model, 'model_best.h5'))
    print(f"Loading model from: {path_to_model}")
    
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False
            
    # Remove the last layer 3 layers
    for _ in range(3):
        model.pop()
    
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])           
    print(f"Total trainable parameters: {total_params}\n")

    return model

def get_latent_space(swimming_data : learning_data, model, user):
    latent_space  = []
    y_labels = []
    
    x, y = swimming_data.get_windows([user])
    x_re = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    latent_space.append(model.predict(x_re))
    y_labels.append(y)

    latent_space = np.vstack(latent_space)
    y_labels = np.vstack(y_labels)
    
    return latent_space, y_labels
        
    
if __name__ == '__main__':
   
    base_model_path = r"original"
    data_path = r'data_to_use'
    

    user_data = 'test_all'
    user = "progressiveML_train_all_replay"
    
    upper_model_path = rf'Master\save_path\{user}'
    results_save_path = rf"Master\results\{user}"
    
    classifiers = [
        "logistic_regression",
        "svm",
        "rf",
        "knn",
        "decision_tree",
        "gradient_boosting"
    ]

    # Fetch the model
    new_labels = [6,7,8,9] 
    model = load_base_model(base_model_path)
    swimming_data = load_data(base_model_path, user_data, data_path, new_labels)
    labels_decoder = {value: key for key, value in swimming_data.labels_index.items()}
    labels_name = [LABEL_NAMES_CM[indx] for indx in swimming_data.labels_index.keys()]

    x_text, y_test = get_latent_space(swimming_data, model, user_data)
    
    # Convert to one hot encoding
    y_test = np.argmax(y_test, axis=1)
    
    print("Data shape: ", x_text.shape)
    print("Labels shape: ", y_test.shape)
    # print unique labels
    print("Unique labels: ", np.unique(y_test))
    
    for ctl in classifiers:
        
        clt_results_path = os.path.join(results_save_path, ctl)
        clt_model_folder  = os.path.join(upper_model_path, ctl)
        
        # Define the results path
        report_file_path = os.path.join(clt_results_path, "report.txt")
        utils.check_folder(clt_results_path)
        utils.remove_all_files_in_folder(clt_results_path)
    
        # Iterate through the classifiers and evaluate their performance
        clf = joblib.load(os.path.join(clt_model_folder, f"{ctl}.pkl"))

        # Make predictions
        y_pred = clf.predict(x_text)
        
        acc = accuracy_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(clt_results_path, "confusion_matrix.png")
        title = f"Classifier: {ctl.replace("_", " ")} - Confusion Matrix"
        utils.plot_confusion_matrix(cm, labels_name, title, cm_path)
        with open(report_file_path, 'w') as f:
            f.write(f"Classifier: {ctl}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(report)