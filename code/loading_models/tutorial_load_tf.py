"""
Script Name: tutorial_load_tf.py
Description:    This is script is used to load and use the models trained with the different transfer learning approaches.
                The main goal is to compare the performance of different configurations of the models on the specific save_path folder.
Author: Omar Almaraz Alonso
Version: 1.0
"""

import sys
sys.path.append("code")
import utils
import learning_data
import os
import glob
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
from constants import LABEL_NAMES

tf.compat.v1.disable_v2_behavior()
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

# Just change this varaiables
user = "freezing_train_all_replay"
user_data = "test_all"

# Define main paths
data_path = r'data_to_use'
results_path = rf"Master\results\{user}"
model_path = rf"Master\save_path\{user}"
utils.check_folder(results_path)

# Get the all model folders
model_folders = utils.get_subfolder_paths(model_path)

# Define metrics variables 
overrall_accuracy = 0 
f_combination  = ""
main_result_file_path = os.path.join(results_path, "main_report.txt")

for model_folder in model_folders:
    all_true_labels = []
    all_pred_labels = []
    model_accuracy = 0
    
    # extract the model name
    model_name = model_folder.split("\\")[-1]
    
    # Define the results path
    model_result_path = os.path.join(results_path, model_name)
    report_file_path = os.path.join(model_result_path, "report.txt")
    utils.check_folder(model_result_path)
    utils.remove_all_files_in_folder(model_result_path)
    result_file = open(report_file_path, "a")
    
    # Write into txt 
    result_file.write(f"\nModel: {model_name}\n")

    # Get the data parameters used for loading
    with open(os.path.join(model_folder, 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0]

    # Fetch the model
    model = load_model(os.path.join(model_folder, 'model_best.h5'))

    # Load the recordings
    swimming_data = learning_data.LearningData()
    swimming_data.load_data(data_path=data_path,
                            data_columns=data_parameters['data_columns'],
                            users=[user_data],
                            labels=data_parameters['labels'])
    if data_parameters['combine_labels'] is not None:
        for label in data_parameters['combine_labels'].keys():
            swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)
    swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

    recs = list(swimming_data.data_dict['original'][user_data].keys())
    prediction_traces = {rec: None for rec in recs}
    for (ii, rec) in enumerate(recs):
        print("Recording %d of %d" % (ii + 1, len(recs)))
        win_starts = swimming_data.window_locs['original'][user_data][rec][0]
        win_stops = swimming_data.window_locs['original'][user_data][rec][1]
        windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
        y_true_windows = np.zeros((len(windows), len(swimming_data.labels)))
        y_true_windows_maj = np.zeros(len(windows))
        for iii in range(len(win_starts)):
            win_start = win_starts[iii]
            win_stop = win_stops[iii]
            window = swimming_data.data_dict['original'][user_data][rec][swimming_data.data_columns].values[
                    win_start:win_stop + 1, :]
            window_norm = swimming_data.normalize_window(window, norm_type=data_parameters['window_normalization'])
            windows[iii] = window_norm
            win_labels = swimming_data.data_dict['original'][user_data][rec]['label'].values[win_start: win_stop + 1]
            win_label_cat, majority_label = swimming_data.get_window_label(win_labels, label_type='proportional',
                                                                        majority_thresh=0.4)
            y_true_windows[iii, :] = win_label_cat
            y_true_windows_maj[iii] = majority_label
        windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
        y_pred_windows = model.predict(windows)
        y_true_raw = swimming_data.data_dict['original'][user_data][rec]['label'].values
        win_mids = win_starts + (win_stops - win_starts) / 2
        x = win_mids
        y = y_pred_windows
        x_new = np.arange(0, len(y_true_raw))
        y_pred_raw = utils.resample(x, y.T, x_new, kind='nearest').T
        prediction_traces[rec] = {'window': {'true': y_true_windows, 'pred': y_pred_windows},
                                'raw': {'true': y_true_raw, 'pred': y_pred_raw}}

    labels_decoder = {value: key for key, value in swimming_data.labels_index.items()}
    labels_name = [LABEL_NAMES[indx] for indx in swimming_data.labels_index.keys()]
    for (i, rec) in enumerate(recs):
        # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        
        # Check if file contains word kick 
        if "Kick" in rec:
            rec_name = rec.split("Kick")[0]
        
        else: rec_name = rec.split("_")[0]
        
        # fig.suptitle("Recording %s" % rec_name)
        
        # ax[0].plot(swimming_data.data_dict['original'][user_data][rec]['ACC_0'].values)
        # ax[0].plot(swimming_data.data_dict['original'][user_data][rec]['ACC_1'].values)
        # ax[0].plot(swimming_data.data_dict['original'][user_data][rec]['ACC_2'].values)
        # ax[1].plot(prediction_traces[rec]['raw']['true'], label='True')
        
        # use swimming_data.labels_index to get the correct label
        yp = np.argmax(prediction_traces[rec]['raw']['pred'], axis=1)
        yp = [labels_decoder[i] for i in yp]
        true_p = prediction_traces[rec]['raw']['true']
        
        # Check if -1 is in the true labels and change it to 0 
        if -1 in true_p:
            true_p = [0 if i == -1 else i for i in true_p]
            
        if -1 in yp:
            yp = [0 if i == -1 else i for i in yp]
            
        all_true_labels.extend(true_p)
        all_pred_labels.extend(yp)
        
        accuracy = accuracy_score(true_p, yp)
        result_file.write(f"Recording: {rec}, Accuracy: {round(accuracy, 4)}\n")
        
        # ax[1].plot(yp, label='Predicted')
        
        # fig.legend()
        
        # save_path = os.path.join(model_result_path, f'{rec}.png')
        # plt.savefig(save_path)
    
    model_overall_acc = accuracy_score(all_true_labels, all_pred_labels)
    model_overall_acc = round(model_overall_acc, 4)
    report = classification_report(all_true_labels, all_pred_labels, target_names = labels_name)
    result_file.write(f"\nAverage accuracy: {model_overall_acc}\n")
    result_file.write(f"\nClassification report:\n{report}\n")

    
    if model_overall_acc > overrall_accuracy:
        overrall_accuracy = model_overall_acc
        f_combination = model_name
    

# close 
result_file.close()

with open(main_result_file_path, "w") as result_file:
    result_file.write(f"\n\nBest performance with: {f_combination}, acc: {overrall_accuracy}")
    result_file.close()