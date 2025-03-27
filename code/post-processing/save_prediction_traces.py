# Load models and predict on recordings
# Save traces to a file for easy post-processing implementations
import sys
sys.path.append("code")
import os
import pickle
import Master.other.learning_data_modified as learning_data_modified
import numpy as np
import keras
import utils
import constants

data_path = r'.\data\processed_30Hz_relabeled'
results_path = r".\save_path\with_class_4\7"
save_path = r'.\results\with_class_4\7'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
with open(os.path.join(results_path, 'data_parameters.pkl'), 'rb') as f:
    data_parameters = pickle.load(f)[0]

users = data_parameters['users']
users_test = ['7']
if "7" not in users:
    users.append("7")

data_parameters["labels"] = [0, 1, 2, 3, 4, 5]

users_all = utils.folders_in_path(data_path)
users = users_all # [u for u in users_all if u not in users_test]
users.sort()

swimming_data = learning_data_modified.LearningData()
swimming_data.load_data(data_path=data_path, data_columns=data_parameters['data_columns'],
                        users=users, labels=data_parameters['labels'])

# swimming_data.normalize_recordings(detrend=data_parameters['detrend'],
#                                    norm_range=data_parameters['norm_recording_range'])
# swimming_data.normalize_global_2(norm_range=data_parameters['norm_global_range'])

for label in data_parameters['combine_labels'].keys():
    swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

prediction_traces = {user: {} for user in users_test}
for (i, user) in enumerate(users_test):
    print("Working on %s. %d of %d" % (user, i+1, len(users)))
    model = keras.models.load_model(os.path.join(results_path, 'model_best.keras'))
    recs = list(swimming_data.data_dict['original'][user].keys())
    for (ii, rec) in enumerate(recs):
        print("Recording %d of %d" % (ii+1, len(recs)))
        win_starts = swimming_data.window_locs['original'][user][rec][0]
        win_stops = swimming_data.window_locs['original'][user][rec][1]
        windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
        y_true_windows = np.zeros((len(windows), 5))
        y_true_windows_maj = np.zeros(len(windows))
        for iii in range(len(win_starts)):
            win_start = win_starts[iii]
            win_stop = win_stops[iii]
            window = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].values[win_start:win_stop+1, :]
            window_norm = swimming_data.normalize_window(window, norm_type=data_parameters['window_normalization'])
            windows[iii] = window_norm
            win_labels = swimming_data.data_dict['original'][user][rec]['label'].values[win_start: win_stop + 1]
            win_label_cat, majority_label = swimming_data.get_window_label(win_labels, label_type='proportional',majority_thresh=0.4)
            y_true_windows[iii, :] = win_label_cat
            y_true_windows_maj[iii] = majority_label
        windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
        y_pred_windows = model.predict(windows)
        y_true_raw = swimming_data.data_dict['original'][user][rec]['label'].values
        win_mids = win_starts + (win_stops - win_starts)/2
        x = win_mids
        y = y_pred_windows
        x_new = np.arange(0, len(y_true_raw))
        y_pred_raw = utils.resample(x, y.T, x_new, kind='nearest').T
        prediction_traces[user][rec] = {'window': {'true': y_true_windows, 'pred': y_pred_windows},
                                        'raw': {'true': y_true_raw, 'pred': y_pred_raw}}

with open(os.path.join(save_path, 'prediction_traces_best.pkl'), 'wb') as f:
    pickle.dump([prediction_traces], f)
