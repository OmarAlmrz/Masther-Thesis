import pandas as pd
import numpy as np
import utils
import os
import segmentation
import constants


class LearningData(object):

    def __init__(self):
        self.users = None
        self.labels = None
        self.labels_index = None
        self.data_dict = None
        self.data_windows = None
        self.data_path = None
        self.windows = None
        self.window_locs = None
        self.win_len = None
        self.slide_len = None
        self.columns = None
        self.data_columns = None
        self.data_columns_ix = None
        self.sensors = None
        self.mirror_prob = None
        self.mirror_ix = None
        self.noise_std = None
        self.time_scale_factors = None
        self.group_probs = {'original': 1.0}
        self.label_type = None
        self.multi_class_thresh = None
        self.majority_thresh = None
        self.norm_type = None

    def normalize_recordings(self, detrend=None, norm_range=None):
        """
        Normalize data on a recording-by-recording basis. Currently provides detrending and range normalization as
        possibly required by the pressure and light sensors.
        :param detrend: A dictionary of booleans whose keys are sensors. Specifies which sensors are detrended
        :param norm_range: A dictionary of booleans whose keys are sensors. Specifies which sensors are normalized
                           across the range.
        :return: self.data_dict is modified appropriately.
        """
        if detrend is None:
            detrend = {sensor: False for sensor in self.sensors}
        if norm_range is None:
            norm_range = {sensor: False for sensor in self.sensors}
            norm_range['LIGHT'] = True
        for user in self.users:
            print("Normalizing recordings for: %s" % user)
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    sensor_cols = [col for col in self.data_columns if col.startswith(sensor)]
                    if detrend[sensor] is True:
                        for col in sensor_cols:
                            self.data_dict['original'][user][rec][col] = \
                                utils.detrend(self.data_dict['original'][user][rec][col].values)
                    if norm_range[sensor] is True:
                        for col in sensor_cols:
                            self.data_dict['original'][user][rec][col] = \
                                utils.normalize_range(self.data_dict['original'][user][rec][col].values)

    def normalize_global(self, norm_range=None):
        """
        Normalize data globally. Range normalization is the only feature available.
        :param norm: A dictionary of booleans whose keys are sensors. Specifies which sensors are normalized over the
                     range.
        :return: self.data_dict is modified appropriately
        """
        if norm_range is None:
            norm = {sensor: True for sensor in self.sensors}
        sensor_max = {key: -np.inf for key in self.sensors}
        sensor_min = {key: np.inf for key in self.sensors}
        print("Computing normalization values...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        sensor_local_max = np.max(self.data_dict['original'][user][rec][cols].values)
                        sensor_local_min = np.min(self.data_dict['original'][user][rec][cols].values)
                        sensor_max[sensor] = np.max([sensor_max[sensor], sensor_local_max])
                        sensor_min[sensor] = np.min([sensor_min[sensor], sensor_local_min])
        sensor_shift = {key: sensor_min[key] for key in self.sensors}
        sensor_scale = {key: sensor_max[key] - sensor_min[key] for key in self.sensors}
        print("sensor_shift: %s, sensor_scale: %s" % (sensor_shift, sensor_scale))
        print("Finished computing normalization values")
        print("Normalizing recordings...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        for col in cols:
                            self.data_dict['original'][user][rec][col] = \
                                (self.data_dict['original'][user][rec][col].values - sensor_shift[sensor]) / \
                                sensor_scale[sensor]
        print("Finished normalizing recordings")

    def normalize_global_2(self, norm_range=None):
        """
        The same as normalize_global but uses pre-calculated normalization values for speed purposes. The
        pre-calculated values can be stored under constants.py
        :param norm: See normalize_global
        :return: See normalize_global
        """
        if norm_range is None:
            norm = {sensor: True for sensor in self.sensors}
        sensor_shift = {'ACC': -90.83861868871521, 'GYRO': -32.85540978272818, 'LIGHT': np.nan,
                        'MAG': -184.9962120725973, 'PRESS': np.nan}
        sensor_scale = {'ACC': 185.55766760223042, 'GYRO': 58.16958899110071, 'LIGHT': np.nan,
                        'MAG': 577.7807153840964, 'PRESS': np.nan}
        print("Normalizing recordings...")
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                for sensor in self.sensors:
                    if norm_range[sensor]:
                        cols = [col for col in self.data_columns if col.startswith(sensor)]
                        for col in cols:
                            self.data_dict['original'][user][rec][col] = \
                                (self.data_dict['original'][user][rec][col].values - sensor_shift[sensor]) / \
                                sensor_scale[sensor]
        print("Finished normalizing recordings")

    def load_data(self, data_path, data_columns, labels, users=None):
        """
        Load processed swimming data.
        :param data_path: Path to processed swimming data.
        :param data_columns: Sensor columns that are used in the data set.
        :param labels: Labels that are read into the data windows
        :param users: Users whose data is loaded. users=None means everybody is loaded.
        :return: A dictionary containing all data
        """
        self.data_columns = data_columns
        self.data_columns_ix = {key: ix for (ix, key) in enumerate(data_columns)}
        self.sensors = list(np.unique([col.split('_')[0] for col in self.data_columns]))
        self.labels = list(np.sort(labels))
        self.labels_index = {label: ix for (ix, label) in enumerate(self.labels)}
        columns = set(data_columns)
        columns.update(['timestamp', 'label'])
        self.columns = list(columns)
        self.mirror_ix = [ix for ix in range(len(data_columns)) if data_columns[ix] in constants.AXIS_MIRROR]
        if users is None:
            self.users = utils.dirs_in_path(data_path)
        else:
            self.users = users
        self.data_dict = {'original': {user: dict() for user in self.users}}
        for user in self.users:
            print("Loading user: %s" % user)
            csv_in_path = os.listdir(os.path.join(data_path, user))
            self.data_dict['original'][user] = {rec: None for rec in csv_in_path}
            for rec in csv_in_path:
                file_path = os.path.join(data_path, user, rec)
                self.data_dict['original'][user][rec] = pd.read_csv(file_path)[self.columns]

    def augment_recordings(self, time_scale_factors=None):
        """
        Create augmented versions of recordings through time scaling
        :param time_scale_factors: A list of factors used to create time-scaled versions of original recordings
        :return: The new augmented versions are stored inside the data dictionary
        """
        if time_scale_factors is not None:
            self.time_scale_factors = time_scale_factors
            for factor in time_scale_factors:
                print("Augmenting with time-scale factor: %s" % factor)
                new_group = 'time_scaled_' + str(factor)
                self.data_dict[new_group] = {user: dict() for user in self.users}
                for user in self.users:
                    for rec in list(self.data_dict['original'][user].keys()):
                        self.data_dict[new_group][user][rec] = \
                            utils.time_scale_dataframe(self.data_dict['original'][user][rec], factor, 'timestamp', 'label')
        groups = list(self.data_dict.keys())
        self.group_probs = {group: 1/len(groups) for group in groups}

    def sliding_window_locs(self, win_len, slide_len):
        """
        Compute sliding window start and stop sample index for each original and augmented recording
        :param win_len: Window length in number of samples
        :param slide_len: Slide length in number of samples
        :return: A dictionary with the same structure as the data dictionary but containing window start and stop
                 timestamps
        """
        self.win_len = win_len
        self.slide_len = slide_len
        groups = self.data_dict.keys()
        self.window_locs = {group: {user: dict() for user in self.users} for group in self.data_dict.keys()}
        for group in groups:
            for user in self.users:
                for rec in self.data_dict[group][user].keys():
                    x = np.arange(len(self.data_dict[group][user][rec]['timestamp'].values))
                    self.window_locs[group][user][rec] = \
                        segmentation.sliding_windows_start_stop(x=x, win_len=self.win_len,
                                                                slide_len=self.slide_len)

    def normalize_window(self, win_data, norm_type='statistical'):
        """
        Normalize the data of a window
        :param win_data: A 2-D array of data. Columns are different sensors.
        :param norm_type: The normalization type employed: 'statistical', 'mean' or 'statistical_combined'
        :return: The same win_data normalized appropriately
        """
        if norm_type == 'statistical':
            win_data = (win_data - np.mean(win_data, axis=0)) / np.std(win_data, axis=0)
        elif norm_type == 'mean':
            win_data = win_data - np.mean(win_data, axis=0)
        elif norm_type == 'statistical_combined':
            win_data = win_data - np.mean(win_data, axis=0)
            for sensor in self.sensors:
                sensor_cols_ix = np.where(self.data_columns.startswith(sensor))
                std_sensor = np.mean(np.std(win_data[:, sensor_cols_ix], axis=0))
                win_data[:, sensor_cols_ix] = win_data[:, sensor_cols_ix] / std_sensor
        else:
            raise ValueError("Invalid window normalization type")
        return win_data

    def get_window_label(self, win_labels, label_type='proportional', multi_class_thresh=0, majority_thresh=0):
        """
        Get an array of categorized labels
        :param win_labels: An array of sample-by-sample labels
        :param label_type: The type of labeling employed: 'proportional', 'majority' or 'multi_class'
        :param multi_class_thresh: Only used if label_type='multi_class'. The proportion of samples that are needed
                                   such that a label is considered as part of the window
        :param majority_thresh: Only used if label_type='majority'. The proportion of samples needed such that the
                                window is labeled of that type.
        :return: An array of categorized labels
        """
        win_labels_list, label_count = np.unique(win_labels, return_counts=True)
        if len(set(win_labels_list) - set(self.labels)) > 0:
            # Ignore windows that contain labels that are not included
            return None, None
        majority_label = win_labels_list[np.argmax(label_count)]
        win_label_cat = np.zeros(len(self.labels))
        for (ii, label) in enumerate(win_labels_list):
            win_label_cat[self.labels_index[label]] = label_count[ii] / self.win_len
        if label_type == 'proportional':
            return win_label_cat, majority_label
        elif label_type == 'multi_class':
            ix_above_thresh = np.where(win_label_cat - multi_class_thresh >= 0)[0]
            if len(ix_above_thresh) == 0:
                # Ignore window with no clear label
                return None, None
            else:
                win_label_cat = np.zeros(len(self.labels))
                win_label_cat[ix_above_thresh] = 1
                win_label_cat = win_label_cat / np.sum(win_label_cat)
                return win_label_cat, majority_label
        elif label_type == 'majority':
            if np.max(win_label_cat) > majority_thresh:
                ix_majority = np.argmax(win_label_cat)
                win_label_cat = np.zeros(len(self.labels))
                win_label_cat[ix_majority] = 1
                return win_label_cat, majority_label
            else:
                # Ignore window if no clear majority
                return None, None
        else:
            raise ValueError("Invalid labeling type")

    def compile_windows(self, norm_type='statistical', label_type='proportional', multi_class_thresh=0.2,
                        majority_thresh=0):
        """
        Compile windows based on window and slide lengths.
        :param norm_type: Window normalization type: 'statistical', 'statistical_combined' or 'mean'
        :param label_type: Labeling type: 'majority', 'proportional' or 'multi_class'
        :param multi_class_thresh: Threshold above which a label is assigned under multi class labeling. Only used if
                                   label_type = 'multi_class'
        :param majority_thresh: A threshold for the minimum proportion of the majority label in a window
        :return: self.data_windows created
        """
        groups = self.data_dict.keys()
        self.data_windows = {group: {label: dict() for label in self.labels} for group in groups}
        for group in groups:
            for user in self.users:
                print("Compiling windows: %s, %s" % (group, user))
                temp_user_windows = {label: {'data': np.zeros((10000, self.win_len, len(self.data_columns))),
                                             'label': np.zeros((10000, len(self.labels)))}
                                     for label in self.labels}
                cnt_user_label = {label: 0 for label in self.labels}
                for rec in self.data_dict[group][user].keys():
                    for i in range(len(self.window_locs[group][user][rec][0])):
                        win_start = self.window_locs[group][user][rec][0][i]
                        win_stop = self.window_locs[group][user][rec][1][i]
                        win_data = self.data_dict[group][user][rec][self.data_columns].values[win_start: win_stop+1]
                        win_data = self.normalize_window(win_data, norm_type=norm_type)
                        win_labels = self.data_dict[group][user][rec]['label'].values[win_start: win_stop+1]
                        win_label_cat, majority_label = self.get_window_label(win_labels, label_type=label_type,
                                                                              multi_class_thresh=multi_class_thresh,
                                                                              majority_thresh=majority_thresh)
                        if win_label_cat is None:
                            # A "bad" window was returned
                            continue
                        temp_user_windows[majority_label]['data'][cnt_user_label[majority_label]] = win_data
                        temp_user_windows[majority_label]['label'][cnt_user_label[majority_label]] = win_label_cat
                        cnt_user_label[majority_label] = cnt_user_label[majority_label] + 1
                # Strip away from temp_user_windows and save in data_windows
                for label in self.labels:
                    if cnt_user_label[label] == 0:
                        continue
                    else:
                        self.data_windows[group][label][user] = \
                            {'data': np.copy(temp_user_windows[label]['data'][0: cnt_user_label[label]]),
                             'label': np.copy(temp_user_windows[label]['label'][0: cnt_user_label[label]])}
        self.norm_type = norm_type
        self.label_type = label_type
        self.multi_class_thresh = multi_class_thresh
        self.majority_thresh = majority_thresh

    def batch_generator_dicts(self, train_dict, batch_size=64, noise_std=None, mirror_prob=None, random_rot_deg=30):
        """
        A generator that yields a random set of windows
        :param train_dict: A dictionary with keys corresponding to labels and values that are lists of users used for
                           the label
        :param batch_size: The number of windows yielded
        :param noise_std: An optional noise standard deviation added onto the windows
        :param mirror_prob: An optional probability of mirroring the axes
        :return: batch_data: A 4 dimensional numpy array of window data. The last dimension is a dummy "1" dimension
                 batch_labels_cat: A 2 dimensional numpy array of corresponding window labels in categorical form
        """
        groups = list(self.data_dict.keys())
        group_probs = [self.group_probs[group] for group in groups]
        labels_avaialble = list(train_dict.keys())
        while True:
            batch_data = np.zeros((batch_size, self.win_len, len(self.data_columns)))
            batch_labels_cat = np.zeros((batch_size, len(self.labels)))
            for i in range(batch_size):
                r_group = np.random.choice(groups, p=group_probs)
                r_label = np.random.choice(labels_avaialble)
                r_user = np.random.choice(train_dict[r_label])
                r_win = np.random.choice(len(self.data_windows[r_group][r_label][r_user]['data']))
                batch_data[i, :, :] = self.data_windows[r_group][r_label][r_user]['data'][r_win]
                batch_labels_cat[i] = self.data_windows[r_group][r_label][r_user]['label'][r_win]
            if mirror_prob is not None:
                mirror_samples = np.random.choice([True, False], p=[mirror_prob, 1 - mirror_prob], size=batch_size)
                if any(mirror_samples) is True:
                    batch_data[mirror_samples][:, self.mirror_ix] = -batch_data[mirror_samples][:,  self.mirror_ix]
            if random_rot_deg is not None:
                r_theta = np.random.uniform(-random_rot_deg/180*np.pi, random_rot_deg/180*np.pi, batch_size)
                for i in range(batch_size):
                    for sensor in self.sensors:
                        if sensor in ['ACC', 'GYRO', 'MAG']:
                            ix_1 = self.data_columns_ix[sensor + '_1']
                            ix_2 = self.data_columns_ix[sensor + '_2']
                            b1 = np.copy(batch_data[i, :, ix_1])
                            b2 = np.copy(batch_data[i, :, ix_2])
                            batch_data[i, :, ix_1] = b1 * np.cos(r_theta[i]) - b2 * np.sin(r_theta[i])
                            batch_data[i, :, ix_2] = b1 * np.sin(r_theta[i]) + b2 * np.cos(r_theta[i])
            if noise_std is not None:
                batch_data = batch_data + np.random.normal(0, noise_std, batch_data.shape)
            batch_data = batch_data.reshape((batch_data.shape[0], batch_data.shape[1], batch_data.shape[2], 1))
            yield (batch_data, batch_labels_cat)

    def get_windows(self, users, group='original'):
        """
        Get all windows from a set of users
        :param users: A list of users
        :return: x_val: A 3 dimensional numpy array of window data
                 y_val: A 1 dimensional numpy array of corresponding labels
        """
        x_val = None
        y_val_cat = None
        for label in self.labels:
            for user in users:
                if user in self.data_windows[group][label].keys():
                    x_val_new = self.data_windows[group][label][user]['data']
                    y_val_cat_new = self.data_windows[group][label][user]['label']
                    if x_val is None:
                        x_val = x_val_new
                        y_val_cat = y_val_cat_new
                    else:
                        x_val = np.concatenate((x_val, x_val_new))
                        y_val_cat = np.concatenate([y_val_cat, y_val_cat_new])
        return x_val, y_val_cat

    def get_windows_dict(self, label_user_dict, return_weights=False):
        """
        Get all windows from a set of users
        :param label_user_dict: A dictionary of labels (keys) and users (values)
               return_weights: A boolean indicating whether to return the sample weights
        :return: x_val: A 3 dimensional numpy array of window data
                 y_val: A 1 dimensional numpy array of corresponding labels
                 sample_weights: Weight for each sample

        """
        x_val = None
        y_val_cat = None
        folds = {'start': [], 'stop': [], 'size': []}
        labels = list(label_user_dict.keys())
        cnt = 0
        for label in labels:
            for user in label_user_dict[label]:
                if user in self.data_windows['original'][label].keys():
                    x_val_new = self.data_windows['original'][label][user]['data']
                    y_val_cat_new = self.data_windows['original'][label][user]['label']
                    if x_val is None:
                        x_val = x_val_new
                        y_val_cat = y_val_cat_new
                        folds['start'].append(0)
                        folds['stop'].append(len(x_val_new))
                    else:
                        x_val = np.concatenate((x_val, x_val_new))
                        y_val_cat = np.concatenate([y_val_cat, y_val_cat_new])
                        folds['start'].append(folds['stop'][cnt - 1])
                        folds['stop'].append(folds['start'][cnt] + len(x_val_new))
                    folds['size'].append(len(x_val_new))
                    cnt = cnt + 1
        if not return_weights:
            return x_val, y_val_cat
        else:
            sample_weights = np.zeros(len(x_val))
            for i in range(len(folds['start'])):
                sample_weights[folds['start'][i]:folds['stop'][i]] = 1/folds['size'][i]
            sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
            return x_val, y_val_cat, sample_weights

    def create_custom_user(self, label_user_dict, name='custom'):
        """
        Create a custom user based on label-user combinations
        :param label_user_dict: A dictionary of label (key) and user (value) combinations
        :param name: The name given to the custom user
        :return: The custom user is created within the object. The originals are removed.
        """
        for style in label_user_dict.keys():
            for user in label_user_dict[style]:
                for group in self.data_windows.keys():
                    if name not in self.data_windows[group][style].keys():
                        self.data_windows[group][style][name] = self.data_windows[group][style][user]
                    else:
                        x_new = self.data_windows[group][style][user]['data']
                        y_new = self.data_windows[group][style][user]['label']
                        self.data_windows[group][style][name]['data'] = \
                            np.concatenate((self.data_windows[group][style][name]['data'], x_new))
                        self.data_windows[group][style][name]['label'] = \
                            np.concatenate((self.data_windows[group][style][name]['label'], y_new))
                    self.data_windows[group][style].pop(user, None)
        self.users.append(name)

    def combine_labels(self, labels, new_label):
        """
        Combine labels under a new label
        :param labels: A list of labels
        :param new_label: A value for the new label
        :return: self.data_dict['original'] and self.labels have been appropriately modified
        """
        for user in self.data_dict['original'].keys():
            for rec in self.data_dict['original'][user].keys():
                current_labels = self.data_dict['original'][user][rec]['label'].values
                ix = np.where(current_labels == labels[0])[0]
                for label in labels[1:]:
                    ix = np.concatenate([ix, np.where(current_labels == label)[0]])
                if len(ix) == 0:
                    continue
                self.data_dict['original'][user][rec]['label'].values[ix] = new_label
        for label in labels:
            self.labels.remove(label)
        self.labels.append(new_label)
        self.labels = list(np.sort(self.labels))
        self.labels_index = {label: ix for (ix, label) in enumerate(self.labels)}

    def draw_train_val_dicts(self, users_train, users_per_class=3, validation=True):
        """
        Draw a set of train/validation label-user combinations
        :param users_train: List of users in the full training set
        :param users_per_class: Number of users for each label in the validation set
        :return: train_dict: A dictionary of label-user combinations used in the training set
                 val_dict: A dictionary of label-user combinations used in the validation set
        """
        if validation:
            val_dict = {l: None for l in self.labels}
            train_dict = {l: None for l in self.labels}
            for l in self.labels:
                users_in_class = list(self.data_windows['original'][l].keys())
                users_train_in_class = [u for u in users_in_class if u in users_train]
                if type(users_per_class) is dict:
                    val_dict[l] = np.random.choice(users_train_in_class, users_per_class[l], replace=False)
                else:
                    val_dict[l] = np.random.choice(users_train_in_class, users_per_class, replace=False)
                train_dict[l] = [u for u in users_train_in_class if u not in val_dict[l]]
            return train_dict, val_dict
        
        else :
            train_dict = {}
            for l in self.data_windows['original'].keys() :
                users_in_class = list(self.data_windows['original'][l].keys())
                users_train_in_class = [u for u in users_in_class if u in users_train]
                if users_train_in_class:    
                    train_dict[l] = users_train_in_class
            return train_dict, None


def main():
    print("Running main")

if __name__ == '__main__':
    main()
