import numpy as np
import pandas as pd
import os
import csv
import scipy.interpolate
import scipy.stats
import constants
import glob 
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    Plot confusion matrix using Seaborn's heatmap.
    Args:
    cm: Confusion matrix (2D array)
    class_names: List of class names 
    """
    plt.title(title, fontsize=18)  # You can also increase the title's font size
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=0, fontsize=12)  # Set xtick font size
    plt.yticks(fontsize=12)  # Set ytick font size
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)

    # Adjust the font size of the color bar label
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def remove_all_files_in_folder(folder_path):
    """
    Remove all files in a folder
    :param folder_path: Path to the folder
    :return: None
    """
    files = glob.glob(os.path.join(folder_path, '*'))
    # Iterate through the list and remove each file
    for file in files:
        if os.path.isfile(file):  # Check if it's a file
            os.remove(file)
            
def clear_txt_file(file_path):
    """
    Clear a text file
    :param file_path: Path to the file
    :return: None
    """
    with open(file_path, 'w') as f:
        f.write("")
        
def check_folder(p):
    """
    Check if a folder exists, if not create it
    :param p: Path
    :return: None
    """
    if not os.path.exists(p):
        os.makedirs(p)
        
def get_subfolder_paths(p):
    """
    Get subfolder paths
    :param p: Path
    :return: List of subfolder paths
    """
    return [os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

def write_latex_confmat(cm, labels, is_integer=False):
    """
    Write confusion matrix into latex table
    :param cm: Two-dimensional confusion matrix
    :param labels: The labels in the confusion matrix
    :param is_integer: A boolean set to True if the values are integers
    :return: The confusion matrix in latex form
    """
    header = '\\begin{tabular}{' + 'c'*(len(labels)+1) + '}\n'
    footer = '\\end{tabular}'
    text = header
    text = text + '&' + '&'.join(label for label in labels) + '\\\\\n'
    for (i, label) in enumerate(labels):
        if is_integer:
            new_line = label + '&' + '&'.join([str(int(v)) for v in cm[i, :]]) + '\\\\\n'
        else:
            new_line = label + '&' + '&'.join(['%.1f' % v for v in cm[i, :]]) + '\\\\\n'
        text = text + new_line
    text = text + footer
    return text


def downsample_dataframe(df, skipstep):
    """
    Down-sample a pandas dataframe. Mainly used for plots.
    :param df: The pandas dataframe
    :param skipstep: The down-sampling factor
    :return:
    """
    df_new = df.iloc[::skipstep]
    x = np.arange(len(df['label'].values))
    x_new = x[::skipstep]
    for (i, col) in enumerate(df.columns):
        y = df[col].values
        if col == 'timestamp':
            continue
        if col == 'label':
            kind = 'nearest'
        else:
            kind = 'cubic'
        y_new = resample(x, y, x_new, kind=kind)
        df_new[col] = y_new
    return df_new


def start_stop(x):
    """
    Start-stop samples where the values in x change
    :param x: An array of values
    :return: Two numpy arrays containing the start and stop sample locations.
    """
    ix = np.where(x == 1)[0]
    if len(ix) == 0:
        starts = np.array([])
        stops = np.array([])
        return starts, stops
    ix_diff = ix[1:] - ix[:-1]
    ix_diff_jump = np.where(ix_diff > 1)[0]
    starts = np.append(ix[0], ix[ix_diff_jump + 1])
    stops = np.append(ix[ix_diff_jump] + 1, ix[-1] + 1)
    return starts, stops


def unclose(x, open_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param open_size: Opening threshold
    :return: An array with the values in x after opening
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x)+1, i+open_size+1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 0:
            ix_uno = np.where(xwin == 1)[0]
            if len(ix_uno) > 0:
                if 0 in xwin[ix_uno[0]:]:
                    ix_end = ix_1 + ix_uno[0] + ix_uno[-1]
                    y[ix_1:ix_end] = 0
    return y


def close(x, close_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param close_size: Closing threshold
    :return: An array with the values in x after closing
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x)+1, i+close_size+1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 1:
            ix_null = np.where(xwin == 0)[0]
            if len(ix_null) == 0:
                y[i] = x[i]
            elif 1 in xwin[ix_null[0]:]:
                ix_uno = np.where(xwin[ix_null[0]:] == 1)[0]
                ix_end = ix_1 + ix_null[0] + ix_uno[-1]
                y[ix_1:ix_end] = 1
            else:
                y[i] = x[i]
    return y


def write_confusion_matrix(cm, labels):
    """
    Write confusion matrix to text. Use to get a cleaner output
    :param cm: Two-dimensional confusion matrix
    :param labels: A list of labels
    :return: The confusion matrix in text format
    """
    m = 15
    o = " ".ljust(m)
    for label in labels:
        o = o + constants.LABEL_NAMES[label].ljust(m)
    o = o + "\n"
    for (i, label) in enumerate(labels):
        o = o + constants.LABEL_NAMES[label].ljust(m)
        for j in range(len(labels)):
            val = cm[i, j]
            if val - int(val) == 0:
                val_str = str(int(val))
            else:
                val_str = "%.1f" % cm[i, j]
            o = o + val_str.ljust(m)
        o = o + "\n"
    return o

def write_confusion_matrix_expanded(cm, labels):
    m = 15
    o = " ".ljust(m)
    for label in labels:
        o = o + constants.LABEL_NAMES[label].ljust(m)
    o = o + "Recall".ljust(m) + "\n"
    for (i, label) in enumerate(labels):
        o = o + constants.LABEL_NAMES[label].ljust(m)
        for j in range(len(labels)):
            val = cm[i, j]
            if val - int(val) == 0:
                val_str = str(int(val))
            else:
                val_str = "%.1f" % cm[i, j]
            o = o + val_str.ljust(m)
        val = cm[i, -1]
        if val - int(val) == 0:
            val_str = str(int(val))
        else:
            val_str = "%.1f" % cm[i, -1]
        o = o + val_str.ljust(m)
        o = o + "\n"
    
    # Add precision
    o = o + "Precision".ljust(m)
    for j in range(len(labels)):
        val = cm[-1, j]
        if val - int(val) == 0:
            val_str = str(int(val))
        else:
            val_str = "%.1f" % cm[-1, j]
        o = o + val_str.ljust(m)
    return o



def normalize_confusion_matrix(cm):
    """
    Normalize confusion matrix w.r.t. the class size
    :param cm: Two-dimensional confusion matrix
    :return:
    """
    cmn = np.zeros(cm.shape)
    label_count = np.sum(cm, axis=1)
    for (i, c) in enumerate(label_count):
        if c != 0:
            cmn[i, :] = cm[i, :] / c
    return cmn

def expand_confusion_matrix(conf_matrix):
    # Number of classes
    num_classes = conf_matrix.shape[0]
    
    # Calculate recall for each class (row)
    row_sums = conf_matrix.sum(axis=1)
    true_positives = np.diag(conf_matrix)
    recall = true_positives / row_sums
    recall = recall[:, np.newaxis]  # Convert to a column vector
    
    # Calculate precision for each class (column)
    column_sums = conf_matrix.sum(axis=0)
    precision = true_positives / column_sums
    precision = np.nan_to_num(precision)  # Handle division by zero by replacing NaN with 0
    
    # Append recall as a new column to the matrix
    conf_matrix_with_recall = np.hstack((conf_matrix, recall))
    
    # Append precision as a new row to the matrix
    precision_with_nan = np.append(precision, np.nan)  # Add NaN for the last cell (precision of recall column)
    expanded_matrix = np.vstack((conf_matrix_with_recall, precision_with_nan))
    expanded_matrix = np.nan_to_num(expanded_matrix)  # Handle NaN values by replacing them with 0
    return expanded_matrix


def normalize_range(x):
    """
    Normalize range, i.e. to  0, 1
    :param x: An array of values
    :return: x normalized
    """
    max_val = np.max(x)
    min_val = np.min(x)
    return (x-min_val) / (max_val-min_val)


def detrend(x, window_length=600, return_trend=False):
    """
    Remove trend form an array
    :param x: An array of values
    :param window_length: The length used to compute the moving average trend
    :param return_trend: Boolean indicating whether to return the trend or x detrended
    :return: x detrended or the trend
    """
    tail_length = np.floor(window_length / 2)
    nose_length = np.ceil(window_length / 2)
    trend = np.zeros(len(x))
    for i in range(len(x)):
        if i < tail_length:
            first_ix = 0
        else:
            first_ix = int(i - tail_length)
        if i > len(x) - nose_length:
            last_ix = len(x)
        else:
            last_ix = int(i + nose_length)
        trend[i] = np.mean(x[first_ix: last_ix])
    if not return_trend:
        return x - trend
    else:
        return trend


def diff(x):
    """
    The derivative on an array
    :param x:   An array of values
    :return:    An array for the derivative of x
    """
    return x[1:] - x[:-1]


def dirs_in_path(p):
    """
    List all directories in a path
    :param p: A full path
    :return: A list of strings of names of all directories in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]


def load_recording(file_path, category='raw'):
    """
    Load a recording into a Pandas DataFrame
    :param file_path:   Path to .csv file containing swimming data
    :param category:    = 'raw', 'labeled' or 'processed'
    :return: Recording data in a Pandas DataFrame. If category is 'raw' or 'labeled', the header and footer are also
             returned
    """
    if category == 'raw' or category == 'labeled':
        df = pd.read_csv(file_path, sep='; ', header=None, skiprows=[0], skipfooter=1, engine='python')
        df.columns = constants.LABELED_COL_NAMES[0: len(df.columns)]
        header = list(pd.read_csv(file_path, sep='; ', nrows=1, engine='python').columns)
        with open(file_path, 'r') as f:
            footer = list(csv.reader(f))[-1]
            footer = footer[0].split("; ")
        return df, header, footer
    if category == 'processed':
        df = pd.read_csv(file_path)
        return df


def resample(x, y, x_new, kind='cubic'):
    """
    A simple wrapper for interp1d
    :param x: Original timestamps
    :param y: Original values
    :param x_new: New timestamps
    :param kind: interpolation type
    :return: The values in y evaluated at x_new
    """
    f = scipy.interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)
    y_new = f(x_new)
    return y_new


def time_scale_dataframe(df, factor, time_col, label_col):
    """
    Time-scale a dataframe
    :param df: A pandas dataframe
    :param factor: Time-scaling factor
    :param time_col: The column name of timestamps
    :param label_col: The column name of labels
    :return: Time-scaled dataframe
    """
    data_cols = [col for col in df.columns if col not in [time_col, label_col]]
    df_new = pd.DataFrame(columns=df.columns)
    dt = df[time_col].values[1] - df[time_col].values[0]
    dts = dt/factor
    t = df[time_col].values
    ts = t[0] + np.arange(len(t))*dts
    t_target = np.arange(t[0], ts[-1], dt)
    df_new['timestamp'] = t_target
    for col in data_cols:
        y = df[col].values
        df_new[col] = resample(ts, y, t_target, 'cubic')
    df_new[label_col] = resample(ts, df[label_col].values, t_target, 'nearest')
    return df_new


def get_sample_weights_new(y_cat):
    """
    Compute sample weights based on class size
    :param y_cat: Labels in categorical form
    :return: The sample weights
    """
    class_weights = np.zeros(y_cat.shape[1])
    for i in range(y_cat.shape[1]):
        if np.sum(y_cat[:, i]) == 0:
            class_weights[i] = 0
        else:
            class_weights[i] = 1/np.sum(y_cat[:, i])
    y_sample_weights = np.sum(y_cat * class_weights, axis=1)/np.sum(y_cat, axis=1)
    y_sample_weights = y_sample_weights/np.sum(y_sample_weights)*len(y_sample_weights)
    return y_sample_weights


def from_categorical(y_cat):
    """
    From categorical to normal labeling
    :param y_cat: Two dimensional array of categorical labels
    :return: An array with normal labeling
    """
    y = np.argmax(y_cat, axis=1)
    return y.astype(int)


def folders_in_path(p):
    """
    Get folders in path
    :param p: Path
    :return: List of folder names in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]


# PCHIP Interpolation: Similar to cubic spline, but focuses on retaining the shape of the data’s monotonicity (preserving the increasing or decreasing nature between points) 
def pchip_interpolate(series):
    # Get the index and values, drop NaN values for interpolation
    index = series.index
    valid = series.dropna()
    interpolator = scipy.interpolate.PchipInterpolator(valid.index, valid.values)
    return pd.Series(interpolator(index), index=index)

def fotmat_df(df, include_label=False):
    pivoted_df = df.pivot(index='timestamp', columns='sensor', values=['value_0', 'value_1', 'value_2'])
    # Flatten the MultiIndex columns
    pivoted_df.columns = [f'{sensor}_{val.split("_")[-1]}' for val, sensor in pivoted_df.columns]
    # Reset the index to make 'timestamp' a column again
    pivoted_df.reset_index(inplace=True)

    # remove columns
    pivoted_df = pivoted_df.drop(columns=['PRESS_1', "PRESS_2","LIGHT_1", "LIGHT_2"])
    
    # rename columns LIGHT_0 and PRESS_0 to LIGHT and PRESS
    pivoted_df = pivoted_df.rename(columns={"LIGHT_0": "LIGHT", "PRESS_0": "PRESS"})
    
    # add label column 
    if include_label: pivoted_df['label'] = -1
    
    return pivoted_df

def main():
    print("Main")


if __name__ == '__main__':
    main()
