import os
import pickle
import Master.other.learning_data_modified as learning_data_modified
import matplotlib.pyplot as plt
import numpy as np
import utils
import postprocessing

results_path = 'RESULTS_PATH'
save_path = 'SAVE_PATH'

with open(results_path, 'rb') as f:
    prediction_traces = pickle.load(f)[0]
users = np.sort(list(prediction_traces.keys()))


p_range = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
t_range = [6, 5, 4, 3, 2, 1]
prod_val = np.zeros(len(p_range) * len(t_range))
pt_pairs_all = [None] * len(p_range) * len(t_range)
fine_thresh = 1.0
cnt = 0
for i in range(len(p_range)):
    for ii in range(len(t_range)):
        pt_pairs_all[cnt] = (p_range[i], t_range[ii])
        prod_val[cnt] = (p_range[i]+0.01)*t_range[ii]
        cnt = cnt + 1
pt_pairs_all = np.array(pt_pairs_all)
a = np.argsort(prod_val).astype(int)[::-1]
pt_pairs_all = pt_pairs_all[a]
prod_val = prod_val[a]
pt_pairs = pt_pairs_all[prod_val>fine_thresh]
pt_pairs_fine = pt_pairs_all[prod_val<=fine_thresh]

alg_parameters = {'p': 0.5,
                  't_low': 1,
                  'close_size_1': 3 * 30,
                  'open_size_1': 6 * 30,
                  'close_size_2': 22 * 30,
                  'min_lane': 22 * 30,
                  'max_lane': 75 * 30,
                  'pt_pairs': pt_pairs,
                  'pt_pairs_fine': pt_pairs_fine,
                  'max_dist': 30 * 6
                  }

min_lane_dist = 10000000000000
max_lane_dist = 0
performance_mat = {user: {'missed': 0, 'correct': 0, 'additional': 0} for user in users}
performance_mat_lanes = {user: {'missed': 0, 'correct': 0, 'additional': 0} for user in users}
miss_add = [np.zeros(10000), np.zeros(10000)]
cnt = 0


num_lanes_pred = np.zeros((10000)) * np.nan
num_lanes_true = np.zeros((10000)) * np.nan
for (d, user) in enumerate(users):
    print("Working on user: %s, %d of %d" % (user, d+1, len(users)))
    recordings = list(prediction_traces[user].keys())
    for (k, rec) in enumerate(recordings):
        y_true = prediction_traces[user][rec]['raw']['true']
        if np.any((y_true == -1) | (y_true == 6)):
            print("Skipping recording: %s, b/c it contains unknowns and/or kicks" % rec)
            continue
        if y_true[-1] != 0:
            print("Skipping recording: %s, b/c it doesnt end with null" % rec)
            continue
        t_ns = np.ones(len(y_true))
        t_ns[y_true > 0] = 0
        is_ns_true = t_ns
        prob_ns = prediction_traces[user][rec]['raw']['pred'][:, 0]
        null_start_true, null_stop_true = utils.start_stop(is_ns_true)
        prob_ns[np.isnan(prob_ns)] = 1

        is_ns = postprocessing.smooth_nulls(x=prob_ns, p=alg_parameters['p'], t_low=alg_parameters['t_low'],
                                            close_size_1=alg_parameters['close_size_1'],
                                            open_size_1=alg_parameters['open_size_1'],
                                            close_size_2=alg_parameters['close_size_2'])
        null_start_init, null_stop_init = utils.start_stop(is_ns)
        # null_start = null_start_init
        # null_stop = null_stop_init
        null_start, null_stop = postprocessing.smooth_turns(x=prob_ns, null_start_init=null_start_init,
                                                            null_stop_init=null_stop_init,
                                                            min_lane=alg_parameters['min_lane'],
                                                            max_lane=alg_parameters['max_lane'],
                                                            pt_pairs=alg_parameters['pt_pairs'])
        # Add turns if missing
        null_start_init = np.copy(null_start)
        null_stop_init = np.copy(null_stop)
        for i in range(len(null_stop_init) - 1):
            lane_dist = null_start_init[i + 1] - null_stop_init[i]
            if lane_dist > alg_parameters['max_lane']:
                turn_start_new, turn_stop_new = \
                    postprocessing.smooth_turns(x=prob_ns, null_start_init=[null_start_init[i], null_start_init[i + 1]],
                                                null_stop_init=[null_stop_init[i], null_stop_init[i + 1]],
                                                min_lane=alg_parameters['min_lane'],
                                                max_lane=alg_parameters['max_lane'],
                                                pt_pairs=alg_parameters['pt_pairs_fine'])
                turn_start_new = turn_start_new[1:-1]
                turn_stop_new = turn_stop_new[1:-1]
                null_start = np.append(null_start, turn_start_new)
                null_stop = np.append(null_stop, turn_stop_new)

        null_start = np.sort(null_start)
        null_stop = np.sort(null_stop)
        pred_mat = np.zeros((len(null_start), len(null_start_true)))
        for i in range(len(null_start)):
            for ii in range(len(null_start_true)):
                if (np.abs(null_start[i] - null_start_true[ii]) < alg_parameters['max_dist']) & \
                        (np.abs(null_stop[i] - null_stop_true[ii]) < alg_parameters['max_dist']):
                    pred_mat[i, ii] = 1
        performance_mat[user]['missed'] = performance_mat[user]['missed'] + len(np.where(np.sum(pred_mat, axis=0) == 0)[0])
        performance_mat[user]['correct'] = performance_mat[user]['correct'] + len(np.where(np.sum(pred_mat, axis=0) == 1)[0])
        performance_mat[user]['additional'] = performance_mat[user]['additional'] + len(np.where(np.sum(pred_mat, axis=1) == 0)[0])
        lane_start = null_stop[:-1]
        lane_stop = null_start[1:]
        lane_start_true = null_stop_true[:-1]
        lane_stop_true = null_start_true[1:]
        pred_mat_lanes = np.zeros((len(lane_start), len(lane_start_true)))
        for i in range(len(lane_start)):
            for ii in range(len(lane_start_true)):
                if (np.abs(lane_start[i]-lane_start_true[ii]) < alg_parameters['max_dist']) & \
                        (np.abs(lane_stop[i]-lane_stop_true[ii]) < alg_parameters['max_dist']):
                    pred_mat_lanes[i, ii] = 1
        performance_mat_lanes[user]['missed'] = performance_mat_lanes[user]['missed'] + \
                                                len(np.where(np.sum(pred_mat_lanes, axis=0) == 0)[0])
        performance_mat_lanes[user]['correct'] = performance_mat_lanes[user]['correct'] + \
                                                 len(np.where(np.sum(pred_mat_lanes, axis=0) == 1)[0])
        performance_mat_lanes[user]['additional'] = performance_mat_lanes[user]['additional'] + \
                                                    len(np.where(np.sum(pred_mat_lanes, axis=1) == 0)[0])
        miss_add[0][cnt] = performance_mat_lanes[user]['missed']
        miss_add[1][cnt] = performance_mat_lanes[user]['missed']
        num_lanes_pred[cnt] = len(lane_start)
        num_lanes_true[cnt] = len(lane_start_true)
        cnt = cnt + 1

        for i in range(len(lane_start_true)):
            ld = lane_stop_true[i] - lane_start_true[i]
            if ld < min_lane_dist:
                min_lane_dist = ld
            if ld > max_lane_dist:
                max_lane_dist = ld

    print(performance_mat_lanes[user])

print(miss_add)
print(miss_add[0] - miss_add[1])
print("Minimum lane distance: %s" % min_lane_dist)
print("Maximum lane distance: %s" % max_lane_dist)
correct = np.sum([performance_mat[u]['correct'] for u in users])
additional = np.sum([performance_mat[u]['additional'] for u in users])
missed = np.sum([performance_mat[u]['missed'] for u in users])
precision = correct / (correct+additional)
recall = correct / (correct+missed)
correct_lanes = np.sum([performance_mat_lanes[u]['correct'] for u in users])
additional_lanes = np.sum([performance_mat_lanes[u]['additional'] for u in users])
missed_lanes = np.sum([performance_mat_lanes[u]['missed'] for u in users])
precision_lanes = correct_lanes / (correct_lanes+additional_lanes)
recall_lanes = correct_lanes / (correct_lanes+missed_lanes)

dirs = os.listdir(save_path)
c = len([d for d in dirs if d.startswith('prediction_lanes')])
file_name = 'prediction_lanes_' + str(int(c+1))
file = open(os.path.join(save_path, file_name + '.txt'), 'w')
file.write("Model path: %s\n\n" % results_path)
file.write("Correct: %d\n" % correct)
file.write("Additional: %d\n" % additional)
file.write("Missed: %d\n" % missed)
file.write("Precision: %s\n" % precision)
file.write("Recall: %s\n" % recall)
file.write("Correct (lanes): %d\n" % correct_lanes)
file.write("Additional (lanes): %d\n" % additional_lanes)
file.write("Missed (lanes): %d\n" % missed_lanes)
file.write("Precision (lanes): %s\n" % precision_lanes)
file.write("Recall (lanes): %s\n" % recall_lanes)
file.write("\nAlgorithm parameters:\n")
for key in alg_parameters.keys():
    file.write("%s: %s\n" % (key, alg_parameters[key]))
file.close()
print("Correct: %d" % correct)
print("Additional: %d" % additional)
print("Missed: %d" % missed)
print("Precision: %s" % precision)
print("Recall: %s" % recall)
print("Correct (lanes): %d" % correct_lanes)
print("Additional (lanes): %d" % additional_lanes)
print("Missed (lanes): %d" % missed_lanes)
print("Precision (lanes): %s" % precision_lanes)
print("Recall (lanes): %s" % recall_lanes)

with open(os.path.join(save_path,  file_name + '.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([performance_mat, performance_mat_lanes], f)

