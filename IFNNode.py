import math
from scipy.stats import chi2
import numpy as np
from operator import is_not
from functools import partial


class IFNNode:
    def __init__(self, n_samples, targets, prev_node_feature_split_value, network, random_features_arr):
        self.X = None
        self.y = None
        self.n_samples = n_samples
        self.targets = targets
        self.next_nodes = []
        self.network = network
        self.targets_prob = {}
        self.input_att_mi = {}
        self.feature_split = None
        self.prev_node_feature_split_value = prev_node_feature_split_value
        self.att_t_split = {}
        self.right_interval = None
        self.left_interval = None
        self.random_feature_arr = random_features_arr

    def my_count(self, x_v, y_v, self_x_iloc):
        slice_y = self.y[self_x_iloc == x_v]
        return slice_y[slice_y.iloc[:, 0] == y_v].shape[0]

    def my_count_special(self, x_v, y_v, i, X_temp, y_temp):
        x_temp_iloc = X_temp.iloc[:, i]
        slice_y = y_temp[x_temp_iloc <= x_v]
        less = slice_y[slice_y.iloc[:, 0] == y_v].shape[0]
        slice_y = y_temp[x_temp_iloc > x_v]
        more = slice_y[slice_y.iloc[:, 0] == y_v].shape[0]
        return less, more

    def calc_continuous_mi(self, att_idx, X_temp, y_temp):
        x_temp_att_idx = X_temp.iloc[:, att_idx]
        diff_target_values = y_temp.iloc[:, 0].unique()
        x_size = X_temp.shape[0]
        min1 = x_temp_att_idx.min()
        max1 = x_temp_att_idx.max()
        t_arr = [self.calc_min_max(min1, max1, t) for t in np.unique(self.network.X.iloc[:, att_idx])]
        t_arr = filter(partial(is_not, None), t_arr)
        t_to_mi_dic = {t: self.calc_mi(att_idx, t, X_temp, y_temp, diff_target_values, x_size, x_temp_att_idx) for t in t_arr}
        t_to_mi_dic.pop(X_temp[X_temp.columns.values[att_idx]].max(), None)
        self_y_iloc_0 = self.y.iloc[:, 0]
        self_x_size = self.X.shape[0]
        for k in diff_target_values:
            self.targets_prob[k] = self.y[self_y_iloc_0 == k].shape[0] / self_x_size
        return t_to_mi_dic

    def significance_helper(self, mi):
        if mi > 0:
            return True
        return False

    def calc_min_max(self, min1, max1, threshold):
        if min1 <= threshold and threshold <= max1:
            return threshold
        return None

    def calc_mi(self, att_idx, threshold, X_temp, y_temp, diff_target_values, x_size, x_in_att_idx):
        mi = 0
        cond_x_less = X_temp[x_in_att_idx <= threshold].shape[0] / x_size
        cond_x_more = X_temp[x_in_att_idx > threshold].shape[0] / x_size
        df_size = self.n_samples
        for k in diff_target_values:
            count_x_y_less, count_x_y_more = self.my_count_special(threshold, k, att_idx, X_temp, y_temp)
            joint_less = count_x_y_less / df_size
            joint_more = count_x_y_more / df_size
            cond_less = count_x_y_less / x_size
            cond_more = count_x_y_more / x_size
            cond_y = y_temp[y_temp.iloc[:, 0] == k].shape[0] / x_size
            if count_x_y_more != 0:
                mi += joint_more * math.log(cond_more / (cond_x_more * cond_y), 2)
            if count_x_y_less != 0:
                mi += joint_less * math.log(cond_less / (cond_x_less * cond_y), 2)
        return mi

    def nominal_mi(self, att_idx):
        X = self.X
        y = self.y
        x_size = X.shape[0]
        mi = 0
        self_y_iloc = self.y.iloc[:, 0]
        for j in X.iloc[:, att_idx].unique():  # move on diff values for the att
            self_x_iloc = X.iloc[:, att_idx]
            cond_x = self.X[self_x_iloc == j].shape[0] / x_size
            for k in y.iloc[:, 0].unique():  # move on diff values for the target
                count_x_y = self.my_count(j, k, self_x_iloc)
                joint = count_x_y / self.n_samples
                cond = count_x_y / x_size
                cond_y = y[self_y_iloc == k].shape[0] / x_size
                self.targets_prob[k] = cond_y
                if count_x_y != 0:
                    mi += joint * math.log(cond / (cond_x * cond_y), 2)
        return mi

    def fit(self, X, y):
        self.X = X
        self.y = y
        y_iloc = y.iloc[:, 0]
        y_unique = y_iloc.unique()
        att_names = list(X.columns.values)
        net_sig = self.network.significance
        for i in self.targets:
            self.targets_prob[i] = 0
        for i in range(X.shape[1]):  # move on all attributes
            att_name_i = att_names[i]
            if att_name_i not in self.network.splitted_att and att_name_i in self.random_feature_arr:
                if 'nominal' in att_name_i:
                    mi = self.nominal_mi(i)
                else:
                    mi_dic = self.calc_continuous_mi(i, self.X, self.y)
                if self.network.preprune:
                    if 'nominal' in att_name_i:
                        G = mi * self.n_samples * 2 * math.log(2, math.e)
                        freedom_degree = (len(X.iloc[:, i].unique()) - 1) * (len(y.iloc[:, 0].unique()) - 1)
                        chi_from_table = chi2.isf(q=net_sig, df=freedom_degree)
                        if G > chi_from_table:
                            self.input_att_mi[att_name_i] = mi
                        else:
                            self.input_att_mi[att_name_i] = 0
                    else:
                        freedom_degree = len(y_unique) - 1
                        chi_from_table = chi2.isf(q=net_sig, df=freedom_degree)
                        self.input_att_mi[att_name_i] = {}
                        self.att_t_split[att_name_i] = {unique: False for unique in np.unique(self.network.X.iloc[:, i])}
                        x_iloc = X.iloc[:, i]
                        min1 = x_iloc.min()
                        max1 = x_iloc.max()
                        for threshold, mi in mi_dic.items():
                            G = self.calc_g_stat_continuous(self.y, threshold, att_name_i, min1, max1, y_iloc, y_unique)
                            if G > chi_from_table:
                                self.input_att_mi[att_name_i][threshold] = mi
                                self.att_t_split[att_name_i][threshold] = True
                            else:
                                self.input_att_mi[att_name_i][threshold] = 0
                                self.att_t_split[att_name_i][threshold] = False
                else:
                    if 'nominal' in att_name_i:
                        self.input_att_mi[att_name_i] = mi
                    else:
                        self.input_att_mi[att_name_i] = {}
                        self.att_t_split[att_name_i] = {unique: False for unique in np.unique(self.network.X.iloc[:, i])}
                        for threshold, mi in mi_dic.items():
                            if mi > 0:
                                self.input_att_mi[att_name_i][threshold] = mi
                                self.att_t_split[att_name_i][threshold] = True
                            else:
                                self.input_att_mi[att_name_i][threshold] = 0
                                self.att_t_split[att_name_i][threshold] = False
            else:
                self.input_att_mi[att_name_i] = 0
        return self.input_att_mi

    def split(self, att_splited, threshold_arr, att_random): #the threshold arr is already sorted from min to max, att_random is random feature arr
        self.feature_split = att_splited
        i = list(self.X.columns.values).index(att_splited)
        X_iloc = self.X.iloc[:, i]
        X = self.X
        y = self.y
        if 'nominal' in att_splited:
            for j in X_iloc.unique():  #good because we only split to nominal in the node == no home status 3
                slice_X = X[X_iloc == j]
                slice_y = y[X_iloc == j]
                temp_node = IFNNode(self.n_samples, self.targets, j, self.network, att_random)
                temp_node.fit(slice_X, slice_y)
                self.next_nodes.append(temp_node)
        else:
            count = 0
            unique_att_values = list(self.network.X.iloc[:, i].unique())
            unique_att_values.sort()
            min_t = self.network.X.iloc[:, i].min()
            max_t = threshold_arr[count]
            slice_X1 = X[min_t <= X_iloc]
            slice_X = slice_X1[slice_X1.iloc[:, i] <= max_t]
            slice_y = y[min_t <= X_iloc]
            slice_y = slice_y[slice_X1.iloc[:, i] <= max_t]
            if slice_X.shape[0] > 0:
                idx = unique_att_values.index(max_t)
                temp_node = IFNNode(self.n_samples, self.targets, str(min_t) + ' - ' + str(unique_att_values[idx+1]-0.000001), self.network, att_random)
                temp_node.fit(slice_X, slice_y)
                self.next_nodes.append(temp_node)
            min_t = max_t
            count += 1
            len_arr = len(threshold_arr)
            if count != len_arr:
                max_t = threshold_arr[count]
            for t in range(len(threshold_arr)-1):
                slice_X1 = X[min_t < X_iloc]
                slice_X = slice_X1[slice_X1.iloc[:, i] <= max_t]
                slice_y = y[min_t < X_iloc]
                slice_y = slice_y[slice_X1.iloc[:, i] <= max_t]
                if slice_X.shape[0] > 0:
                    idx_max = unique_att_values.index(max_t)
                    idx_min = unique_att_values.index(min_t)
                    temp_node = IFNNode(self.n_samples, self.targets, str(unique_att_values[idx_min+1]) + ' - ' + str(unique_att_values[idx_max+1]-0.000001), self.network, att_random)
                    temp_node.fit(slice_X, slice_y)
                    self.next_nodes.append(temp_node)
                min_t = max_t
                count += 1
                if count != len_arr:
                    max_t = threshold_arr[count]
            min_t = threshold_arr[len_arr-1]
            max_t = self.network.X.iloc[:, i].max()
            slice_X1 = X[min_t < X_iloc]
            slice_X = slice_X1[slice_X1.iloc[:, i] <= max_t]
            slice_y = y[min_t < X_iloc]
            slice_y = slice_y[slice_X1.iloc[:, i] <= max_t]
            if slice_X.shape[0] > 0:
                idx_min = unique_att_values.index(min_t)
                temp_node = IFNNode(self.n_samples, self.targets, str(unique_att_values[idx_min+1]) + ' - ' + str(max_t), self.network, att_random)
                temp_node.fit(slice_X, slice_y)
                self.next_nodes.append(temp_node)

    def global_discretization_in_node(self,threshold, att, threshold_arr, flag):
        myflag = False
        temp_t_arr = threshold_arr
        if flag:
            temp_t_arr.append(threshold)
        temp_t_arr.sort()
        t_idx = temp_t_arr.index(threshold)
        if t_idx == 0:
            min_interval = self.network.X[att].min()
            myflag = True
        else:
            min_interval = temp_t_arr[t_idx-1]
        if t_idx == len(temp_t_arr)-1:
            max_interval = self.network.X[att].max()
        else:
            max_interval = temp_t_arr[t_idx+1]
        i = list(self.X.columns.values).index(att)
        X_freedom, y_freedom = self.get_slices(min_interval, max_interval, att, myflag)
        freedom_degree = len(y_freedom.iloc[:, 0].unique()) - 1
        chi_from_table = chi2.isf(q=self.network.significance, df=freedom_degree)
        slice_X, slice_y = self.get_slices(min_interval, threshold, att, myflag)
        slice_y_iloc = slice_y.iloc[:, 0]
        slice_y_unique = slice_y_iloc.unique()
        slice_x_iloc = slice_X.iloc[:, i]
        min1 = slice_x_iloc.min()
        max1 = slice_x_iloc.max()
        self.left_interval = self.calc_continuous_mi(list(self.X.columns.values).index(att), slice_X, slice_y)
        if self.network.preprune:
            self.left_interval = dict([(threshold_unique_val, self.chi_stat(val, chi_from_table, threshold_unique_val, att, slice_y, min1, max1, slice_y_iloc, slice_y_unique)) for threshold_unique_val, val in self.left_interval.items()])
        slice_X, slice_y = self.get_slices(threshold, max_interval, att, False)
        slice_y_iloc = slice_y.iloc[:, 0]
        slice_y_unique = slice_y_iloc.unique()
        slice_x_iloc = slice_X.iloc[:, i]
        min1 = slice_x_iloc.min()
        max1 = slice_x_iloc.max()
        self.right_interval = self.calc_continuous_mi(list(self.X.columns.values).index(att), slice_X, slice_y)
        if self.network.preprune:
            self.right_interval = dict([(threshold_unique_val, self.chi_stat(val, chi_from_table,threshold_unique_val,att, slice_y, min1, max1, slice_y_iloc, slice_y_unique)) for threshold_unique_val, val in self.right_interval.items()])

    def get_slices(self, min_interval, max_interval, att, myflag):
        i = list(self.X.columns.values).index(att)
        X = self.X
        y = self.y
        self_x_iloc = self.X.iloc[:, i]
        if myflag:
            slice_X1 = X[min_interval <= self_x_iloc]
            slice_X = slice_X1[slice_X1.iloc[:, i] <= max_interval]
            slice_y = y[min_interval <= self_x_iloc]
            slice_y = slice_y[slice_X1.iloc[:, i] <= max_interval]
        else:
            slice_X1 = X[min_interval < self_x_iloc]
            slice_X = slice_X1[slice_X1.iloc[:, i] <= max_interval]
            slice_y = y[min_interval < self_x_iloc]
            slice_y = slice_y[slice_X1.iloc[:, i] <= max_interval]
        return slice_X, slice_y

    def chi_stat(self, mi, chi_from_table, threshold, att, slice_y, min1, max1, slice_y_iloc, slice_y_unique):
        G = self.calc_g_stat_continuous(slice_y, threshold, att, min1, max1, slice_y_iloc, slice_y_unique)
        if G > chi_from_table:
            self.att_t_split[att][threshold] = True
            return mi
        else:
            self.att_t_split[att][threshold] = False
            return 0

    def reduce_thresholds(self, t_arr, att):
        res = []
        for t in t_arr:
            if self.att_t_split[att][t]:
                res = t_arr
                res.sort()
                return res
        return res

    def calc_g_stat_continuous(self, slice_y, t, att, min1, max1, slice_y_iloc, slice_y_unique):
        sum = 0
        slice_X_less_equal_t, slice_y_less_equal_t = self.get_slices(min1, t, att, True)
        slice_X_more_t, slice_y_more_t = self.get_slices(t, max1, att, False)
        slice_y_less_equal_t_iloc = slice_y_less_equal_t.iloc[:, 0]
        slice_y_more_t_iloc = slice_y_more_t.iloc[:, 0]
        for target_value in slice_y_unique:
            num = slice_y_less_equal_t[slice_y_less_equal_t_iloc == target_value].shape[0]
            num_father = slice_y[slice_y_iloc == target_value].shape[0] / slice_y.shape[0]
            if num > 0:
                sum += num * math.log(num/(num_father * slice_y_less_equal_t.shape[0]), math.e)
            num = slice_y_more_t[slice_y_more_t_iloc == target_value].shape[0]
            if num > 0:
                sum += num * math.log(num / (num_father * slice_y_more_t.shape[0]), math.e)
        return 2*sum
