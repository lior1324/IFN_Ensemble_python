import numpy as np


class IFNLayer:
    def __init__(self, layerNum, prevLayer, network, random_features_arr):
        self.layerNum = layerNum
        self.prevLayer = prevLayer
        self.inputAttributesMI = {}
        self.nextLayer = None
        self.NodesArr = []
        self.splitBy = None
        self.network = network
        self.splitBy_t_arr = None
        self.inputAttributesMI_t = {}
        self.random_features_arr = random_features_arr

    def fit(self):
        for att in list(self.NodesArr[0].X.columns.values):
            if att not in self.network.splitted_att and att in self.random_features_arr:
                if 'nominal' in att:
                    sum_mi = 0
                    for node in self.NodesArr:
                        sum_mi += node.input_att_mi[att]
                    self.inputAttributesMI[att] = sum_mi
                else:
                    potential_t_in_network = np.unique(self.network.X[att])
                    self.inputAttributesMI[att] = {i: self.fit_helper_continuous(att, i) for i in potential_t_in_network}
                    key = max(self.inputAttributesMI[att], key=self.inputAttributesMI[att].get)
                    val = self.inputAttributesMI[att][key]
                    self.inputAttributesMI[att] = val
                    self.inputAttributesMI_t[att] = []
                    self.inputAttributesMI_t[att].append(key)
                    if self.inputAttributesMI[att] > 0:
                        self.recursive_descretization(key, att, False)
            else:
                self.inputAttributesMI[att] = 0
        max_mi = 0
        for key, val in self.inputAttributesMI.items():
            if val > max_mi and key not in self.network.splitted_att:
                max_mi = val
                self.splitBy = key
                if key in self.inputAttributesMI_t.keys():
                    self.splitBy_t_arr = self.inputAttributesMI_t[key]
                else:
                    self.splitBy_t_arr = None
        if max_mi == 0:
            self.splitBy = None
        self.network.splitted_att.append(self.splitBy)

    def fit_helper_continuous(self, att, i):
        sum_mi = 0
        for node in self.NodesArr:
            if i in node.input_att_mi[att]:
                sum_mi += node.input_att_mi[att][i]
        return sum_mi

    def recursive_descretization(self, threshold, att, flag=True):
        right_dic = {}
        left_dic = {}
        for node in self.NodesArr:
            node.global_discretization_in_node(threshold, att, self.inputAttributesMI_t[att], flag)
            if node.right_interval is not None:
                for key, val in node.right_interval.items():
                    if key in right_dic.keys():
                        right_dic[key] += val
                    else:
                        right_dic[key] = val
            if node.left_interval is not None:
                for key, val in node.left_interval.items():
                    if key in left_dic.keys():
                        left_dic[key] += val
                    else:
                        left_dic[key] = val
        if len(right_dic) != 0:
            key_right = max(right_dic, key=right_dic.get)
            val_right = right_dic[key_right]
            if val_right > 0:
                self.inputAttributesMI_t[att].append(key_right)
                self.inputAttributesMI[att] += val_right
                self.recursive_descretization(key_right, att, False)
        if len(left_dic) != 0:
            key_left = max(left_dic, key=left_dic.get)
            val_left = left_dic[key_left]
            if val_left > 0:
                self.inputAttributesMI_t[att].append(key_left)
                self.inputAttributesMI[att] += val_left
                self.recursive_descretization(key_left, att, False)

    def buildNewLayer(self, att_random):
        if self.splitBy is None:
            return None
        new_layer = IFNLayer(self.layerNum + 1, self, self.network, att_random)
        for node in self.NodesArr:
            split_by = self.splitBy
            if 'nominal' in split_by:
                if node.input_att_mi[split_by] != 0:
                    node.split(split_by, 0, new_layer.random_features_arr)
                    for nodeS in node.next_nodes:
                            new_layer.NodesArr.append(nodeS)
            else:
                t_arr_to_node = node.reduce_thresholds(self.inputAttributesMI_t[split_by], split_by)
                if len(t_arr_to_node) > 0:
                    node.split(split_by, t_arr_to_node, new_layer.random_features_arr)
                    for nodeS in node.next_nodes:
                            new_layer.NodesArr.append(nodeS)
        new_layer.fit()
        return new_layer
