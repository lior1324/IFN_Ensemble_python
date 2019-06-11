from IFNNode import IFNNode
from IFNLayer import IFNLayer
import numpy as np
import operator


class InfoFuzzyNetwork:
    def __init__(self, max_depth=None, min_samples_split=1, max_feature=-1, significance=0.001, preprune=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_feature = max_feature
        self.n_classes_ = 0
        self.n_features_ = 0
        self.X = None
        self.y = None
        self.significance = significance
        self.preprune = preprune
        self.layerArr = [] #hold our layers
        self.splitted_att = [] #show us which att splitted in the past

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_features_ = X.shape[1]
        n_feature = self.n_features_
        att_random = self.get_random_att()
        startNode = IFNNode(X.shape[0], y.iloc[:, 0].unique(), None, self, att_random)
        startNode.fit(X, y)
        layer0 = IFNLayer(0, None, self, att_random)
        layer0.NodesArr.append(startNode)
        layer0.fit()
        #print(layer0.splitBy, layer0.splitBy_t_arr)
        self.layerArr.append(layer0)
        if self.max_depth is None:
            for i in range(n_feature):
                att_random = self.get_random_att()
                temp_layer = self.layerArr[len(self.layerArr) - 1].buildNewLayer(att_random)
                if temp_layer is None:
                    break
                self.layerArr.append(temp_layer)
                #print(temp_layer.splitBy, temp_layer.splitBy_t_arr)
        else:
            for i in range(self.max_depth):
                att_random = self.get_random_att()
                temp_layer = self.layerArr[len(self.layerArr) - 1].buildNewLayer(att_random)
                if temp_layer is None:
                    break
                self.layerArr.append(temp_layer)
                #print(temp_layer.splitBy, temp_layer.splitBy_t_arr)

    def predict(self, X):
        preds = []
        for sample_index, sample in X.iterrows():
            predNode = self.layerArr[0].NodesArr[0]
            flag = False
            while len(predNode.next_nodes) > 0 and flag is False:
                flag1 = False
                feature_split_name = predNode.feature_split
                if 'nominal' in feature_split_name:
                    for node in predNode.next_nodes:
                        str1 = node.prev_node_feature_split_value
                        str2 = sample[feature_split_name]
                        if str1 == str2:
                            predNode = node
                            flag1 = True
                            break
                    if flag1 is False:
                        flag = True
                else:
                    for node in predNode.next_nodes:
                        min_max_arr = node.prev_node_feature_split_value.split('-')
                        min_interval = float(min_max_arr[0])
                        max_interval = float(min_max_arr[1])
                        current = round(sample[feature_split_name], 4)
                        if current <= max_interval and current >= min_interval:
                            predNode = node
                            flag1 = True
                            break
                    if flag1 is False:
                        flag = True
            preds.append(max(predNode.targets_prob.items(), key=operator.itemgetter(1))[0])
        return np.array(preds)

    def print_network(self):
        for layer in self.layerArr:
            print("layer" + str(layer.layerNum))
            print("layer is splitted by "+ str(layer.splitBy) + " and threshold arr is "+ str(layer.splitBy_t_arr))
            for node in layer.NodesArr:
                print("this node have "+ str(len(node.next_nodes))+" next nodes")
                for next_node in node.next_nodes:
                    print("the node is come from "+ str(next_node.prev_node_feature_split_value))

    def get_random_att(self):
        if self.max_feature == -1:
            return list(self.X.columns.values)
        att_idx = np.random.choice(a=len(list(self.X.columns.values)), size=self.max_feature, replace=False)
        res = list(np.array(list(self.X.columns.values))[att_idx])
        return res