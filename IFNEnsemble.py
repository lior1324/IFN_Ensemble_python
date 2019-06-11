import numpy as np
from InfoFuzzyNetwork import InfoFuzzyNetwork
from sklearn.metrics import confusion_matrix
import operator


class IFNForestClassifier:
    def __init__(self, bootstrap=True, max_depth=None, max_features=-1, n_estimators=10, n_jobs=1, significance=0.1, preprune=False):
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.X = None
        self.y = None
        self.clf_arr = []
        self.significance = significance
        self.preprune = preprune
        self.oob_score = 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        for i in range(self.n_estimators):
            clf = InfoFuzzyNetwork(max_depth=self.max_depth, max_feature=self.max_features, significance=self.significance, preprune=self.preprune)
            X_train, y_train, X_test, y_test = self.get_bootstrap(X, y)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            my_result = confusion_matrix(y_test, preds)
            sum = 0
            for j in range(np.sqrt(my_result.size).astype(int)):
                sum = sum + my_result[j][j]
            oob_score = sum / preds.size
            self.oob_score = self.oob_score + (1 - oob_score)
            self.clf_arr.append(clf)
            #print("done ifn no. "+str(i))
        self.oob_score = self.oob_score / self.n_estimators

    def get_bootstrap(self, X, y):
        idx = np.random.choice(a=X.shape[0], size=X.shape[0], replace=True)
        X_bootstrap = X.iloc[idx, :]
        y_bootstrap = y.iloc[idx, :]
        X_oob = X[~X.index.isin(X_bootstrap.index)]
        y_oob = y[~y.index.isin(y_bootstrap.index)]
        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def predict(self, X):
        clf_preds = []
        res = []
        for clf in self.clf_arr:
            clf_preds.append(clf.predict(X))
        for sample_index, sample in X.iterrows():
            dic = {val: 0 for val in self.y.iloc[:, 0].unique()}
            for i in range(len(self.clf_arr)):
                dic[clf_preds[i][sample_index]] += 1
            res.append(max(dic.items(), key=operator.itemgetter(1))[0])
        return np.array(res)

