from InfoFuzzyNetwork import InfoFuzzyNetwork
from IFNEnsemble import IFNForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def runSingleIFN():
    SIGNIFICANCE = 0.001
    PREPRUNE = True
    CONFUSION_MATRIX = True
    PRINT_NETWORK = True
    DATA_PATH = 'credit_approval.csv' #for Liver use 'Liver.csv'

    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    ifn = InfoFuzzyNetwork(significance=SIGNIFICANCE, preprune=PREPRUNE)
    ifn.fit(X, y)
    preds = ifn.predict(X)
    if PRINT_NETWORK:
        ifn.print_network()
    if CONFUSION_MATRIX:
        print(confusion_matrix(y, preds))


def runSingleIFN_TrainTest():
    SIGNIFICANCE = 0.001
    PREPRUNE = True
    CONFUSION_MATRIX = True
    PRINT_NETWORK = True
    TEST_SIZE = 0.2
    DATA_PATH = 'credit_approval.csv' #for Liver use 'Liver.csv'

    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    ifn = InfoFuzzyNetwork(significance=SIGNIFICANCE, preprune=PREPRUNE)
    ifn.fit(X_train, y_train)
    preds = ifn.predict(X_test)
    if PRINT_NETWORK:
        ifn.print_network()
    if CONFUSION_MATRIX:
        print(confusion_matrix(y_test, preds))


def runIFNForest():
    SIGNIFICANCE = 0.001
    PREPRUNE = True
    CONFUSION_MATRIX = True
    BOOTSTRAP = True
    MAX_FEATURES = 4 #for Liver dataset use 3
    N_ESTIMATORS = 10
    DATA_PATH = 'credit_approval.csv' #for Liver use 'Liver.csv'

    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    ifn_forest = IFNForestClassifier(significance=SIGNIFICANCE, preprune=PREPRUNE, bootstrap=BOOTSTRAP, max_features=MAX_FEATURES, n_estimators=N_ESTIMATORS)
    ifn_forest.fit(X, y)
    preds = ifn_forest.predict(X)
    if CONFUSION_MATRIX:
        print(confusion_matrix(y, preds))


def runIFNForest_TrainTest():
    SIGNIFICANCE = 0.001
    PREPRUNE = True
    CONFUSION_MATRIX = True
    BOOTSTRAP = True
    MAX_FEATURES = 4 #for Liver dataset use 3
    N_ESTIMATORS = 10
    DATA_PATH = 'credit_approval.csv' #for Liver use 'Liver.csv'
    TEST_SIZE = 0.2

    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    ifn_forest = IFNForestClassifier(significance=SIGNIFICANCE, preprune=PREPRUNE, bootstrap=BOOTSTRAP, max_features=MAX_FEATURES, n_estimators=N_ESTIMATORS)
    ifn_forest.fit(X_train, y_train)
    preds = ifn_forest.predict(X_test)
    if CONFUSION_MATRIX:
        print(confusion_matrix(y_test, preds))


#runSingleIFN()
#runSingleIFN_TrainTest()
#runIFNForest()
runIFNForest_TrainTest()

