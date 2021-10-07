

#from mnist_lec.mnist_lec.mnist_lec.valmetrics import X_test, X_train, X_val
import matplotlib.pyplot as plt
import os
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

# Preprocessing
def preprocessing(digits):
    #digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits


# Create Splits
def create_splits(data, digits, test_size, val_size):
    X_tr, X_test, y_tr,  y_test = train_test_split(
            data, digits.target, test_size=test_size, shuffle=False)
    X_train, X_val, y_train,  y_val = train_test_split(
            X_tr, y_tr,  test_size=val_size, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test




def test_model_writing():
    digits = datasets.load_digits()
    # output_folder = "testing_models"
    # os.mkdir(output_folder)
    fname = 'model_test_writing.sav'
    run_classification_experiment(digits, fname)
    assert os.path.isfile(fname)
    
    


def run_classification_experiment(dta, expected_model_file):
    data, digits = preprocessing(dta)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(data, digits, 0.15, 0.15)
    
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    
    pickle.dump(clf, open(expected_model_file, 'wb'))
 



