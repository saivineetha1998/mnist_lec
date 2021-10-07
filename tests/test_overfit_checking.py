
import test_model_write
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


def test_small_data_overfit_checking():
    digits = datasets.load_digits()
    data, digits = preprocessing(digits)
    # sample_size = 500
    x_tr, x_ts, y_tr, y_ts = train_test_split(data, digits.target, test_size=0.5)
    # sampled_data, sampled_digits = data[sample_size], digits[sample_size]

    

    # output_folder = "testing_models"
    # os.mkdir(output_folder)
    fname = 'model_overfit_checking.sav'
    train_metrics = run_classification_experiment(x_tr, y_tr, x_tr, y_tr)

    assert train_metrics['acc']  > 0.80

    assert train_metrics['f1'] > 0.80    


def run_classification_experiment(x_train, y_train, x_val, y_val):
    
    metrics_tr = {}
    
    
    clf = svm.SVC(gamma=0.001)
    clf.fit(x_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(x_val)
    
    # pickle.dump(clf, open(expected_model_file, 'wb'))
    if (accuracy_score(y_val, predicted))<0.11:
        print("Skipping")
        # continue
    else:
        pickle.dump(clf, open('model_test_overfitting', 'wb'))
        metrics_tr['acc'] = accuracy_score(y_val, predicted)
        metrics_tr['f1'] = f1_score(y_val, predicted, average='weighted')
        return metrics_tr

 




