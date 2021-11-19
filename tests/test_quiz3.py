import math

#from mnist_lec.mnist_lec.mnist_lec.valmetrics import X_test, X_train, X_val
import matplotlib.pyplot as plt
import os
from sklearn import tree
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

test_size = 0.15
val_size = 0.15
depth_tree = 7
gamma = 0.001


X_train, X_test_valid, y_train, y_test_valid = train_test_split(
        data, digits.target, test_size=test_size + val_size, shuffle=False
    )

X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_valid,
        y_test_valid,
        test_size=val_size / (test_size + val_size),
        shuffle=False,
    )

svm_best_model_path = '/home/saivineetha/mnist_lec/mnist_lec/mnist_lec/svm_dt_models/bst_nodel_svm.sav'
dt_best_model_path = '/home/saivineetha/mnist_lec/mnist_lec/mnist_lec/svm_dt_models/bst_model_dt.sav'

def load(model_path):
    load_file = open(model_path, "rb")
    model = pickle.load(load_file)
    return model



svm_model =  load(svm_best_model_path)
dt_model = load(dt_best_model_path)

prediction = svm_model.predict(X_test)
print(prediction)


prediction_dt = dt_model.predict(X_test)
print(prediction_dt)

def test_digit_svm_0():
    # prediction = svm_model.predict(X_test)
    # print(prediction)
    assert prediction[1]==0


def test_digit_svm_1():
    # prediction = svm_model.predict(X_test)
    assert prediction[7]==1

def test_digit_svm_2():
    # prediction = svm_model.predict(X_test)
    assert prediction[13]==2

def test_digit_svm_3():
    # prediction = svm_model.predict(X_test)
    assert prediction[3]==3

def test_digit_svm_4():
    # prediction = svm_model.predict(X_test)
    assert prediction[0]==4

def test_digit_svm_5():
    # prediction = svm_model.predict(X_test)
    assert prediction[2]==5

def test_digit_svm_6():
    # prediction = svm_model.predict(X_test)
    assert prediction[4]==6

def test_digit_svm_7():
    # prediction = svm_model.predict(X_test)
    assert prediction[8]==7

def test_digit_svm_8():
    # prediction = svm_model.predict(X_test)
    assert prediction[14]==8

def test_digit_svm_9():
    # prediction = svm_model.predict(X_test)
    assert prediction[5]==9

# Decision Tree

def test_digit_dt_0():

    
    assert prediction_dt[1]==0

def test_digit_dt_1():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[7]==1

def test_digit_dt_2():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[15]==2

def test_digit_dt_3():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[3]==3

def test_digit_dt_4():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[0]==4

def test_digit_dt_5():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[2]==5

def test_digit_dt_6():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[4]==6

def test_digit_dt_7():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[12]==7

def test_digit_dt_8():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[8]==8

def test_digit_dt_9():
    # prediction = dt_model.predict(X_test)
    assert prediction_dt[5]==9