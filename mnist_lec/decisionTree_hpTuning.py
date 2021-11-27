

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





def run_classification_experiment(classifier, x_train, y_train, x_val, y_val):
    
    metrics_tr = {}
    
    
    #clf = svm.SVC(gamma=0.001)
    classifier.fit(x_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = classifier.predict(x_val)
    
    # pickle.dump(clf, open(expected_model_file, 'wb'))
    # if (accuracy_score(y_val, predicted))<0.11:
    #     print("Skipping")
    #     # continue
    # else:
    #     # pickle.dump(classifier, open('model_test_overfitting', 'wb'))
    metrics_tr['acc'] = accuracy_score(y_val, predicted)
    metrics_tr['f1'] = f1_score(y_val, predicted, average='weighted')
    return metrics_tr


digits = datasets.load_digits()
data, digits = preprocessing(digits)
# sample_size = 500
gma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.06, 0.10, 0.15, 0.2]

x_tr, x_ts, y_tr, y_ts = train_test_split(data, digits.target, test_size=0.5)
# sampled_data, sampled_digits = data[sample_size], digits[sample_size]
# output_folder = "testing_models"
# os.mkdir(output_folder)
#fname = 'model_overfit_checking.sav'
acc_svc = {}
f1_svc = {}
for i in gma:
    # print(i)
    clf_svc = svm.SVC(gamma=i)
    train_metrics_svc = run_classification_experiment(clf_svc, x_tr, y_tr, x_ts, y_ts)
    
    acc_svc[i] = train_metrics_svc['acc']
    f1_svc[i] = train_metrics_svc['f1']
    # print(train_metrics_svc['acc']," ", train_metrics_svc['f1'])
max_acc = max(acc_svc.values())
# print(max_acc)
for key, value in acc_svc.items():
    if value == max_acc:
        opt_gamma = key
print(opt_gamma, acc_svc[opt_gamma], f1_svc[opt_gamma])

depth_tree = [2, 3, 4, 5, 6, 7]

x_tr, x_ts, y_tr, y_ts = train_test_split(data, digits.target, test_size=0.5)
# sampled_data, sampled_digits = data[sample_size], digits[sample_size]
# output_folder = "testing_models"
# os.mkdir(output_folder)
#fname = 'model_overfit_checking.sav'
acc_dt = {}
f1_dt = {}
for i in depth_tree:
    # print(i)
    clf_dt = tree.DecisionTreeClassifier(max_depth = i)
    train_metrics_dt = run_classification_experiment(clf_dt, x_tr, y_tr, x_ts, y_ts)
    
    acc_dt[i] = train_metrics_dt['acc']
    f1_dt[i] = train_metrics_dt['f1']
    # print(train_metrics_svc['acc']," ", train_metrics_svc['f1'])
max_acc_dt = max(acc_dt.values())
# print(max_acc_dt)
for key, value in acc_dt.items():
    if value == max_acc_dt:
        opt_depth = key
print(opt_depth, acc_dt[opt_depth], f1_dt[opt_depth])



# clf = tree.DecisionTreeClassifier()
# train_metrics = run_classification_experiment(clf, x_tr, y_tr, x_ts, y_ts)

# assert train_metrics['acc']  > 0.80

# assert train_metrics['f1'] > 0.80    

# print("Accuracy ->", train_metrics['acc'], "F1-score", train_metrics['f1'])


