# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn import tree
import pickle
import statistics
from joblib import dump, load
import matplotlib.pyplot as plt


digits = datasets.load_digits()



n_samples = len(digits.images)
data_ = digits.images.reshape((n_samples, -1))

X_train, X_tst, y_train, y_tst = train_test_split(data_, digits.target, test_size=0.20 ,shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_tst, y_tst, test_size=0.10, shuffle=True)

acc_svm = {}
acc_dt = {}
f1_svm = {}
f1_dt = {}



def svm_classifier(ratio):
    # print("SVM")
    gma = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.06, 0.10, 0.15, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
    if ratio!=100:
        X_train_fnl, X_rem, y_train_fnl, y_rem = train_test_split(X_train, y_train, test_size = 1 - (ratio/100), shuffle = True)
    else:
        X_train_fnl, y_train_fnl = X_train, y_train
    for i in gma:
        clf = svm.SVC(gamma=i)
        clf.fit(X_train_fnl, y_train_fnl)
        predicted = clf.predict(X_val)
        acc_svm[i] = round(accuracy_score(y_val, predicted), 3)
        f1_svm[i] = round(f1_score(y_val, predicted, average='macro'), 3)

    max_acc_svm = max(acc_svm.values())
    # print(max_acc_svm)

    # for key, value in acc_svm.items():
        # if value == max_acc_svm:
            # print(key)

    max_f1_svm = max(f1_svm.values())
    # print(max_f1_svm)

    # for key, value in f1_svm.items() mn:
    #     if value == max_f1_svm:
    #         print(key[1])

    return max_acc_svm, max_f1_svm


def dt_classifier(ratio):
    # print("dt")
    depth_tree = [2, 3, 4, 5, 6, 7]
    if ratio!=100:
        X_train_fnl, X_rem, y_train_fnl, y_rem = train_test_split(X_train, y_train, test_size = 1 - (ratio/100), shuffle = True)
    else:
        X_train_fnl, y_train_fnl = X_train, y_train
    for i in depth_tree:
        clf = tree.DecisionTreeClassifier(max_depth = i)
        clf.fit(X_train_fnl, y_train_fnl)
        predicted = clf.predict(X_val)
        acc_dt[i] = round(accuracy_score(y_val, predicted), 3)
        f1_dt[i] = round(f1_score(y_val, predicted, average='macro'), 3)

    max_acc_dt = max(acc_dt.values())
    # print(max_acc_svm)

    # for key, value in acc_svm.items():
        # if value == max_acc_svm:
            # print(key)

    max_f1_dt = max(f1_dt.values())
    # print(max_f1_svm)

    # for key, value in f1_svm.items():
    #     if value == max_f1_svm:
    #         print(key[1])

    return max_acc_dt, max_f1_dt


split_ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
acc_svm_best = []
f1_svm_best = []
acc_dt_best = []
f1_dt_best = []


for i in split_ratio:
    acc, f1 = svm_classifier(i)
    acc_svm_best.append(acc)
    f1_svm_best.append(f1)
    acc1, f11 = dt_classifier(i)
    acc_dt_best.append(acc1)
    f1_dt_best.append(f11)




# print(acc_dt_best)
# print(acc_svm_best)
# print(f1_dt_best)
# print(f1_svm_best)
plt.plot(split_ratio , f1_svm_best , color='r',label='SVM Classifier')
plt.plot(split_ratio , f1_dt_best , color='g',label='Decision Tree Classifier')
plt.xlabel('Train Split ratio')
plt.ylabel('F1 score')
plt.legend()
plt.title('F1 Score V/s Train Split ratio')
plt.show()

plt.savefig('f1_train_split.png')


plt.plot(split_ratio , acc_svm_best , color='r',label='SVM Classifier')
plt.plot(split_ratio , acc_dt_best , color='g',label='Decision Tree Classifier')
plt.xlabel('Train Split ratio')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy V/s Train Split ratio')
plt.show()

plt.savefig('acc_train_split.png')
