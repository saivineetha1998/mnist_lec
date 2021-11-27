from sklearn import datasets, svm, metrics, tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import os
from joblib import load, dump

def preprocess(images, rescale_factor):
    image_resized = []
    for i in range(images.shape[0]):
        image_resized.append(resize(images[i],(rescale_factor,rescale_factor)))
    return image_resized

def create_splits(images, targets, test_size, valid_size):
    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=test_size + valid_size, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=valid_size/(test_size + valid_size), shuffle=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test    

def test(X_valid, y_valid, clf):
    predicted_valid = clf.predict(X_valid)
    acc_valid = accuracy_score(predicted_valid, y_valid)
    f1_valid = f1_score(predicted_valid, y_valid , average="macro")
    return {'acc':acc_valid,'f1':f1_valid}

def run_classification_experiment(classifier, X_train, X_valid, X_test, y_train, y_valid, y_test, gamma_idx, output_folder):
    if classifier == svm.SVC:
        clf = classifier(gamma=gamma_idx)
        clf.fit(X_train, y_train)
        metric_dic = test(X_valid, y_valid, clf)
        if metric_dic['acc'] < 0.11:
            print("Skipping for gamma {}".format(gamma_idx))
            return None
        if not (os.path.exists(output_folder)):
            os.mkdir(output_folder)
            #print("NO")
        else:
            #print("YES")
            pass
         
        dump(clf, os.path.join(output_folder,"model.joblib"))
        return metric_dic
    elif classifier == tree.DecisionTreeClassifier:
        clf = classifier(max_depth=gamma_idx)
        clf.fit(X_train, y_train)
        metric_dic = test(X_valid, y_valid, clf)
        if not (os.path.exists(output_folder)):
            os.mkdir(output_folder)
            #print("NO")
        else:
            #print("YES")
            pass
         
        dump(clf, os.path.join(output_folder,"model.joblib"))
        return metric_dic

