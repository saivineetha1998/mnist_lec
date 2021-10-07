

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#def metrics(ratio):



digits = datasets.load_digits()

#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, label in zip(axes, digits.images, digits.target):
#    ax.set_axis_off()
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    ax.set_title('Training: %i' % label)



# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
acc = {}
#f1 = []
# Create a classifier: a support vector classifier
gma = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.06, 0.10, 0.15, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
print("On test data")
print("Gamma -> Accuracy -> F1 score")
splits  = [(0.15, 0.15), (0.20, 0.10)]
# Checking for different values of hyperparameter gamma
for tstSize, valSize in splits:

    for i in gma:
        clf = svm.SVC(gamma=i)
    

        # Split data into 50% train and 50% test subsets

        X_tr, X_test, y_tr,  y_test = train_test_split(
            data, digits.target, test_size=tstSize, shuffle=False)
        X_train, X_val, y_train,  y_val = train_test_split(
            X_tr, y_tr,  test_size=valSize, shuffle=False)
    
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

   

 #   _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
 #  for ax, image, prediction in zip(axes, X_test, predicted):
 #       ax.set_axis_off()
 #      image = image.reshape(8, 8)
 #       ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
 #       ax.set_title(f'Prediction: {prediction}')

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    # print(f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n")

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()
        #acc.append(accuracy_score(y_test, predicted))
        acc[(i, valSize, tstSize)] = (accuracy_score(y_test, predicted))
        #f1.append(f1_score(y_test, predicted, average='weighted'))
        print(valSize,  tstSize, ": \t", i, " -> ", accuracy_score(y_test, predicted), " ->", f1_score(y_test, predicted, average='weighted'))
    
        # Learn the digits on the train subset
    

    # Predict the value of the digit on the test subset
        #predicted_val = clf.predict(X_val)

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

#    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#    for ax, image, prediction in zip(axes, X_test, predicted):
#        ax.set_axis_off()
#        image = image.reshape(8, 8)
#        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#        ax.set_title(f'Prediction: {prediction}')

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    # print(f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n")

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()
    
    #print("val: \t", i, " -> ", accuracy_score(y_val, predicted_val), " ->", f1_score(y_val, predicted_val, average='weighted'))
    # Predict the value of the digit on the test subset
    #predicted_train = clf.predict(X_train)

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    # print(f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n")

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()
    
    #print("train \t", i, " -> ", accuracy_score(y_train, predicted_train), " ->", f1_score(y_train, predicted_train, average='weighted'))
max_acc = max(acc.values())
print(max_acc)
for key, value in acc.items():
      if value == max_acc:
         print(key)
#max_f1 = max(f1)
#print(max_f1)
 