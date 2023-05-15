from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def apply_crossvalidation(model, X_train, Y_train, k=10):

    # create a k-fold cross-validation iterator
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # perform k-fold cross-validation and compute accuracy
    scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
    # print the average accuracy score and its standard deviation
    print('Accuracy: {} +/- {}'.format(scores.mean(), scores.std()))

    # perform k-fold cross-validation and compute F1-score
    scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='f1_weighted')
    # print the average F1-score and its standard deviation
    print('F1-score: {} +/- {}'.format(scores.mean(), scores.std()))

    # # perform k-fold cross-validation and compute AUC
    # scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='roc_auc')
    # # print the average AUC score and its standard deviation
    # print('AUC: {} +/- {}'.format(scores.mean(), scores.std()))

def Evaluate(model, X_test, Y_test):
    
    # predict the class labels for the test set
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(Y_test, y_pred)

    # calculate the precision
    precision = precision_score(Y_test, y_pred, average='weighted')

    # calculate the recall
    recall = recall_score(Y_test, y_pred, average='weighted')

    # calculate the F1 score
    f1 = f1_score(Y_test, y_pred, average='weighted')

    # calculate the confusion matrix
    cm = confusion_matrix(Y_test, y_pred)

    # print the results
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Confusion matrix:\n', cm)