import os
import numpy as np


def remove(path):
    if os.path.exists(path):
        os.remove(path)
        print('removed ', path)
    else:
        print('does not exist ', path)


def get_clf_report(y_true, y_pred, y_pred_prob=None):

    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import roc_auc_score

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    report = classification_report(y_true, y_pred)
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print('Accuracy : ', acc)

    num_classes = len(np.unique(y_true))
    if num_classes == 2:
        prec = precision_score(y_true, y_pred)
        print('Precision : ', prec)
        recall = recall_score(y_true, y_pred)
        print('Recall : ', prec)
        auc = roc_auc_score(y_true, y_pred_prob, average='macro')
        print('AUC Score : ', auc)
        return cm, report, acc, prec, recall, auc

    return cm, report, acc


def plot_model_history(log_path, save_path, addn_info=''):

    import pylab as plt
    import pandas as pd

    title = log_path.split('/')[-1]
    title += str(addn_info)

    df = pd.read_csv(log_path)
    y = df['acc']
    x = range(len(y))
    plt.plot(x, y, '-b', label='train_acc')

    y = df['val_acc']
    x = range(len(y))
    plt.plot(x, y, '-r', label='valid_acc')

    y = df['loss']
    x = range(len(y))
    plt.plot(x, y, '-b', label='train_loss')

    y = df['val_loss']
    x = range(len(y))
    plt.plot(x, y, '-r', label='val_loss')

    plt.title(title)
    plt.legend(loc='upper left')
    plt.ylim(-1.5, 2.0)

    plt.savefig(save_path)
    plt.show()


def normalize(arr, low=0, high=255):
    diff = high - low
    if diff > 0:
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr = arr * diff
    return arr
