from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# import necessary libraries and configurations

history = np.load(os.path.join(sys.path[0], "history.npy"),allow_pickle='TRUE').item()

plt.plot(history['loss'], label='Train Set Loss')
plt.plot(history['val_loss'], label='Validation Set Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss and Validation Loss History')
plt.legend()
plt.savefig(os.path.join(sys.path[0], "loss_history.png"))
plt.show()

# plot the loss

plt.plot(history['accuracy'], label='Train Set Accuracy')
plt.plot(history['val_accuracy'], label='Validation Set Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy and Validation Accuracy History')
plt.legend()
plt.savefig(os.path.join(sys.path[0], "accuracy_history.png"))
plt.show()

# plot the accuracy

def roc_auc(test_predict, test_labels):
  fpr, tpr, _ = roc_curve(test_labels, test_predict)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(sys.path[0], "roc_auc.png"))
  plt.show()

  # plot the AUC & ROC

def precision_recall(test_labels, test_predict):
  from object_detection.utils.metrics import compute_precision_recall
  precision, recall = compute_precision_recall(test_predict, test_labels,728)
  plt.figure()
  plt.step(recall, precision, where='post' )
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision Recall Curve')
  plt.xlim((0, 1))
  plt.ylim((0, 1))
  plt.savefig(os.path.join(sys.path[0], "recall_vs_precision.png"))
  plt.show()

def conf_matrix(test_labels, test_preclasses):
  cf_matrix = confusion_matrix(test_labels, test_preclasses)
  group_names = ["True Neg","False Pos","False Neg","True Pos"]
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sn.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
  plt.savefig(os.path.join(sys.path[0], "confusion_matrix.png"))
  plt.show()

def class_report(test_preclasses, test_labels):
  text_file = open(os.path.join(sys.path[0], "classification_report.txt"), "w")
  n = text_file.write(classification_report(test_preclasses, test_labels))
  text_file.close()

test_predict = np.load(os.path.join(sys.path[0], "test_predict.npy"))
test_labels = np.load(os.path.join(sys.path[0], "test_labels.npy"))
test_labels = test_labels.astype('float64')
test_preclasses = []

for prediction in test_predict:
  if prediction > 0.5:
    test_preclasses.append(1)
  else:
    test_preclasses.append(0)

roc_auc(test_predict, test_labels)
precision_recall(test_labels, test_predict)
conf_matrix(test_preclasses, test_labels)
class_report(test_preclasses, test_labels)
print(matthews_corrcoef(test_labels, test_preclasses, sample_weight=None ))