'''
The file implements the method to draw the confusion matrix
and calculate the recall, precision and specificity
'''

# import libraries
import torch
import numpy as np

'''
The row index of the confusion matrix indicates the actual class label,
The column index of the confusion matrix indicates the predicted class label
i.e.
 pre_0 pre_1 pre_2 pre_3 pre_4 pre_5 pre_6
0 x00   x01   x02   x03   x04   x05   x06
1 x10   x11   x12   x13   x14   x15   x16
2 x20   x21   x22   x23   x24   x25   x26
3 x30   x31   x32   x33   x34   x35   x36
4 x40   x41   x42   x43   x44   x45   x46
5 x50   x51   x52   x53   x54   x55   x56
6 x60   x61   x62   x63   x64   x65   x66

So when calculates the recall, precision and specificity, we should construct a binary confusion matrix for each class
i.e. for class 0
      pre_0 not_pre_0
0      TP     FN
not_0  FP     TN
where,
  TP = x00
  FN = confusion[0, 1:] (that is x01+x02+x03+x04+x05+x06)
  FP = confusion[1:, 0] (that is x10+x20+x30+x40+x50+x60)
  TN = confusion[1:, 1:] (the rest expect above)
So,
  recall_0 = TP/(TP+FN)
  precision_0 = TP/(TP+FP)
  specificity_0 = TN/(TN+FP) 
'''
# define a function to plot confusion matrix and calculate the class wise recall, precision and specificity
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    # Calculate the class wise recall, precision and specificity
    evaluation = [] # store the evaluation results of all classes
    confusion_mat = confusion.numpy()
    slice = [i for i in range(num_classes)]
    for label in range(num_classes):
        evaluate_class = [] # store the recall, precision and specificity of a class
        index = slice.copy()
        index.remove(label)
        TP = confusion_mat[label, label]
        FN = np.sum(confusion_mat[label,:][index])
        FP = np.sum(confusion_mat[:, label][index])
        # sum of the values in confusion_mat[:label,:label] and confusion_mat[label+1:,label+1:]
        TN = np.sum(confusion_mat[:label, :label]) + np.sum(confusion_mat[(label+1):, (label+1):])
        if TP+FN == 0:
            recall = 0
        else:
            recall = 100 * TP/(TP+FN)
        if TP+FP == 0:
            precision = 0
        else:
            precision = 100 * TP/(TP+FP)
        if TN+FN == 0:
            specificity = 0
        else:
            specificity = 100 * TN/(TN+FP)
        evaluate_class.append(recall)
        evaluate_class.append(precision)
        evaluate_class.append(specificity)
        evaluation.append(evaluate_class)


    return evaluation, confusion