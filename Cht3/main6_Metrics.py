import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1, 1] # 正確答案
y_pred = [0, 1, 0, 1, 0, 1, 0, 1] # 預測結果

# Confusion Matrix
# Convert 2x2 matrix to 1D array.
print('Confusion Matrix = ',confusion_matrix(y_true, y_pred))
tn, fp, fn, tp  = confusion_matrix(y_true, y_pred).ravel()
# TP=3, FP=1(錯判斷成對的), TN=2, FN=2
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')

fig, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow([[1, 0], [0, 1]], cmap=plt.cm.Reds, alpha=0.3)

ax.text(x=0, y=0, s=tp, va='center', ha='center')
ax.text(x=1, y=0, s=fp, va='center', ha='center')
ax.text(x=0, y=1, s=tn, va='center', ha='center')
ax.text(x=1, y=1, s=fn, va='center', ha='center')

plt.xlabel('Real', fontsize=20)
plt.ylabel('Predict', fontsize=20)

plt.xticks([0,1], ['T', 'F'])
plt.yticks([0,1], ['P', 'N'])
plt.show()

print(f'accuracy_score:{accuracy_score(y_true, y_pred)}')
print(f'recompute = {(tp+fn) / (tp+tn+fp+fn)}')

print(f'precision_score:{precision_score(y_true, y_pred)}')
print(f'recompute={(tp) / (tp+fp)}')

print(f'recall_score:{recall_score(y_true, y_pred)}')
print(f'recompute={(tp) / (tp+fn)}')
