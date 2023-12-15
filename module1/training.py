from pandas import read_csv
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sc2
# import shap


X_train = read_csv('train_-2400.csv').values
X_test = read_csv('test-2400.csv').values
train_X = X_train[:,1:42]
train_Y = X_train[:,51].astype('int')
test_X = X_test[:,1:42]
test_Y = X_test[:,51].astype('int')

SSscaler = StandardScaler()
# MMScaler = MinMaxScaler()
train_X = SSscaler.fit_transform(train_X)
test_X = SSscaler.transform(test_X)
train_X = train_X[:, [20, 21, 14,  0, 12, 24, 13, 17, 15, 25, 19,  7,  1, 29, 23, 27, 26,
            28, 11, 18, 10, 16, 22, 34,  2,  8,  9,  6, 36]]
test_X = test_X[:, [20, 21, 14,  0, 12, 24, 13, 17, 15, 25, 19,  7,  1, 29, 23, 27, 26,
            28, 11, 18, 10, 16, 22, 34,  2,  8,  9,  6, 36]]
# smtom = SMOTETomek(random_state=7414)
# X_res, Y_res = smtom.fit_resample(train_X,train_Y)
# random_indices = np.random.permutation(X_res.shape[0])
# X_res = X_res[random_indices]
# Y_res = Y_res[random_indices]

# model = XGBClassifier(max_depth = 17, 
#                       min_child_weight = 3,
#                       gamma = 0.7,
#                       subsample = 0.8,
#                       colsample_bytree = 0.8,
#                       reg_alpha = 0,
#                       reg_lambda = 1,
#                       learning_rate = 0.1)
# model = XGBClassifier(max_depth = 17, 
#                       min_child_weight = 3,
#                       gamma = 0.5,
#                       subsample = 0.8,
#                       colsample_bytree = 0.8,
#                       reg_alpha = 0,
#                       reg_lambda = 1,
#                       learning_rate = 0.1)
# model.fit(X_res, Y_res)
# result = model.predict(test_X)
# # print(accuracy_score(model.predict(test_X), test_Y))
# joblib.dump(model, 'xgb_classifier_model_ss.pkl')


loaded_model = joblib.load('xgb_classifier_model_ss29.pkl')
result = loaded_model.predict(test_X)
pd.DataFrame(result).to_csv('predicted_label_for_2400.csv', index=False)

# param_grid = [
#   {'max_depth': [9, 12, 15, 17], 
#    'min_child_weight': [3, 5, 7],
#    'gamma' : [0.3, 0.5, 0.7],
#    'subsample' : [0.8],
#    'colsample_bytree' : [0.8],
#    'reg_alpha' : [0],
#    'reg_lambda' : [1],
#    'learning_rate' : [0.1],
#    }
# ]
# grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5,
#                           scoring='precision', verbose=10)
# grid_search.fit(X_res, Y_res)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
# print('Best score: ', grid_search.best_score_)
# print('Test Accuracy: ', grid_search.score(test_X, test_Y))

cm = confusion_matrix(np.array(test_Y).flatten(),np.array(result).flatten())
print(cm)
accuracy = accuracy_score(np.array(test_Y).flatten(),np.array(result).flatten())
precision = precision_score(np.array(test_Y).flatten(),np.array(result).flatten())
recall = recall_score(np.array(test_Y).flatten(),np.array(result).flatten())
print(f'The accuracy of the model is {accuracy:.4f}, the precision is {precision:.4f}, and the recall is {recall:.4f}.')

CLASSES = ["Normal Values", "Abnormal Values"]

confusion = confusion_matrix(np.array(test_Y).flatten(),np.array(result).flatten())

# Calculate normalized confusion matrix
normalized_confusion = np.round(confusion / np.sum(confusion, axis=0), decimals=3)

# Create heatmap
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
plt.figure(figsize=(6, 6))
ax = sns.heatmap(normalized_confusion, xticklabels=CLASSES, yticklabels=CLASSES,
                annot=False, cmap=cmap, linewidths=.5, cbar=True, annot_kws={"fontsize": 18})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=16)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=16)
plt.xlabel('Prediction', fontsize=18)
plt.ylabel('True label', fontsize=18)
plt.title(f'Classification Results')
plt.tight_layout()

for i in range(len(CLASSES)):
  for j in range(len(CLASSES)):
    count = confusion[i, j]
    prob = normalized_confusion[i, j]
    if i == j:
      ax.text(j + 0.5, i + 0.5, f'{count} ({prob:.2f})', ha='center', va='center', color='white', fontsize=16)
    else:
      ax.text(j + 0.5, i + 0.5, f'{count} ({prob:.2f})', ha='center', va='center', color='black', fontsize=16)


plt.show()
plt.close()
