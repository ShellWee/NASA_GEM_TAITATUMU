from pandas import read_csv
import numpy as np
import math
import pandas as pd
import ipdb

def preprocessing(X):
  for i in range(1, X.shape[1]):
    print(i)
    for j in range(X.shape[0]):
      median = np.median(X[:, i])
      X[j][i] = median if math.isnan(X[j][i]) else X[j][i]
    #   X[j][i] = 0 if math.isnan(X[j][i]) else X[j][i]

  ##check
#   for i in range(X.shape[1]):
#     for j in range(X.shape[0]):
#       if math.isnan(X[j][i]):
#         print("XD")

  return X

X_train = read_csv('test.csv').values
# train_X_processed = preprocessing(X_train)
X_train = pd.DataFrame(X_train)
# ipdb.set_trace()
train_X_processed = X_train.iloc[:, 1:].fillna(X_train.iloc[:, 1:].median())

train_X_processed.to_csv('test_fill_with_median.csv', index=True)
