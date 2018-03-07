from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import pickle
import sklearn

with open('../features/set_of_features/set_neg13', 'rb') as f:
    neg = pickle.load(f)
    
with open('../features/set_of_features/set_pos13', 'rb') as f:
    pos = pickle.load(f)

del neg['dat']
del pos['dat']

'''
list of dicts with the features
'''
neg_feat = []
for k,v in neg.items():
    neg_feat.append(v)

pos_feat = []
for k,v in pos.items():
    pos_feat.append(v)

'''
create positive and negative labels
and put them together in a list. 
Same with the list of dicts
'''
neg_labels = [0] * len(neg_feat)
pos_labels = [1] * len(pos_feat)

feat = neg_feat + pos_feat
labels = neg_labels + pos_labels

'''
vectorizing the features
'''
v = DictVectorizer(sparse=False)
X = v.fit_transform(feat)

print('X shape: {}'.format(X.shape))

'''
let's work with DataFrames!
'''
df = pd.DataFrame(X)
se = pd.Series(labels)
df['label'] = se.values

features = df.iloc[:, df.columns != 'label']
labels = df['label']

'''
trains/test split
'''
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels, 
                                                          test_size=0.2,
                                                          random_state=42)

'''
try different classifiers
'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)

from sklearn.svm import SVC
clf = SVC(kernel='linear',C=0.1, gamma=0.01, random_state=42)
model2 = clf.fit(train, train_labels)
preds2 = clf.predict(test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=150, max_features='log2', random_state=42)
model3 = clf.fit(train, train_labels)
preds3 = clf.predict(test)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=20)
model4 = neigh.fit(train, train_labels) 
preds4 = neigh.predict(test)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
model5 = ada.fit(train, train_labels)
preds5 = ada.predict(test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
model6 = mlp.fit(train, train_labels)
preds6 = mlp.predict(test)


from sklearn.metrics import accuracy_score, precision_score, recall_score

'''
evaluate accuracy, precision
and recall score
'''
print('Naive: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds),
      precision_score(test_labels, preds), recall_score(test_labels, preds)))
print('SVM: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds2),
      precision_score(test_labels, preds2), recall_score(test_labels, preds2)))
print('RF: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds3),
      precision_score(test_labels, preds3), recall_score(test_labels, preds3)))
print('KNN: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds4),
      precision_score(test_labels, preds4), recall_score(test_labels, preds4)))
print('Ada: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds5),
      precision_score(test_labels, preds5), recall_score(test_labels, preds5)))
print('MLP: accuracy = {:0.2f}, precision = {:0.2f}, recall = {:0.2f}'.format(accuracy_score(test_labels, preds6),
      precision_score(test_labels, preds6), recall_score(test_labels, preds6)))
