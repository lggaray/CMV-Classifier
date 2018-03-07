from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import sklearn, pickle
from sklearn.model_selection import cross_val_score

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

'''
let's work with DataFrames!
'''
df = pd.DataFrame(X)
se = pd.Series(labels)
df['label'] = se.values

features = df.iloc[:, df.columns != 'label']
labels = df['label']

'''
folds for cv
'''
folds = 10

'''
cross-validating the highest-performing
classifier with the sets we've chosen
'''
from sklearn.svm import SVC
clf = SVC(kernel='linear',C=0.1, gamma=0.01, random_state=42)
scores = np.array(cross_val_score(clf, features, labels, cv=folds))
mean = np.mean(scores)
std = np.std(scores)
print('SVM')
print('scores: {}, mean: {:0.2f}, std: {:0.2f}'.format(scores,mean,std))

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
scores = np.array(cross_val_score(ada, features, labels, cv=folds))
mean = np.mean(scores)
std = np.std(scores)
print('Ada')
print('scores: {}, mean: {:0.2f}, std: {:0.2f}'.format(scores,mean,std))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
scores = np.array(cross_val_score(mlp, features, labels, cv=folds))
mean = np.mean(scores)
std = np.std(scores)
print('MLP')
print('scores: {}, mean: {:0.2f}, std: {:0.2f}'.format(scores,mean,std))
