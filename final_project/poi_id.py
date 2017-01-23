#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop('TOTAL')  #remove outliers

###Create new feature(s)
#After delete outliers,the parameters is that as following
'''
GaussianNB()
        Accuracy: 0.79660       Precision: 0.46398      Recall: 0.10950 F1: 0.17718     F2: 0.12925
        Total predictions: 10000        True positives:  219    False positives:  253   False negatives: 1781   True negatives: 7747
'''
features_list.append('bonus')
'''
GaussianNB()
        Accuracy: 0.77760       Precision: 0.37692      Recall: 0.17150 F1: 0.23574     F2: 0.19248
        Total predictions: 10000        True positives:  343    False positives:  567   False negatives: 1657   True negatives: 7433
'''

features_list.append('exercised_stock_options')
'''
GaussianNB()
        Accuracy: 0.84277       Precision: 0.48281      Recall: 0.30900 F1: 0.37683     F2: 0.33297
        Total predictions: 13000        True positives:  618    False positives:  662   False negatives: 1382   True negatives: 10338
'''
features_list.append('long_term_incentive')
'''
GaussianNB()
        Accuracy: 0.83577       Precision: 0.45223      Recall: 0.31950 F1: 0.37445     F2: 0.33942
        Total predictions: 13000        True positives:  639    False positives:  774   False negatives: 1361   True negatives: 10226
'''
#features_list.append('total_payments')  add this feature,Precision and Recall will lower,so don't append this feature
'''
GaussianNB()
        Accuracy: 0.84507       Precision: 0.42476      Recall: 0.23850 F1: 0.30548     F2: 0.26143
        Total predictions: 14000        True positives:  477    False positives:  646   False negatives: 1523   True negatives: 11354
'''
features_list.append('total_stock_value')
'''
GaussianNB()
        Accuracy: 0.83723       Precision: 0.45748      Recall: 0.31200 F1: 0.37099     F2: 0.33319
        Total predictions: 13000        True positives:  624    False positives:  740   False negatives: 1376   True negatives: 10260
'''
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#feature scaling 
#from sklearn.preprocessing import MinMaxScaler
#scale = MinMaxScaler()
#scale_features = scale.fit_transform(features)   



#do visualization first,to see if there is some outliers
'''import matplotlib.pyplot as plt
feature = []
for index in features:
    feature.append(index)
plt.plot(feature)
plt.show()
throuht this 1d plot,I found an outliers(index is 87),it's keys is 'TOTAL'
'''
#remove outliers
#labels.remove(labels[87])
#features.remove(features[87])



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
'''
GaussianNB()
        Accuracy: 0.83723       Precision: 0.45748      Recall: 0.31200 F1: 0.37099     F2: 0.33319
        Total predictions: 13000        True positives:  624    False positives:  740   False negatives: 1376   True negatives: 10260
'''

'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
        Accuracy: 0.78277       Precision: 0.30154      Recall: 0.31300 F1: 0.30716     F2: 0.31064
        Total predictions: 13000        True positives:  626    False positives: 1450   False negatives: 1374   True negatives: 9550
'''
'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
        Accuracy: 0.55592       Precision: 0.05094      Recall: 0.10700 F1: 0.06902     F2: 0.08770
        Total predictions: 13000        True positives:  214    False positives: 3987   False negatives: 1786   True negatives: 7013
'''
'''
from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'kernal':('linear','rbf'),'C':[0.01,0.1,1,10,100,10000,100000]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
'''         
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)