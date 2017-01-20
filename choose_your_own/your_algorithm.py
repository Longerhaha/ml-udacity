#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary



import time
from sklearn.metrics import accuracy_score

#try knn algorithm
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)
'''
training time cost: 0.0 s
predict time cost: 0.0 s
accuracy: 0.944
knn参数为8的时候准确率超过了93.6%
'''
'''
#try random forest algorithm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
'''
'''
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
'''
start_time = time.time()
clf = clf.fit( features_train,labels_train )
end_time = time.time()
print 'training time cost:',round(end_time-start_time,3),'s'

start_time = time.time()
pred = clf.predict( features_test )
end_time = time.time()
print 'predict time cost:',round(end_time-start_time,3),'s'

acc = accuracy_score( labels_test,pred)
print 'accuracy:',round( acc,3 )


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
