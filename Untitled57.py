
# coding: utf-8

# In[ ]:


#import important and necessary libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
#dimension of the dataset
print(dataset.shape)
#peek head and tail
print(dataset.head())
print(dataset.tail())
#statistical description
print(dataset.describe())
print(dataset.groupby('class').size())
'''
univaraiate and multivariate plots
univariate plot is a plot for better understanding of individual attributes
multivariate plot is a plot for better understanding of relationship between the attributes

'''
#univariate plots:box and whisker plot
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()
#histograms
dataset.hist()
plt.show()
#multivariate plot: scatter plot matrix
scatter_matrix(dataset)
plt.show()
#split for validation dataset
arr=dataset.values
a=arr[:,0:4]
b=arr[:,4]
validation_size=0.2
seed=7
a_train,a_validation,b_train,b_validation=model_selection.train_test_split(a,b,test_size=validation_size,random_state=seed)
#test options
seed=7
scoring='accuracy'
#check
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
#evaluate each models
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,a_train,b_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)





