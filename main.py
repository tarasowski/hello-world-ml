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

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset

## shape
dataset.shape
dataset.head()

## statistical summary
dataset.describe()

## class distribution
dataset.loc[:, 'class'].value_counts()
# dataset.groupby('class').size()

## univariable plots
dataset.plot(kind='box', figsize=(10,10), subplots=True, layout=(2,2), sharex=False, sharey=False) 

## histogram for each input variable of the iris flowers dataset
dataset.hist(figsize=(10,10))
plt.show()

## multivariate plots
# now we can look at the interactions between the variables
scatter_matrix(dataset, figsize=(15,15))
plt.show()

## create a validation dataset
a = dataset.values
X = a[:, 0:4]
Y = a[:, 4]
validation_size = 0.20

# seed values initialize randomization. Setting this value to the same number each time guarantees that the algorithm will come up with the same results (identical on each run)
seed = 7

# the generation of the data is randomized
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)
Y_train

## test harness == test drivers or stubs
# 10-fold cross validation == this will split our dataset into 10 parts, train on 9 and test on 1 and repreat for all combinations of train-test splits
seed = 7
scoring = 'accuracy' # this is a ratio of the number of correctly predicted instances / total number of instances * 100

# build models 
# we don't know which algorithm would be good on this problem or what configuration to use. So we'll brute force.

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f'{name}, {cv_results.mean()}, ({cv_results.std()})'
    print(msg)

# SVM (Support Vector Machines) win

## Compare algorithms
# outliers are draws as individual points on a boxplot
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## make predictions
# we've got some results on the training data, now we want to test the algorithm on the test set.
# it's the final check on the accuracy of the best model
# overly optimistic result = a model overfit or a data leak

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

