from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from NaiveBayesClassifier import Gaussian_NB
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

breast_cancer = load_breast_cancer()
iris = load_iris()

X1, y1 = breast_cancer.data, breast_cancer.target
X2, y2 = iris.data, iris.target

print("Let's look at the classes histogram\n")
plt.hist(y1, label = 'breast cancer')
plt.hist(y2, label = 'iris')
plt.legend()
plt.show()

#split data in train and test
X1_tr, X1_ts, y1_tr, y1_ts = train_test_split(X1, y1, test_size=0.3)
X2_tr, X2_ts, y2_tr, y2_ts = train_test_split(X2, y2, test_size=0.3)

model_sklearn = GaussianNB()
my_model = Gaussian_NB()

print('Fiting models with breast cancer dataset...\n')
model_sklearn.fit(X1_tr, y1_tr)
my_model.fit(X1_tr, y1_tr)

print('Predicting...\n')
yp_sk = model_sklearn.predict(X1_ts)
yp_my = my_model.predict(X1_ts)

print('Scoring...\n')
print('slearn:\n', 'accuracy-', my_model.score('accuracy', y1_ts, yp_sk), 
      'Cohen_Kappa-', my_model.score('kappa', y1_ts, yp_sk),'\n')

print('My implementation:\n', 'accuracy-', my_model.score('accuracy', y1_ts, yp_my), 
      'Cohen_Kappa-', my_model.score('kappa', y1_ts, yp_my),'\n')

print("##################################################################\n")

print("Now let's Fit the models with iris dataset...\n")
model_sklearn.fit(X2_tr, y2_tr)
my_model.fit(X2_tr, y2_tr)

print('Predicting...\n')
yp_sk = model_sklearn.predict(X2_ts)
yp_my = my_model.predict(X2_ts)

print('Scoring...\n')
print('slearn:\n', 'accuracy-', my_model.score('accuracy', y2_ts, yp_sk), 
      'Cohen_Kappa-', my_model.score('kappa', y2_ts, yp_sk),'\n')

print('My implementation:\n', 'accuracy-', my_model.score('accuracy', y2_ts, yp_my), 
      'Cohen_Kappa-', my_model.score('kappa', y2_ts, yp_my))
