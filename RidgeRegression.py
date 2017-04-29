import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.kernel_approximation import RBFSampler

test = False #Set test = True if you want to test RidgeRegression implementation

class RidgeRegression:
    
    def __init__(self, lambd):
        self.lambd = lambd   
    
    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        L = self.lambd * np.eye(X.shape[1])
        wrr = np.dot(np.linalg.inv(np.dot(X.T, X) + 
        np.dot(L.T, L)), np.dot(X.T, y))
        self.wrr = wrr
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        yp = np.dot(X, self.wrr)
        return yp
    
    def score(self, metric, y, yp):
        MAE = np.mean(np.abs(y-yp))
        MAPE = (MAE/np.mean(y))*100
        if metric == 'mae':
            return MAE
        elif metric == 'mape':
            return MAPE
        else:
            return print("Invalid metric. Availabe metrics: 'mae' and 'mape'")
 
if test == True:
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    print("Let's look at the target histogram\n")
    plt.hist(y, label = 'Diabetes', bins = 50)
    plt.show()
    print("Now Let's apply log transformation to the target to make it more Gaussian")
    y_log = np.log(y+50)
    plt.hist(y_log, label = 'Diabetes', bins = 50)
    #Split X and y in train and test
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y_log, test_size=0.3, random_state=0)
    print('Fiting model...\n')
    model = RidgeRegression(0.1)
    model.fit(X_tr, y_tr)
    print('Predicting...\n')
    yp = model.predict(X_ts)
    ##Need to make inverse transfor before calculate score
    yp1 = np.exp(yp)-50
    y_ts1 = np.exp(y_ts)-50
    print('MAPE(%) -', model.score('mape', y_ts1, yp1), '\n')
    print('MAE -', model.score('mae', y_ts1, yp1), 'n')
    print('It works fine, but error needs to get smaller\n')
    print("Let's try to use RBF Kernel\n")
    
    rbf = RBFSampler(gamma=0.2, n_components=300, random_state=0)
    X_rbf_tr = rbf.fit_transform(X_tr)
    X_rbf_ts = rbf.transform(X_ts)
    print('Fiting model...\n')
    model = RidgeRegression(0.3)
    model.fit(X_rbf_tr, y_tr)
    print('Predicting...\n')
    yp = model.predict(X_rbf_ts)
    ##Need to make inverse transfor before calculate score
    yp2 = np.exp(yp)-50
    y_ts2 = np.exp(y_ts)-50
    print('MAPE(%) -', model.score('mape', y_ts2, yp2), '\n')
    print('MAE -', model.score('mae', y_ts2, yp2), '\n')
    print('Now we can try to stack the 2 predictions\n')
    yp = (yp1 + yp2)/2
    print('MAPE(%) -', model.score('mape', y_ts2, yp), '\n')
    print('MAE -', model.score('mae', y_ts2, yp), '\n')
