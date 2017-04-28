import numpy as np
import math
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss

def separate_classes(X, y):
    nclasses = np.max(y)
    vector_X = [[] for k in range(nclasses + 1)]
    Xs = []
    # [[]]*nclasses doesn't work correctly when doing append().
    for i in range(len(y)):                     
        for n in range (nclasses + 1):
            if y[i] == n :
                vector_X[n].append(X[i])
    for l in range (len(vector_X)):
        Xs.append(np.matrix(vector_X[l]))
    del vector_X
    return Xs
    

def Prior(X_, y): #X_ is the output of separate_classes funcion
    priors = np.zeros(len(X_))
    for i in range(len(X_)):
        priors[i] = X_[i].shape[0]/len(y)
    return priors

def Gaussian_prob(x, V): 
    #claculates the probability of a given value 'x' occours in the vector 'V'
    #This funcion will be needed to calculate likelihood and evidence
    exp = math.exp((-(x-np.mean(V))**2)/(2*np.std(V)**2))
    return exp/(math.sqrt(2*math.pi*np.std(V)**2))


def likelihood(x_row, X_):  #X_ is the output of separate_classes funcion and
    likelihood = np.ones(len(X_))
    for i in range(len(X_)):
        for j in range(X_[i].shape[1]):
            likelihood[i] *= Gaussian_prob(x_row[j], X_[i][:, j])
    return likelihood

def evidence(x, X):
    #use Gaussian_prob to calculate P(x) for ev
    P_X = 1
    for i in range(X.shape[1]):
        P_X *= Gaussian_prob(x[i], X[:, i]) #PROBLEM IS HERE!!!!
    return P_X

#Bayes theorem
#P(C/x) = (P(C) * P(X/C))/P(X) or Posterior = (Prior * Likelihood)/evidence
class Gaussian_NB:
    
    def fit(self, X, y):
        self.X_ = separate_classes(X, y)
        self.prior = Prior(self.X_, y)
        return self
    
    def predict_proba(self, X_test):
        proba = [[] for k in range(X_test.shape[0])]
        for i in range(X_test.shape[0]):
            row = X_test[i][:]
            l = likelihood(row, self.X_)
            P_X = evidence(row, X) 
            proba[i] = self.prior*l/P_X
        self.proba = np.matrix(proba)
        return self.proba
    
    def predict(self, X_test): #MAP solution
        #proba = self.predict_proba(X_test)
        yp = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            yp[i] = np.argmax(self.proba[i])
        return yp
    
    def score(self, metric, y, yp):
        if metric == 'accuracy':
            score = accuracy_score(y, yp)
        elif metric == 'kappa':
            score = cohen_kappa_score(y, yp)
        elif metric == 'log_loss':
            score = log_loss(y, yp) #yp must be a the result of predict_proba()
        else:
            return print('Invalid metric name. Use "accuracy", "kappa" or "log_loss"')
        return score

'''
#Uncomment to test the code

X = np.array([np.random.normal(1, 1, 10000), np.random.normal(2, 1, 10000), np.random.normal(3, 1, 10000)])
X = X.reshape(X.shape[1], X.shape[0])
y = np.random.choice(range(5), size = 10000)

model = Gaussian_NB()
model.fit(X, y)
yprob = model.predict_proba(X)
yp = model.predict(X)
score = model.score('accuracy', y, yp)
print (score)
score2 = model.score('kappa', y, yp)
print (score2)
score3 = model.score('log_loss', y, yprob)
print(score3)
score4 = model.score('random_metric', y, yp)
'''

