from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import pandas as pd


iris = datasets.load_iris()
y = iris.target
sc = StandardScaler()

def twodim(d):
    lda = LinearDiscriminantAnalysis(n_components=2)
    d = sc.fit_transform(d)
    lda_object = lda.fit(d, y)
    d = lda_object.transform(d)
    return d
    
X = twodim(iris.data)

def plot_scatter(X,y):
    colors = ["red","green","blue"]
    markers = ('s', 'x', 'o')
    
    for i, yi in enumerate(np.unique(y)):
        Xi = X[y==yi]
        plt.scatter(Xi[:,0], Xi[:,1],
                        color=colors[i], marker=markers[i], label=yi)
    
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.legend(loc='upper left')
#import the required libraries

class Perceptron(object):
    #The constructor of our class.
    def __init__(self, learningRate=0.01, n_iter=50, random_state=1):
        self.learningRate = learningRate
        self.n_iter = n_iter
        self.random_state = random_state
        self.errors_ = []
        
    def fit(self, X, y):
        #for reproducing the same results
        random_generator = np.random.RandomState(self.random_state)
        
        #Step 0 = Get the shape of the input vector X
        #We are adding 1 to the columns for the Bias Term
        x_rows, x_columns = X.shape
        x_columns = x_columns+1
        
        #Step 1 - Initialize all weights to 0 or a small random number  
        #weight[0] = the weight of the Bias Term
        self.weights = random_generator.normal(loc=0.0, scale=0.001, size=x_columns) 
        
        #for how many number of training iterrations where defined
        for _ in range(self.n_iter):
            errors = 0
            for xi, y_actual in zip(X, y):
                #create a prediction for the given sample xi
                y_predicted = self.predict(xi)
                #print(y_actual, y_predicted)
                #calculte the delta
                delta = self.learningRate*(y_actual - y_predicted)
                #update all the weights but the bias
                self.weights[1:] += delta * xi
                #for the bias delta*1 = delta
                self.weights[0] += delta
                #if there is an error. Add to the error count for the batch
                errors += int(delta != 0.0)
            #add the error count of the batch to the errors variable
            self.errors_.append(errors)           
        
        
            
    def Errors(self):
        return self.errors_
    
    def z(self, X):
        #np.dot(X, self.w_[1:]) + self.w_[0]
        z = np.dot(X, self.weights[1:]) + self.weights[0] 
        return z
        
    def predict(self, X):
        #Heaviside function. Returns 1 or 0 
        return np.where(self.z(X) >= 0.0, 1, 0)





plot_scatter(X,y)
#plt.show()  

ppn = Perceptron(learningRate=0.1, n_iter=15)
ppn.fit(X, y)  
print(ppn.errors_)



print(ppn.predict([1, 2]))
print(ppn.predict([4, 5]))
print(ppn.predict([-3,-1]))
print(ppn.predict([-1,-1]))

print(ppn.weights)