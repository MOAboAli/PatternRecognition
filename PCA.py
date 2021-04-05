import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
sc = StandardScaler()


def twodim(d):
    pca = decomposition.PCA(n_components=2)
    pca.fit(d)
    d = pca.transform(d)
    return d


def onedim(d):
    pca = decomposition.PCA(n_components=1)
    pca.fit(d)
    d = pca.transform(d)
    return d


def plottwodim():
    for i in range(len(twodim(X))):
        if(y[i]==0):
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='s',c='red',edgecolor='black') 
        elif  (y[i]==1):
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='x',c='green',edgecolor='black') 
        else :
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='o',c='blue',edgecolor='black')
    plt.show()



def plotonedim():
    for i in range(len(onedim(twodim(X)))):
        if(y[i]==0):
            plt.scatter(onedim(twodim(X))[i][0],0,marker='s',c='red',edgecolor='black') 
        elif  (y[i]==1):
            plt.scatter(onedim(twodim(X))[i][0],0,marker='x',c='green',edgecolor='black') 
        else :
            plt.scatter(onedim(twodim(X))[i][0],0,marker='o',c='blue',edgecolor='black')
    plt.show()
    
    



plottwodim()
plotonedim()









