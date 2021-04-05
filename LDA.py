from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler



iris = datasets.load_iris()
X = iris.data
y = iris.target
sc = StandardScaler()


def twodim(d):
    lda = LinearDiscriminantAnalysis(n_components=2)
    d = sc.fit_transform(d)
    lda_object = lda.fit(d, y)
    d = lda_object.transform(d)
    return d

def onedim(d):
    lda = LinearDiscriminantAnalysis(n_components=1)
    d = sc.fit_transform(d)
    lda_object = lda.fit(d, y)
    d = lda_object.transform(d)
    return d

def plottwodim():
    for l,c,m in zip(np.unique(y),['r','g','b'],['s','x','o']):
        plt.scatter(twodim(X)[y==l,0],twodim(X)[y==l,1],c=c, marker=m, label=l,edgecolors='black')
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


















# lda2 = LinearDiscriminantAnalysis(n_components=1)
# lda_object2 = lda2.fit(X, y)
# X = lda_object2.transform(X)



# for i in range(len(X)):
#     if(y[i]==0):
#         plt.scatter(X[i][0],0,marker='s',c='red',edgecolor='black') 
#     elif  (y[i]==1):
#         plt.scatter(X[i][0],0,marker='x',c='green',edgecolor='black') 
#     else :
#         plt.scatter(X[i][0],0,marker='o',c='blue',edgecolor='black') 


















# import numpy as np
# #from sklearn.lda import LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])


# clf = LDA()
# clf.fit(X, y)

# LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
#   store_covariance=False, tol=0.0001)


# print(clf.predict([[-1, -1]]))

#print(X)
#print(y)

#print(X)
# i=0
# for l,c,m in zip(np.unique(y),['r','g','b'],['s','x','o']):
#     plt2.scatter(X[i],0,c=c, marker=m, label=l,edgecolors='black')
#     i =i+1
#     print(X[i])