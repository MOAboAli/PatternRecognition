

import numpy as np
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler


style.use('fivethirtyeight')
np.random.seed(seed=42)

Features1 = pd.read_excel('E:\education\master\Pattern Recognition\project\mycode\dataset-x.xlsx').as_matrix()
Class1 = pd.read_excel('E:\education\master\Pattern Recognition\project\mycode\datasetclass.xlsx').as_matrix()


Features =np.array(Features1)
Class =np.array(Class1)

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

for i in range(len(Features)):
    if(Class[i]==0):
        ax0.scatter(Features[i][0],Features[i][1],marker='s',c='grey',edgecolor='black') 
    else:
        ax0.scatter(Features[i][0],Features[i][1],marker='^',c='yellow',edgecolor='black') 



# clf = LDA()
# clf.fit(Features, Class)

# LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
#   store_covariance=False, tol=0.0001)

#print(clf.predict([[70, 2]]))

#plt.show()

sc = StandardScaler()
X = sc.fit_transform(Features)

lda = LDA(n_components=2)
lda_object = lda.fit(X, Class)
X = lda_object.transform(X)

print(X)

# for l,c,m in zip(np.unique(Class),['r','g','b'],['s','x','o']):
#     plt.scatter(X,0,c=c, marker=m, label=l,edgecolors='black')


fig2 = plt.figure(figsize=(10,10))
ax1 = fig2.add_subplot(111)

for i in range(len(X)):
    if(Class[i]==0):
        ax1.scatter(X[i][0],0,marker='s',c='blue',edgecolor='black') 
    else:
        ax1.scatter(X[i][0],0,marker='^',c='red',edgecolor='black') 



plt.show()