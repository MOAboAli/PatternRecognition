from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# X=twodim(X)

kmeans = KMeans(n_clusters=3, random_state=0).fit(twodim(X))

test_X=np.array([[1, 2], [4, 5],[-3,-1] , [-1,-1]])
print(kmeans.predict(test_X))

print(kmeans.cluster_centers_)

def plotonedim():
    for i in range(len(twodim(X))):
        if(y[i]==0):
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='s',c='red',edgecolor='black') 
        elif  (y[i]==1):
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='x',c='green',edgecolor='black') 
        else :
            plt.scatter(twodim(X)[i][0],twodim(X)[i][1],marker='o',c='blue',edgecolor='black')




# plt.scatter(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],marker=r'$\clubsuit$',c='red',edgecolor='black') 
# plt.scatter(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],marker=r'$\clubsuit$',c='green',edgecolor='black') 
# plt.scatter(kmeans.cluster_centers_[2][0],kmeans.cluster_centers_[2][1],marker=r'$\clubsuit$',c='blue',edgecolor='black')


plotonedim()

plt.plot(-3,-1, '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
plt.plot(-1,-1, '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
plt.plot(1,2, '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2) 
plt.plot(4,5, '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2) 


# plt.plot('-3','-1', '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
# plt.plot('-1','-1', '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
# plt.plot('1','2', '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
# plt.plot('4','5', '-p', color='gray',markersize=15, linewidth=4,markerfacecolor='white',markeredgecolor='gray',markeredgewidth=2)
 






plt.show()