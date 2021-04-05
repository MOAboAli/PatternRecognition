import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X
    
def plottwodim():
    for l,c,m in zip(np.unique(y),['r','g','b'],['s','x','o']):
        plt.scatter(twodim(X)[y==l,0],twodim(X)[y==l,1],c=c, marker=m, label=l,edgecolors='black')
    plt.show()

test_X=np.array([[1, 2], [4, 5],[-3,-1] , [-1,-1]])
test_y=np.array([[0], [1],[2],[0]])

train_X = Variable(torch.Tensor(X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(y).long())
test_y = Variable(torch.Tensor(test_y).long())




net = Net()

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print ('number of epoch', epoch, 'loss', loss.item())   
        # loss.data[0])

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print(predict_y.data)

#plottwodim()






# print ('prediction accuracy', accuracy_score(test_y.data, predict_y.data))
# print ('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
# print ('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
# print ('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
# print ('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))



# load IRIS dataset
# dataset = pd.read_csv('dataset/iris.csv')

# # transform species to numerics
# dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
# dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
# dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2


# train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
#                                                     dataset.species.values, test_size=0.8)

# # wrap up with Variable in pytorch


