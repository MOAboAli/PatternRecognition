
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch as t
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        #self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(2, 1)


def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x)

net = Net()
print(net)

# create a stochastic gradient descent optimizer
optimizer = t.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()

# run the main training loop
for epoch in range(20):
    for batch_idx, (data, target) in enumerate(X):
        data, target = Variable(data), Variable(target)
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        data = data.view(-1, 28*28)
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

net_out = net(data)
loss = criterion(net_out, target)

loss.backward()
optimizer.step()

if batch_idx % log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))




