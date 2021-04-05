import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from numpy import linalg as LA

style.use('fivethirtyeight')
np.random.seed(seed=42)

# Create data

yes1 = pd.read_excel('D:\personal\Pattern\project\dataset\yes.xlsx')
no1 = pd.read_excel('D:\personal\Pattern\project\dataset\yesnot.xlsx')
yes1.as_matrix()
no1.as_matrix()
#print (yes)
#print (no)

yes =np.array(yes1)
no =np.array(no1)


#Plot the data
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

f=len(yes)
v=len(no)

for i in range(f):
  ax0.scatter(yes[i][0],yes[i][1],marker='s',c='grey',edgecolor='black') 

for i in range(v):
  ax0.scatter(no[i][0],no[i][1],marker='^',c='yellow',edgecolor='black') 




# Calculate the mean vectors per class
mean_yes = np.mean(yes.reshape(2,f),axis=1)#.reshape(2,f)
mean_no = np.mean(no.reshape(2,v),axis=1)#.reshape(2,v)



# Calculate the scatter matrices for the SW (Scatter within) and sum the elements up

# Mind that we do not calculate the covariance matrix here because then we have to divide by n or n-1 as shown below
#print((1/7)*np.dot((rectangles-mean_rectangles),(rectangles-mean_rectangles).T))
#print(np.var(rectangles[0],ddof=0))

scatter_yes = np.dot((yes-mean_yes).T,(yes-mean_yes))
scatter_no = np.dot((no-mean_no).T,(no-mean_no))


# Calculate the SW by adding the scatters within classes 

SW=scatter_yes+scatter_no

w, v = LA.eig(SW)
print(w)
print(v)

plt.plot(w)

plt.show()
