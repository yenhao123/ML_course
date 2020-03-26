#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv(r"ml2020spring-hw1/train.csv",encoding="big5")


# # Preprocessing

# In[2]:


train = train.drop(["日期","測站","測項"],axis=1)
train[train == "NR"] = 0
train = train.to_numpy()


# # Extract Features

# In[3]:


#(18*20*12,24) -> (12,18,24*20)
month_data = np.empty([12,18,24*20],dtype=float)
for month in range(12):
    for day in range(20):
        month_data[month,:,day*24:(day+1)*24] = train[18*(20*month+day):18*(20*month+day+1),:]


# In[4]:


#(12,18*24*20) -> (471*12,18*9) 、(471*12,1)
x_train = np.empty([471*12,18*9],dtype=float)
y_train = np.empty([471*12,1],dtype=float)
for mon in range(12):
    for day in range(20):
        for hour in range(24):
            if (day*24 + hour) >= 471:
                continue
            x_train[mon*471+day*24+hour,:] = month_data[mon,:,(day*24+hour):(day*24+hour+9)].reshape(1,-1)
            y_train[mon*471+day*24+hour,0] = month_data[mon,9,(day*24+hour+9)]


# # Normalize

# In[5]:


#對各列normalize
x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        if x_std[j] != 0:
            x_train[i][j] = (x_train[i][j] - x_mean[j])/x_std[j]


# # Training

# In[6]:


dim = 18*9+1
x = np.concatenate([x_train,np.ones([471*12,1])],axis=1).astype(float)
w = np.zeros([dim,1])
loss = np.empty([20000])
adag = np.zeros([dim,1])
iteration = 20000
lr = 0.1
eps = 0.0000000001

for t in range(iteration):
    loss[t] = np.sqrt(np.sum(np.power((np.dot(x,w)-y_train),2))/471/12)
    gradient = 2*np.dot(x.transpose(),np.dot(x,w)-y_train)
    adag += gradient ** 2
    w -= lr * gradient/np.sqrt(adag+eps)
    if t%100 == 0:
        print("loss:",loss[t])
np.save("weight.npy",w)


# In[7]:


plt.plot(loss)
plt.xlabel("iteration")
plt.ylabel("loss")


# # Testing

# In[8]:


test = pd.read_csv(r"./ml2020spring-hw1/test.csv",index_col=False,header=None,encoding="big5")
id = test[0].unique()
test = test.iloc[:,2:]
test[test=="NR"] = 0
test = test.to_numpy()
#(18*24*10,9)->(24*10,18*9)
x_test = np.empty([24*10,18*9],dtype=float)
for i in range(240):
    x_test[i,:] = test[(18*i):(18*(i+1)),:].reshape(1,-1)
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        if x_std[j] != 0:
            x_test[i][j] = (x_test[i][j]-x_mean[j])/x_std[j]
x_test = np.concatenate((x_test,np.ones([24*10,1])),axis=1).astype(float)


# # Prediction

# In[9]:


weight = np.load("weight.npy")
pred = np.dot(x_test,weight).reshape(-1)


# # Submission

# In[10]:


sub = pd.DataFrame({
    "id" : id,
    "value" : pred
})
sub.to_csv("./ml2020spring-hw1/submission/sub.csv",index=False)


# In[ ]:




