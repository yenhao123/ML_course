#!/usr/bin/env python
# coding: utf-8

# # (a) Set Data 

# In[1]:


#set zero-mean and one std normal data
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import truncnorm

lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[20,1])

x = np.linspace(-3,3,20).reshape(-1,1)

y = 2*x + e

#recordes training_loss for deg 1、5、10、14
loss = np.empty([4,3])


# # Preprocessing

# In[2]:


def preprocessing(x,y,mode=None,num=20):
    #holdoutcv
    id = np.random.permutation(num)
    id_train,id_test = id[:math.floor(num*0.75)],id[math.floor(num*0.75):]
    x_train = x[id_train]
    y_train = y[id_train]
    x_test = x[id_test]
    y_test = y[id_test]
    
    if mode == "levOneOut":
        num = math.floor(num*0.75)
        x_train_l = np.empty([num,num-1,1]).astype(float)
        x_val_l = np.empty([num,1,1]).astype(float)
        y_train_l = np.empty([num,num-1,1]).astype(float)
        y_val_l = np.empty([num,1,1]).astype(float)
        for i in range(num):
            x_train_l[i] = np.delete(x_train,i,0)
            x_val_l[i] = x_train[i]
            y_train_l[i] = np.delete(y_train,i,0)
            y_val_l[i] = y_train[i]
            
        return x_train_l,y_train_l,x_val_l,y_val_l,x_test,y_test
    
    elif mode == "fiveFold":
        num = math.floor(num*0.75)
        x_train_f = np.empty([5,int(num/5*4),1]).astype(float)
        x_val_f = np.empty([5,int(num/5),1]).astype(float)
        y_train_f = np.empty([5,int(num/5*4),1]).astype(float)
        y_val_f = np.empty([5,int(num/5),1]).astype(float)
        for i in range(5):
            list = range(int(num/5*i),int(num/5*(i+1)))
            x_val_f[i] = x_train[list]
            y_val_f[i] = y_train[list]
            k_x = 0
            for j in range(num):
                if j not in list:
                    x_train_f[i][k_x] = x_train[j]
                    y_train_f[i][k_x] = y_train[j]
                    k_x+=1
        return x_train_f,y_train_f,x_val_f,y_val_f,x_test,y_test
    
    return x_train,y_train,x_test,y_test


# # Linear Regression

# In[3]:


def linearRegression(x,y):
    X = np.concatenate((x,np.ones([x.shape[0],1])),axis=1)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return W


# In[4]:


def mae(y_hat,y):
    err = np.sum(np.power(y_hat-y,2))/y.shape[0]
    return err


# In[5]:


#training err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = linearRegression(x_train,y_train)

y_hat = x_train*W[0]+W[1]
train_err = mae(y_hat,y_train)
y_hat = x_test*W[0]+W[1]
test_err = mae(y_hat,y_test)
loss[0][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_tr = np.linspace(3,-3,200)
ys_tr = W[0] * xs_tr + W[1]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = linearRegression(x_train[i],y_train[i])
    y_hat = x_val[i]*W[0]+W[1]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = x_test*best_w[0]+best_w[1]
test_err = mae(y_hat,y_test)
loss[0][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_lev = np.linspace(3,-3,200)
ys_lev = best_w[0] * xs_lev + best_w[1]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = linearRegression(x_train[i],y_train[i])
    y_hat = x_val[i]*W[0]+W[1]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = x_test*best_w[0]+best_w[1]
test_err = mae(y_hat,y_test)
loss[0][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_f = np.linspace(3,-3,200)
ys_f = best_w[0] * xs_f + best_w[1]


# # fitting plot

# In[6]:


plt.plot(xs_tr,ys_tr,"r",label="traing_err")
plt.plot(xs_lev,ys_lev,"b",label="lev_err")
plt.plot(xs_f,ys_f,"g",label="fold_err")
plt.legend()


# # (b)Polynomial Regression

# In[7]:


def polynomialRegression(x,y,deg):
    X = np.concatenate((np.ones([x.shape[0],1]),x),axis=1)
    for i in range(deg-1):
        X = np.concatenate((X,x**(i+2)),axis=1)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return W


# In[8]:


#degree 5
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,5)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
loss[1][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d5_tr = np.linspace(-3,3,200)
ys_d5_tr = W[0]+xs_d5_tr*W[1]+(xs_d5_tr**2)*W[2]+(xs_d5_tr**3)*W[3]+(xs_d5_tr**4)*W[4]+(xs_d5_tr**5)*W[5]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],5)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
loss[1][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d5_lev = np.linspace(-3,3,200)
ys_d5_lev = W[0]+xs_d5_lev*W[1]+(xs_d5_lev**2)*W[2]+(xs_d5_lev**3)*W[3]+(xs_d5_lev**4)*W[4]+(xs_d5_lev**5)*W[5]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],5)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
loss[1][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d5_f = np.linspace(-3,3,200)
ys_d5_f = W[0]+xs_d5_f*W[1]+(xs_d5_f**2)*W[2]+(xs_d5_f**3)*W[3]+(xs_d5_f**4)*W[4]+(xs_d5_f**5)*W[5]


# # fitting plot

# In[9]:


plt.plot(xs_d5_tr,ys_d5_tr,"r",label="traing_err")
plt.plot(xs_d5_lev,ys_d5_lev,"b",label="lev_err")
plt.plot(xs_d5_f,ys_d5_f,"g",label="fold_err")
plt.legend()


# In[10]:


#degree 10
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,10)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
loss[2][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d10_tr = np.linspace(-3,3,100)
ys_d10_tr = W[0]+xs_d10_tr*W[1]+(xs_d10_tr**2)*W[2]+(xs_d10_tr**3)*W[3]+(xs_d10_tr**4)*W[4]+(xs_d10_tr**5)*W[5]+(xs_d10_tr**6)*W[6]+(xs_d10_tr**7)*W[7]+(xs_d10_tr**8)*W[8]+(xs_d10_tr**9)*W[9]+(xs_d10_tr**10)*W[10]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],10)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
loss[2][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d10_lev = np.linspace(-3,3,200)
ys_d10_lev = W[0]+xs_d10_lev*W[1]+(xs_d10_lev**2)*W[2]+(xs_d10_lev**3)*W[3]+(xs_d10_lev**4)*W[4]+(xs_d10_lev**5)*W[5]+(xs_d10_lev**6)*W[6]+(xs_d10_lev**7)*W[7]+(xs_d10_lev**8)*W[8]+(xs_d10_lev**9)*W[9]+(xs_d10_lev**10)*W[10]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],10)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
loss[2][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d10_f = np.linspace(-3,3,200)
ys_d10_f = W[0]+xs_d10_f*W[1]+(xs_d10_f**2)*W[2]+(xs_d10_f**3)*W[3]+(xs_d10_f**4)*W[4]+(xs_d10_f**5)*W[5]+(xs_d10_f**6)*W[6]+(xs_d10_f**7)*W[7]+(xs_d10_f**8)*W[8]+(xs_d10_f**9)*W[9]+(xs_d10_f**10)*W[10]


# # fitting plot

# In[11]:


plt.plot(xs_d10_tr,ys_d10_tr,"r",label="traing_err")
plt.plot(xs_d10_lev,ys_d10_lev,"b",label="lev_err")
plt.plot(xs_d10_f,ys_d10_f,"g",label="fold_err")
plt.legend()


# In[12]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,100)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# # fitting plot

# In[13]:


plt.plot(xs_d14_tr,ys_d14_tr,"r",label="traing_err")
plt.plot(xs_d14_lev,ys_d14_lev,"b",label="lev_err")
plt.plot(xs_d14_f,ys_d14_f,"g",label="fold_err")
plt.legend()


# # fitting plot degree  5、10、14

# In[14]:


plt.scatter(x,y)
plt.plot(xs_tr,ys_tr,"c",label="deg_1")
plt.plot(xs_d5_tr,ys_d5_tr,"r",label="deg_5")
plt.plot(xs_d10_tr,ys_d10_tr,"b",label="deg_10")
plt.plot(xs_d14_tr,ys_d14_tr,"g",label="deg_14")
plt.legend()


# # deg 1、5、10、14 training_loss

# In[15]:


import prettytable as pt
tb1 = pt.PrettyTable()
tb1.field_names = ["Degree","Training Error","Leave-One-Out","Five-Fold"]
tb1.add_row(["1",loss[0][0],loss[0][1],loss[0][2]])
tb1.add_row(["5",loss[1][0],loss[1][1],loss[1][2]])
tb1.add_row(["10",loss[2][0],loss[2][1],loss[2][2]])
tb1.add_row(["14",loss[3][0],loss[3][1],loss[3][2]])
print(tb1)


# # (c) Set Data y=sin(2*pi*x) +ｅ

# In[16]:


#set zero-mean and one std normal data
lower,upper = 0,1
mu,sigma = 0,0.02
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[20,1])

x = np.linspace(0,1,20).reshape(-1,1)

y = np.sin(2*np.pi*x) + e


# In[17]:


#degree 5
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,5)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
print("train_err:",train_err,"test_err:",test_err)

xs_d5_tr = np.linspace(-3,3,200)
ys_d5_tr = W[0]+xs_d5_tr*W[1]+(xs_d5_tr**2)*W[2]+(xs_d5_tr**3)*W[3]+(xs_d5_tr**4)*W[4]+(xs_d5_tr**5)*W[5]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],5)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d5_lev = np.linspace(-3,3,200)
ys_d5_lev = W[0]+xs_d5_lev*W[1]+(xs_d5_lev**2)*W[2]+(xs_d5_lev**3)*W[3]+(xs_d5_lev**4)*W[4]+(xs_d5_lev**5)*W[5]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],5)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]
test_err = mae(y_hat,y_test)
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d5_f = np.linspace(-3,3,200)
ys_d5_f = W[0]+xs_d5_f*W[1]+(xs_d5_f**2)*W[2]+(xs_d5_f**3)*W[3]+(xs_d5_f**4)*W[4]+(xs_d5_f**5)*W[5]


# In[18]:


#degree 10
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,10)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
print("train_err:",train_err,"test_err:",test_err)

xs_d10_tr = np.linspace(-3,3,100)
ys_d10_tr = W[0]+xs_d10_tr*W[1]+(xs_d10_tr**2)*W[2]+(xs_d10_tr**3)*W[3]+(xs_d10_tr**4)*W[4]+(xs_d10_tr**5)*W[5]+(xs_d10_tr**6)*W[6]+(xs_d10_tr**7)*W[7]+(xs_d10_tr**8)*W[8]+(xs_d10_tr**9)*W[9]+(xs_d10_tr**10)*W[10]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],10)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d10_lev = np.linspace(-3,3,200)
ys_d10_lev = W[0]+xs_d10_lev*W[1]+(xs_d10_lev**2)*W[2]+(xs_d10_lev**3)*W[3]+(xs_d10_lev**4)*W[4]+(xs_d10_lev**5)*W[5]+(xs_d10_lev**6)*W[6]+(xs_d10_lev**7)*W[7]+(xs_d10_lev**8)*W[8]+(xs_d10_lev**9)*W[9]+(xs_d10_lev**10)*W[10]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],10)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]
test_err = mae(y_hat,y_test)
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d10_f = np.linspace(-3,3,200)
ys_d10_f = W[0]+xs_d10_f*W[1]+(xs_d10_f**2)*W[2]+(xs_d10_f**3)*W[3]+(xs_d10_f**4)*W[4]+(xs_d10_f**5)*W[5]+(xs_d10_f**6)*W[6]+(xs_d10_f**7)*W[7]+(xs_d10_f**8)*W[8]+(xs_d10_f**9)*W[9]+(xs_d10_f**10)*W[10]


# In[19]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,100)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# # (d) Data Augmentation : 60、160、320

# In[20]:


# 20 data
lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[20,1])

x = np.linspace(-3,3,20).reshape(-1,1)

y = 2*x + e

loss = np.empty([4,3])


# In[21]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,100)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold")

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[22]:


# 60 data
lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[60,1])

x = np.linspace(-3,3,60).reshape(-1,1)

y = 2*x + e


# In[23]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=60)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=60)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=60)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[24]:


#160 data
lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[160,1])

x = np.linspace(-3,3,160).reshape(-1,1)

y = 2*x + e


# In[25]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=160)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=160)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=160)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[26]:


# 320 data
lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[320,1])

x = np.linspace(-3,3,320).reshape(-1,1)

y = 2*x + e


# In[27]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=320)

W = polynomialRegression(x_train,y_train,14)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=320)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=320)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = polynomialRegression(x_train[i],y_train[i],14)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# # Data train_loss compare

# In[28]:


tb2 = pt.PrettyTable()
tb2.field_names = ["Data","Training Error","Leave-One-Out","Five-Fold"]
tb2.add_row(["20",loss[0][0],loss[0][1],loss[0][2]])
tb2.add_row(["60",loss[1][0],loss[1][1],loss[1][2]])
tb2.add_row(["160",loss[2][0],loss[2][1],loss[2][2]])
tb2.add_row(["320",loss[3][0],loss[3][1],loss[3][2]])
print(tb2)


# # (e) Regularization

# In[29]:


lower,upper = 0,1
mu,sigma = 0,1
e = truncnorm.rvs((lower - mu)/sigma,(upper - mu)/sigma,loc = mu,scale=sigma,size=[20,1])

x = np.linspace(-3,3,20).reshape(-1,1)

y = 2*x + e

loss = np.empty([4,3])


# In[30]:


def Ridge(x,y,deg,alpha):
    X = np.concatenate((np.ones([x.shape[0],1]),x),axis=1)
    for i in range(deg-1):
        X = np.concatenate((X,x**(i+2)),axis=1)
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*np.identity(deg+1)),X.T),y)
    return W


# In[31]:


# alpha = 0 
m = 20
alpha = 0/m


# In[32]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=20)

W = Ridge(x_train,y_train,14,alpha)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[0][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[33]:


#alpha = 0.001/20
m = 20
alpha = 0.001/m


# In[34]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=20)

W = Ridge(x_train,y_train,14,alpha)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[1][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[35]:


#alpha = 1/20
m = 20
alpha = 1/m


# In[36]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=20)

W = Ridge(x_train,y_train,14,alpha)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[2][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[37]:


#alpha = 1000/20
m = 20
alpha = 1000/m


# In[38]:


#degree 14
#traing_err
x_train,y_train,x_test,y_test = preprocessing(x,y,None,num=20)

W = Ridge(x_train,y_train,14,alpha)

y_hat = W[0]+x_train*W[1]+(x_train**2)*W[2]+(x_train**3)*W[3]+(x_train**4)*W[4]+(x_train**5)*W[5]+(x_train**6)*W[6]+(x_train**7)*W[7]+(x_train**8)*W[8]+(x_train**9)*W[9]+(x_train**10)*W[10]+(x_train**11)*W[11]+(x_train**12)*W[12]+(x_train**13)*W[13]+(x_train**14)*W[14]
train_err = mae(y_hat,y_train)
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][0] = train_err
print("train_err:",train_err,"test_err:",test_err)

xs_d14_tr = np.linspace(-3,3,200)
ys_d14_tr = W[0]+xs_d14_tr*W[1]+(xs_d14_tr**2)*W[2]+(xs_d14_tr**3)*W[3]+(xs_d14_tr**4)*W[4]+(xs_d14_tr**5)*W[5]+(xs_d14_tr**6)*W[6]+(xs_d14_tr**7)*W[7]+(xs_d14_tr**8)*W[8]+(xs_d14_tr**9)*W[9]+(xs_d14_tr**10)*W[10]+(xs_d14_tr**11)*W[11]+(xs_d14_tr**12)*W[12]+(xs_d14_tr**13)*W[13]+(xs_d14_tr**14)*W[14]

#levOneOut
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"levOneOut",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(y_hat - y_val[i])
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(np.power(err,2))/err.shape[0]
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][1] = train_err
print("train_lev_err:",train_err,"test_err:",test_err)

xs_d14_lev = np.linspace(-3,3,200)
ys_d14_lev = W[0]+xs_d14_lev*W[1]+(xs_d14_lev**2)*W[2]+(xs_d14_lev**3)*W[3]+(xs_d14_lev**4)*W[4]+(xs_d14_lev**5)*W[5]+(xs_d14_lev**6)*W[6]+(xs_d14_lev**7)*W[7]+(xs_d14_lev**8)*W[8]+(xs_d14_lev**9)*W[9]+(xs_d14_lev**10)*W[10]+(xs_d14_lev**11)*W[11]+(xs_d14_lev**12)*W[12]+(xs_d14_lev**13)*W[13]+(xs_d14_lev**14)*W[14]

#five-fold
x_train,y_train,x_val,y_val,x_test,y_test = preprocessing(x,y,"fiveFold",num=20)

best = 1000000000000000
best_w = np.empty([2,1]).astype(float)
err = np.empty(x_train.shape[0]).astype(float)
for i in range(x_train.shape[0]):
    W = Ridge(x_train[i],y_train[i],14,alpha)
    y_hat = W[0]+x_val[i]*W[1]+(x_val[i]**2)*W[2]+(x_val[i]**3)*W[3]+(x_val[i]**4)*W[4]+(x_val[i]**5)*W[5]+(x_val[i]**6)*W[6]+(x_val[i]**7)*W[7]+(x_val[i]**8)*W[8]+(x_val[i]**9)*W[9]+(x_val[i]**10)*W[10]+(x_val[i]**11)*W[11]+(x_val[i]**12)*W[12]+(x_val[i]**13)*W[13]+(x_val[i]**14)*W[14]
    err[i] = np.sum(np.power((y_hat-y_val[i]),2))
    if err[i] <= best:
        best = err[i]
        best_w = W
train_err = np.sum(err)/15
y_hat = W[0]+x_test*W[1]+(x_test**2)*W[2]+(x_test**3)*W[3]+(x_test**4)*W[4]+(x_test**5)*W[5]+(x_test**6)*W[6]+(x_test**7)*W[7]+(x_test**8)*W[8]+(x_test**9)*W[9]+(x_test**10)*W[10]+(x_test**11)*W[11]+(x_test**12)*W[12]+(x_test**13)*W[13]+(x_test**14)*W[14]
test_err = mae(y_hat,y_test)
loss[3][2] = train_err
print("train_fold_err:",train_err,"test_err:",test_err)

xs_d14_f = np.linspace(-3,3,200)
ys_d14_f = W[0]+xs_d14_f*W[1]+(xs_d14_f**2)*W[2]+(xs_d14_f**3)*W[3]+(xs_d14_f**4)*W[4]+(xs_d14_f**5)*W[5]+(xs_d14_f**6)*W[6]+(xs_d14_f**7)*W[7]+(xs_d14_f**8)*W[8]+(xs_d14_f**9)*W[9]+(xs_d14_f**10)*W[10]+(xs_d14_f**11)*W[11]+(xs_d14_f**12)*W[12]+(xs_d14_f**13)*W[13]+(xs_d14_f**14)*W[14]


# In[39]:


tb3 = pt.PrettyTable()
tb3.field_names = ["alpha","Training Error","Leave-One-Out","Five-Fold"]
tb3.add_row(["0",loss[0][0],loss[0][1],loss[0][2]])
tb3.add_row(["0.001/m",loss[1][0],loss[1][1],loss[1][2]])
tb3.add_row(["1/m",loss[2][0],loss[2][1],loss[2][2]])
tb3.add_row(["1000/m",loss[3][0],loss[3][1],loss[3][2]])
print(tb3)

