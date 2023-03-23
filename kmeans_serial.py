
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from sklearn.datasets.samples_generator import make_blobs
from timeit import default_timer as timer
from datetime import timedelta


# In[50]:


from numpy import genfromtxt


# In[51]:


k = 10
loop_itr = 20


# In[52]:



def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


# In[53]:

#define intial centroid
def initial_centroid(data, k): 
    centroid = []
    x = np.random.randint (data.shape[0])
    centroid.append(data [x, :] )
    for c_id in range(k - 1):
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
            for j in range(len(centroid)):
                temp_dist = distance(point, centroid[j])
                d = min(d, temp_dist)
            dist.append(d)
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroid.append(next_centroid)
    return centroid


# In[54]:

#define data to the nearest cluster
def assign_centroid(data, centroid, count):
    new_data = []
    for i in range(len(data)):
        dist = []
        for j in range(len(centroid)):
            
            dist.append(distance(data[i], centroid[j]))
        new_data.append(dist)
  
    label = [] # label of all data after clustering   
    data_assigned = []
    for i in range(len(data)):
        label.append(new_data[i].index(min(new_data[i])))
        data_changed = np.append(data[i],np.array(label[i]))
        data_assigned.append(data_changed)

    return data_assigned, label


# In[55]:

#update centroids after clustering
def update_centorid(data_cent, centroid):
    data_mean = []
    for j in range(len(centroid)):
        
        x = []
        for i in range(len(data_cent)):
    
            if data_cent[i][2] == float(j):
                
                x.append(data_cent[i][:2])
               
        data_mean.append(x)
    new_cent = [] #new centroids
    sum = 0
    for i in range(len(data_mean)):
        
        new_cent.append(np.mean(data_mean[i][:],axis = 0 ))
    return new_cent


# In[143]:


#data, y = make_blobs(n_samples=10, centers=k, random_state=0)

data1 = genfromtxt('E:\\My_Desktop\\final_proj\\code\\test_data_102.csv', delimiter=',')
data = data1[1:]

start1 = timer()
centroid = initial_centroid(data, k)
print("Firstcentroid",centroid,'\n')
end1 = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end1 -start1))

assigned_cent, label = assign_centroid(data, centroid,0)



start1 = timer()
count = 0
for i in range(loop_itr):
    count += 1
    centroid = update_centorid(assigned_cent, centroid)
    assigned_cent, label = assign_centroid(data, centroid, count)
  


end1 = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end1 -start1))
#print('thisISlabel:', label)
print('this_is_centroid:', centroid)


# In[144]:


len(data)


# In[145]:


centroid[1]


# In[146]:


y1 = genfromtxt('E:\\My_Desktop\\final_proj\\code\\y_102.csv', delimiter=',')


# In[147]:


y = y1[1:]


# In[148]:


y = y.astype(int)


# In[149]:


len(y)


# In[150]:


list_array = np.array(label)


# In[151]:


len(data)


# In[152]:


y


# In[153]:


list_array


# In[154]:


plt.scatter(data[:, 0], data[:, 1], c=list_array)


# In[155]:


plt.scatter(data[:, 0], data[:, 1], c=y)


# In[156]:


my_label = []
for i in range(k):
    my_label.append(len(np.where(list_array == i)[0]))


# In[157]:


true_label = []
for i in range(k):
    true_label.append(len(np.where(y == i)[0]))
        


# In[158]:


error = np.sum(np.absolute([my_label - true_label for my_label, true_label in zip(my_label, true_label)]))
((len(y) - error) / len(y)) * 100


# In[97]:


count = 0
for i in range(len(y)):
    
    if label[i] != y[i]:
        count += 1
print(count)
accuracy = ((len(y) - count )/len(y) )*100
print('accuracy:', accuracy)







