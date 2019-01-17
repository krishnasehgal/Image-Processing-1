
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
UBIT='ksehgal';
from copy import deepcopy
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy
import cv2
from matplotlib import pyplot as plt


# In[2]:


def dist (a,b,axis):
        return np.linalg.norm(a-b,axis=axis)


# In[3]:


def mykmean(k,z):
    width, height, depth = z.shape
    reshape_z = np.reshape(
    z, (width * height, depth))   
    a=0
    labels=[]
    label=[0 for j in range(len(reshape_z))]
    for i in range(k):
        labels.append(a)
        a+=1
    #print(labels)
    x=np.array(reshape_z)
    x = (x-x.min())/(x.max()-x.min())
    print(x)

    center=np.random.randint(k, size=(k,3))
    center=x[:k,:]
    
    C_old = np.zeros(center.shape)
    
    error = dist(center, C_old, None)
    print(error)
    distances=[]
    while error != 0:
        clusters=[]
        for i in range(len(x)):
            distances=[]
            for j in range(k):
                distances.append(dist(x[i], center[j],None))
            cluster=distances.index(min(distances))
            clusters.append(cluster)
            
        C_old = deepcopy(center)
        x *= 255.0/x.max()
    # Finding the new centroids by taking the average value
        for i in range(k):
            points=[]
            for j in range(len(x)):
                if clusters[j]==i:
                    points.append(x[j,:])
                    label[j]=labels[i]  
            
            center[i] = np.mean(points, axis=0)
           
        error = dist(center, C_old, None)
    return center, clusters, label,x
    
    


# In[ ]:


img = cv2.imread('/Users/krishna/Downloads/data/baboon.jpg')

width, height, depth = img.shape
z = np.float32(img)

k = [3,5,10,20]
i=0
lis=['task3_baboon_3.png','task3_baboon_5.png','task3_baboon_10.png','task3_baboon_15.png']
for item in k:
    center,cluster,label,x = mykmean(item,z)
    center = np.uint8(center)
    label=np.asarray(label)
    print(label.shape)
    print(label)

    print(center[0])
    res=center[label]
    print(res.shape)
    print(img.shape)
    res2=res.reshape((img.shape)).astype('uint8')
    res2=np.asarray(res2)

    cv2.imwrite(lis[i],res2)
    i+=1
    

