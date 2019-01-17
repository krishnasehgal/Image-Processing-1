
# coding: utf-8

# In[1]:


UBIT='ksehgal';
import random
import math
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt
import warnings



#Euclidian Distance between two d-dimensional points
def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)

#K-Means Algorithm
def kmeans(k,datapoints):
    
    # d - Dimensionality of Datapoints
    d = len(datapoints[0]) 
    
    #Limit our iterations
    Max_Iterations = 1000
    i = 0
    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    #Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        cluster_centers=  np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
        c1_x=cluster_centers[:,:1]
        c1_y=cluster_centers[:,1:]
        f1=datapoints[:,:1]
        f2=datapoints[:,1:]
        
        plt.scatter(f1, f2,marker="^",facecolors='none',edgecolors='b', s=30)
        colors=["red","green","blue"]
        plt.scatter(c1_x, c1_y, marker=".", s=200, c=colors)
        plt.savefig("/Users/krishna/Desktop/MLPROJECT2/task3_iter1_a")
    l=0
    while (cluster != prev_cluster) or (i > Max_Iterations):
        prev_cluster = list(cluster)
        i += 1
        #Update Point's Cluster 
        for p in range(0,len(datapoints)):
            min_dist = float("inf")
            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):
                dist = eucldist(datapoints[p],cluster_centers[c])
                if (dist < min_dist):
                    min_dist = dist
                    cluster[p] = c   # Reassign Point to new Cluster       
        print("======== Iteration :",i-3,"========")
        print("Clusters",cluster_centers)
        print("Assignments",cluster)
        
        print("                         ")
        print("                         ")         
        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1
            
            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                
               
            
            cluster_centers[k] = new_center
        
        str=["task3_iter1_b","task3_iter2_a","task3_iter2_b"]
        
        #print("======== Results ========")
        #print("Clusters",cluster_centers)
        print("Assignments",cluster)
        fig, ax = plt.subplots()
        c1_x=cluster_centers[:,:1]
        c1_y=cluster_centers[:,1:]
        colors=["red","green","blue"]
        dataPointColorSet = []
        for x in range(0,len(cluster)):
            dataPointColorSet.append(colors[cluster[x]])
            
        print(dataPointColorSet) 
        plt.scatter(datapoints[:,:1], datapoints[:,1:],facecolors=dataPointColorSet,edgecolors=dataPointColorSet, s=30)
        plt.scatter(c1_x, c1_y,marker="^",facecolors=colors,edgecolors=colors, s=30)
        print(l)
        plt.savefig("/Users/krishna/Desktop/MLPROJECT2/"+str[l])
        l+=1
if __name__ == "__main__":
    
    
    datapoints = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
    
    #print(datapoints)
    k = 3 # K - Number of Clusters
      
    kmeans(k,datapoints) 

