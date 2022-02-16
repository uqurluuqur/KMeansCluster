# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:17:44 2021

@author: Pc
"""

import numpy as np
from PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1234)
import os




class KMeans:

    def __init__(self,k, max_iters ,plot_steps):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        
    def PCA_(self, x):
        pca = PCA(2) 
        pca.fit(x)
        xy = pca.transform(x)
        return xy
    
    def random_centroids(self,xy):
        """
        ----------
        Chooses random centroids
        -------
        """
        random = np.random.choice(len(xy), self.k, replace=False)
        return (xy[random,:])
    
    
    
    def new_centroids(self,df):
        returnle=[]
        for i in range(self.k):
            mean=[]
            the_cluster=df.loc[df['cluster']==i]
            the_cluster= the_cluster.drop(['cluster'],axis= 1)
            #print(the_cluster.head(),"dffff")
            the_cluster=the_cluster.to_numpy()
            #print(the_cluster.shape,'hüüüüüüüüü')
            deneme= np.mean(the_cluster,axis=0)
            #print('deneme',deneme.shape)
            returnle.append(deneme)
            
        returnle= np.array(returnle)
        #print(returnle.shape,"asdasddasdasd")
        return returnle

    def plotting_(self,x,cluster,centroids):
        x=self.PCA_(x)
        #print(centroids.shape,'pcasız')
        centroid=self.PCA_(centroids)
        #print(centroid.shape,'pcalı ')
        groups = np.unique(cluster)
        for i in groups:
            plt.scatter(x[cluster == i , 0] , x[cluster == i , 1] , label = i)
            
        for j in range(self.k):
            #print(centroids[j,:])
            centroid= self.PCA_(centroids)
            plt.scatter(centroid[j][0],centroid[j][1],marker="+",color = "black")
        plt.show()
        return

    def predict(self,x):
        centroids= self.random_centroids(x)
        s=0
        
        while s<self.max_iters:
            cluster=[]  
            #clustering
            for row in x:
                distances= []
                for centroid in centroids:
                    dist= np.sum(abs((row**2)-(centroid**2)))
                    dist= np.sqrt(dist)
                    distances.append(dist)
                cluster.append(distances.index(min(distances))) 

            df= pd.DataFrame(data=x)
            df['cluster']= cluster

            print(s)
   
            df=df.drop(['cluster'],axis=1)
            if self.plot_steps:
                self.plotting_(df,cluster,centroids)
            centroid_prev=centroids
            centroids= self.new_centroids(df)#new centroids
            if np.array_equal(centroid_prev,centroids):
                break
            s=s+1
        return cluster

                    
                
            
        