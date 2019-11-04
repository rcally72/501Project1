# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:57:54 2019

@author: Ruchikaa Kanar
"""
#### Ruchi Code ####
dfnew.to_csv('dfnew.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score
from sklearn import preprocessing
import pylab as pl
import sklearn.metrics as sm
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from matplotlib import cm

# I will be picking 3 scenarios to cluster (different attributes) using all 3 clustering techniques.
def Clustering():
    # Import the dataset
    dfnew = pd.read_csv('dfnew.csv')
    Scenario1(dfnew) #For Scenario 1, lets consider the following attributes: Name, Population, MinT, and MaxT.
    Scenario2(dfnew) #For Scenario 2, lets consider the following attributes: Regions and % increase in urban land.
    Scenario3(dfnew) #For Scenario 3, lets consider the following attributes: Name, Temperature, Automobiles Per Capita
    Scenario4(dfnew)  #For Scenario 4, lets consider the following attributes: Name, Number of Factories, MinT, MaxT.
    Dbscan(dfnew) #Overall Dataset.
    #print (dfnew.dtypes)
def Scenario1 (dfnew) :
    print("For Scenario 1, lets consider the following attributes: Name, Population, MinT, and MaxT.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[0,1,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
   
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=2, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=2, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=2, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.title('Graphing The Cluster for Scenario 1')
    plt.legend()
    plt.show()

    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew['Name'] = pd.Categorical(dfnew['Name'])
    dfnew['Name'] = dfnew['Name'].cat.codes
    dfnew.shape[1]
    dfnew=pd.concat([dfnew['Name'], dfnew['Population'], dfnew['MinT'], dfnew['MaxT']], 
                 axis=1, keys=['Name', 'Population', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 500
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    k2 = 100
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title('PCA Graph for Scenario 1')
    plt.show()
    
    # Clear plot
    plt.clf()
    
def Scenario2 (dfnew):
    print("For Scenario 2, lets consider the following attributes: Regions and % increase in urban land.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[20,21,22,23,24,25,26,27]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=5, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')
    plt.title('Graph for Scenario 2')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['NE'], dfnew['SE'], dfnew['PS'],dfnew['NC'],dfnew['SC'],dfnew['RM'],dfnew['GP'],dfnew['PN'], dfnew['Percent increase in urban land 2000-2010']],
                 axis=1, keys=['NE', 'SE', 'PS', 'NC','SC','RM','GP','PN','Percent increase in urban land 2000-2010'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 5
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 10
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentional Plot for Scenario 2")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()
def Scenario3 (dfnew):
    print("For Scenario 3, lets consider the following attributes: Name, MinT, MaxT, and Automobile Per Capita.")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[0,9,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Ward:')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: Using linkage type Complete :')
    plt.show()
    plt.clf()
    
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram using linkage type Average :')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=3, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')
    plt.title('Graph for Scenario 3')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['Name'], dfnew['Automobiles Per Capita'], dfnew['MinT'],dfnew['MaxT']],
                 axis=1, keys=['Name', 'Automobiles Per Capita', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 35
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 250
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentional Plot for Scenario 3")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()
    
def Scenario4 (dfnew):
    print("For Scenario 4, lets consider the following attributes: Name,Number of Factories, MinT, MaxT,")
    print()   
    print("Results for Hierarchical Clustering:")
    print()
    dfnew.fillna(method ='ffill', inplace = True) 
    X=dfnew.iloc[:,[0,30,31,32]].values
    labelencoder_X= LabelEncoder ()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
       
    Y= pd.DataFrame(X)
    #Creating a dendogram to find the optimal number of clusters for the columns mentioned above.
    print("Dendrogram: Using linkage type 'Ward':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Ward')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Complete':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'complete')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Complete')
    plt.show()
    plt.clf()
    print("Dendrogram: Using linkage type 'Average':")
    dendrogram= sch.dendrogram(sch.linkage(X, method= 'average')) #Let's use the ward method first.
    plt.title('Dendrogram: using linkage: Average')
    plt.show()
    plt.clf()
    # Fitting H.Clustering to the dataset
    
    #TYPE 1= Affinity: Euclidean and Linkage Type: Ward
    hc= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'ward')
    y_hc= hc.fit_predict(X)
    accuracy1 = sm.accuracy_score(y_hc, hc.labels_)
    print("The Accuracy Score for Type 1 is : " + str(accuracy1))
    
    #TYPE 2= Affinity: Euclidean and Linkage Type: Complete
    hc2= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage = 'complete')
    y_hc2= hc2.fit_predict(X)
    accuracy2 = sm.accuracy_score(y_hc2, hc2.labels_)
    print("The Accuracy Score for Type 2 is : " + str(accuracy2))
    
    #TYPE 3= Affinity: Manhattan and Linkage Type: Average
    hc3= AgglomerativeClustering(n_clusters=3, affinity= 'manhattan', linkage = 'average')
    y_hc3= hc3.fit_predict(X)
    accuracy3 = sm.accuracy_score(y_hc3, hc3.labels_)
    print("The Accuracy Score for Type 3 is : " + str(accuracy3))
    
    ####Lets visualize the clusters######
    plt.scatter(X[y_hc== 0,0], X[y_hc== 0,1], s=5, c= 'blue', label= 'Cluster 1')
    plt.scatter(X[y_hc== 1,0], X[y_hc== 1,1], s=5, c= 'red', label= 'Cluster 2')
    plt.scatter(X[y_hc== 2,0], X[y_hc== 2,1], s=5, c= 'green', label= 'Cluster 3')

    plt.title('Graph for Scenario 4')
    plt.legend()
    plt.show()
    
    print("Results for KMEANS Clustering:")
    print()
    # First remove missing data
    nullCount = dfnew.isnull().values.ravel().sum()
    #print("\nNull Count:\n");
    pprint(nullCount)
    dfnew.dropna()
    #pprint(len(dfnew.index))
    #Conversion to numerical 
    dfnew=pd.concat([dfnew['Name'], dfnew['Number of Factories'], dfnew['MinT'],dfnew['MaxT']],
                 axis=1, keys=['Name', 'Number of Factories', 'MinT', 'MaxT'])
    x = dfnew.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #pprint(normalizedDataFrame[:10])
    # Create clusters (TEST 1)
    k = 80
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    
    # Create clusters (TEST 1)
    k2 = 500
    kmeans = KMeans(n_clusters=k2)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average silhouette_score is :", silhouette_avg)
    CH_avg = calinski_harabasz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k2, "The average CH_score is :", CH_avg) #Using the "calinski_harabasz_score" library, one can calculate the CH score.
    centroids = kmeans.cluster_centers_
    # Checking to see how it fits on different dimensions
    #print(pd.crosstab(cluster_labels, dfnew['Population']))
    #print(pd.crosstab(cluster_labels, dfnew['MinT']))
    #print(pd.crosstab(cluster_labels, dfnew['MaxT']))
    
    # PCA
    print("PCA projection:")
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    
    # This shows how good the PCA performs on this dataset
    print(pca2D.explained_variance_)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("PCA 2-Dimentioanl Plot for Scenario 4")
    
    # Write to file
    #plt.savefig("pca2D.png")
    plt.show()
    # Clear plot
    plt.clf()

def Dbscan(dfnew):
    X, label = make_moons(n_samples=200, noise= 0.15 , random_state=30)
    print (X[:3,])
    model= DBSCAN(eps=0.25, min_samples=12).fit(X)
    print(model)
    fig,ax= plt.subplots(figsize=(10,8))
    sctr= ax.scatter(X[:,0],X[:,1], c= model.labels_, s=140, alpha=0.8)
    plt.title('DBSCAN for the dataset')
    fig.show()
    
if __name__=="__main__":
    Clustering ()
#### End of Ruchi Code###