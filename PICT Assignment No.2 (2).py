

# In[5]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd


# In[6]:


#Uploading the Dataset from the system
dataset= pd.read_csv('Mall_Customers.csv')
dataset.describe()


# In[9]:


#Importing the Dataset
Data=pd.read_csv('Mall_Customers.csv')
print(Data)


# In[10]:


#Handling Duplicates Data #Remove Duplicates
dataset=dataset.drop_duplicates()
print(dataset)


# In[5]:


#Handling Missing Data
#Print the top few rows of the Dataset
dataset.head()


# In[11]:


#Checking Null Values Category Wise
dataset.isnull().sum()


# In[12]:


#Import Label Encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
Data['Genre']=label_encoder.fit_transform(Data['Genre'])
print(Data.head())


# In[28]:


Data.dtypes


# In[93]:


# K Means Clustering Algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[94]:


#Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[85]:


#Hierarchical Clustering Algorithm
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  


# In[86]:


dataset = pd.read_csv('Mall_Customers.csv') 
x = dataset.iloc[:, [3, 4]].values 


# In[88]:


import scipy.cluster.hierarchy as shc  
dendro = shc.dendrogram(shc.linkage(x, method="ward"))  
mtp.title("Dendrogrma Plot")  
mtp.ylabel("Euclidean Distances")  
mtp.xlabel("Customers")  
mtp.show()  


# In[89]:


from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
y_pred= hc.fit_predict(x)  


# In[91]:


mtp.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'cyan', label = 'C1')  
mtp.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green', label = 'C2')  
mtp.scatter(x[y_pred== 2, 0], x[y_pred == 2, 1], s = 100, c = 'magenta', label = 'C3')  
mtp.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'orange', label = 'C4')  
mtp.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'violet', label = 'C5')  
mtp.title('Clusters of customers')  
mtp.xlabel('Annual Income')  
mtp.ylabel('Spending Score')  
mtp.legend()  
mtp.show()  


# In[14]:


# Accuracy
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
dataset= pd.read_csv('Mall_Customers.csv')
names = ['CustomerId', 'Genre', 'Age', 'Annual Income', 'Spending Score']
dataset = pandas.read_csv(dataset, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:





# In[ ]:





# In[ ]:




