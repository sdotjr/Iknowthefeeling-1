from math import sqrt, floor, dist
import numpy as np
import random


#Create random centroids
def create_centroids(ds, k):
    """
    Create random cluster centroids.
    
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """
    np.random.seed(30)

    return ds[np.random.choice(ds.shape[0], k, replace=False), :]

#Find closest centroid to each point and label
def label(ds, centroids):
  labels = []
  for i in ds:
    dists = []
    for c in centroids:
      dists.append(np.linalg.norm(i-c))
      
    labels.append(dists.index(min(dists)))
  
  return np.array(labels)


#Move centroid to center of each point (Average of each coordinate)

def center(ds, centroids, labels):
  new_centroids = []
  
  for i in range(len(centroids)):
    points = ds[np.where(labels==i)]
    new_cent = []
    new_cent.append(points.mean(axis=0))
    new_centroids.append(new_cent[0])

  return np.stack(new_centroids)
  


#Relabel and Recenter until no more movement

def fit(ds,k):
  centroids = create_centroids(ds,k)
  labels = label(ds, centroids)
  new_cents = center(ds,centroids,labels)

  for i in range(200):
    labels = label(ds, new_cents)
    new_cents = center(ds,new_cents,labels)



  return labels
