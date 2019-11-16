# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:25:03 2019

Description: 
Functions and Classes for image manipulation.

@author: David 
Github: DTSquid
"""

#import libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

#import matplotlib.pyplot as plt
#import os as os

#auto kmeans clustering of colors
def get_kmeans_clusters(image, div=1.5, num_clusters=None):
    
    #convert to numpy array
    img = np.array(image)
    
    #flatten the image 
    flat_image=np.reshape(img, [-1, 3])
    
    #if thenumber of clusters isn't given, find a suitable number
    if num_clusters == None:
        #estimate the bandwidth of the image using sklearn.cluster
        bandwidth2 = estimate_bandwidth(flat_image, quantile=.05, n_samples=200)
        num_clusters=int(bandwidth2/div)
        
    kmeans = KMeans(n_clusters=num_clusters)   
    kmeans.fit(flat_image)
    labels = kmeans.predict(flat_image)

    H,W = img.shape[:2]
    im = np.reshape(labels, [H,W])
    
    return(im)

#get the edges
def get_edges(image, sigma=0.33):
    
    #get the image gray scale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #get the median single channel (gray) intensity
    v = np.median(gray)
 
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma)*v))
    upper = int(min(255, (1.0 + sigma)*v))
    edged = cv2.Canny(gray, lower, upper)
 
	# return the edged image
    return edged

#get the corners from the image
def get_corners(image):
    
    #make image gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #get corners
    dst = cv2.cornerHarris(gray,2,3,0.04) 
    H,W = dst.shape
    corners = np.zeros((H,W,3), dtype=int)
    corners[dst>0.01*dst.max()]=[0,0,255]
    
    return(corners)
    
#get the regions
def get_ms_clusters(image, channels=3, quantile=0.1, number_of_samples=100):
    img = np.array(image)
    flat_image = np.reshape(img, [-1, channels])
    bandwidth2 = estimate_bandwidth(flat_image, quantile=quantile, n_samples=number_of_samples)
    ms = MeanShift(bandwidth2, bin_seeding=True)
    ms.fit(flat_image)
    labels=ms.labels_

    H,W = img.shape[:2]
    im = np.reshape(labels, [H,W])
    return(im)
    
#from regions get get colored regions
#im = get_regions(img)

def color_clusters_image(clusters,image):
    img = np.array(image)
    groups = np.unique(clusters)
    arr = np.zeros(img.shape, dtype=int)

    for i in groups:
    
        colors = img[np.where(clusters == i)]
        col = np.zeros(3)
        col[0] = int(np.mean(colors[:,0]))
        col[1] = int(np.mean(colors[:,1]))
        col[2] = int(np.mean(colors[:,2]))
        arr[np.where(clusters == i)] = col
    
    return(arr)
    
#just get all the regions colors
def get_cluster_colors(clusters,image):
    img = np.array(image)
    groups = np.unique(clusters)
    colors = np.zeros((len(groups),3))

    for i in groups:
        colors = img[np.where(clusters == i)]
        col = np.zeros(3)
        col[0] = int(np.mean(colors[:,0]))
        col[1] = int(np.mean(colors[:,1]))
        col[2] = int(np.mean(colors[:,2]))
        colors[i] = col
    
    return(colors)
    
