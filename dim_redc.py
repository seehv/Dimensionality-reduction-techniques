# -*- coding: utf-8 -*-
"""
Created on Thu May 7 14:01:00 2020

@author: Harsha vardhan Seelam 
"""


import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
Axes3D

from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# =====================|| Natural MNIST dataset seperation ||=========================

mnist_, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True) 
mnist_data = mnist_[:10000]
mnist_labels_data = mnist_labels[:10000]
mnist_data_train = mnist_data[:6000]
mnist_data_test = mnist_data[6000:]

# =====================|| Natural OLIVETTI FACES dataset seperation ||================

olivetti_, olivetti_labels = sklearn.datasets.fetch_olivetti_faces(return_X_y=True)
olivetti_faces_train = olivetti_[:300]
olivetti_faces_test = olivetti_[300:]
olivetti_labels_data = olivetti_labels[:400]

# =============================================================================

# =====================|| Function for generating artificial datasets ||=========================

def get_artificial_dataset(nameofdataset):
    X, color = nameofdataset(n_samples=5000, noise=0.0, random_state=5)
    labels = plot(X, color);
    return X, labels

# =====================|| Function for Labeling and plotting Artificial datasets ||=========================

def plot(X, color):
    
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    label = ward.labels_
    print("Number of unique labels: %i" % label.size)
    
    # Plot result
    fig = plt.figure()

    ax = p3.Axes3D(fig)
    for l in np.unique(label):
        ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                   color=plt.cm.jet(np.float(l) / np.max(label + 1)), s=20, edgecolor='k')
    
    print("Number of clusters: ", np.unique(label).size)
    plt.title('Original data')
    plt.show()
    
    return label
# =====================|| Generation of artificial dataset swissroll ||=========================

def swissroll(n_samples=100, noise=0.0, random_state=None):
    generator = check_random_state(random_state)

    t = (3 * np.pi)/2 * (1 + 2 * generator.rand(1, n_samples))
    
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = 30 * generator.rand(1, n_samples)

    X = np.concatenate((x, y, z))
    X += noise * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)

    return X, t

# =====================|| Generation of artificial dataset broken swissroll ||=========================

def brokenswiss(n_samples=100, noise=0.0, random_state=None):
    generator = check_random_state(random_state)
    
    counter = 0
    while(counter < n_samples):
        rand_pi = generator.rand(1, n_samples)
        if((2/5 < rand_pi).all and (rand_pi < 4/5).all ):
            t = (3 * np.pi)/2 * (1 + 2 * rand_pi)
            x = t * np.cos(t)
            y = t * np.sin(t)
            z = 30 * generator.rand(1, n_samples)
            
            X = np.concatenate((x, y, z))
            X += noise * generator.randn(3, n_samples)
            X = X.T
            t = np.squeeze(t)
            counter = counter + 1
                
    
    return X, t

# =====================|| Generation of artificial dataset helix ||=========================

def helix(n_samples=100, noise=0.0, random_state=None):
    generator = check_random_state(random_state)

    t = (3 * np.pi)/2 * (1 + 2 * generator.rand(1, n_samples))
    
    x = (2 + np.cos(8 * t)) * np.cos(t)
    y = (2 + np.cos(8 * t)) * np.sin(t)
    z = np.sin(8 * t)

    X = np.concatenate((x, y, z))
    X += noise * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)

    return X, t

# =====================|| Dimentionality reduction using PCA ||=========================

def pca_dr(data_set, com_dimentions):
    
    print("Dimentions of Dataset before PCA: ", data_set.shape)
    
    pca = PCA(n_components=com_dimentions)
    data_transform = pca.fit_transform(data_set)  # Fit the model with X and Apply dimensionality reduction to X.
    
    print("Dimentions of Dataset after PCA: ", data_transform.shape)
    return data_transform

# =====================|| Dimentionality reduction using Kernel PCA ||=========================

def K_pca_dr(data_set, com_dimentions, kernal_type= 'poly'):
    
    print("Dimentions of Dataset before Kernel PCA: ", data_set.shape)
    
    K_pca = KernelPCA(n_components= com_dimentions, kernel= kernal_type, gamma = None, degree = 5, coef0= 1)
    data_transform = K_pca.fit_transform(data_set)  # Fit the model with X and Apply dimensionality reduction to X.
    
    print("Dimentions of Dataset after Kernel PCA: ", data_transform.shape)
    return data_transform

# =====================|| Dimentionality reduction using LLE ||=========================
   
def LLE_dr(data_set, com_dimentions):
    
    print("Dimentions of Dataset before LLE: ", data_set.shape)
    
    lle = LocallyLinearEmbedding(n_components=com_dimentions)
    data_transform = lle.fit_transform(data_set)  # Fit the model with X and Apply dimensionality reduction to X.
    
    print("Dimentions of Dataset after LLE: ", data_transform.shape)
    return data_transform

# =====================|| Dimentionality reduction using Auto Encoders ||=========================

def autoencoders_reduction(desired_dim, train_data, test_data, ori_dim, whole_data):
    
    ori_dShape = whole_data.shape
    
    desired_dim = desired_dim
    input_data = tf.keras.Input(shape=(ori_dim,))               # this is our input placeholder

    # Hidden layers
    encoded_layer1 = tf.keras.layers.Dense(128, activation='relu')(input_data)
    encoded_layer2 = tf.keras.layers.Dense(64, activation='relu')(encoded_layer1)
    encoded_layer3 = tf.keras.layers.Dense(desired_dim, activation='relu')(encoded_layer2)
    decoded_layer4 = tf.keras.layers.Dense(128, activation='relu')(encoded_layer3)
    decoded_layer5 = tf.keras.layers.Dense(
        ori_dim, activation='sigmoid')(decoded_layer4)
    
    auto_encoder = tf.keras.Model(input_data, decoded_layer5)   #data reconstruction
    encoder = tf.keras.Model(input_data, encoded_layer3)        #Encodedlayer data

    auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    auto_encoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True,
                    validation_data=(test_data, test_data))

    reduced_dim = encoder.predict(whole_data)   # we take the data from the test set to encode and decode some digits
    
    print( "Dimentions of Dataset before AutoEncoders: ", ori_dShape)
    print( "Dimentions of Dataset after AutoEncoders: ", reduced_dim.shape)
    return reduced_dim

# =====================|| Accuracy calculation using 1-KNN ||=========================

def K_NN(num, dr_dataset_train, dr_labels_train, dataset_test, labels_test):
    knn = KNeighborsClassifier(n_neighbors = num)    #Create KNN Classifier of 1 nearest neighbour
    knn.fit(dr_dataset_train, dr_labels_train)
    predict = knn.predict(dataset_test)
    print("Accuracy: ",metrics.accuracy_score(labels_test, predict))


# =============================================================================
#                                MAIN - PROGRAM
# =============================================================================

# =====================|| Generating artificial datasets  ||=========================

swiss_data, labels_sr = get_artificial_dataset(swissroll)
BRswiss_data, labels_bsr = get_artificial_dataset(brokenswiss)
helix_data, labels_hx = get_artificial_dataset(helix)

# =====================|| Dimentionality reduction fucntion call for PCA ||=========================

dr_pca_sr = pca_dr(swiss_data, 2)
dr_pca_bsr = pca_dr(BRswiss_data, 2)
dr_pca_hx = pca_dr(helix_data, 1)
dr_pca_mt = pca_dr(mnist_data, 20)
dr_pca_of = pca_dr(olivetti_, 20)

# =====================|| Dimentionality reduction fucntion call for KPCA ||=========================

dr_Kpca_sr  = K_pca_dr(swiss_data, 2)
dr_Kpca_bsr  = K_pca_dr(BRswiss_data, 2)
dr_Kpca_hx  = K_pca_dr(helix_data, 2)
dr_Kpca_mt  = K_pca_dr(mnist_data, 20)
dr_Kpca_of  = K_pca_dr(olivetti_, 20)

# =====================|| Dimentionality reduction fucntion call for LLE ||=========================

dr_lle_sr  = LLE_dr(swiss_data, 2)
dr_lle_bsr  = LLE_dr(BRswiss_data, 2)
dr_lle_hx  = LLE_dr(helix_data, 2)
dr_lle_mt  = LLE_dr(mnist_data, 20)
dr_lle_of  = LLE_dr(olivetti_, 20)

# =====================|| Dimentionality reduction fucntion call for Auto_encoders ||=========================

dr_ae_sr = autoencoders_reduction(2, swiss_data[:4000], swiss_data[4000:], 3, swiss_data)
dr_ae_bsr = autoencoders_reduction(2, BRswiss_data[:4000], BRswiss_data[4000:], 3, BRswiss_data)
dr_ae_hx = autoencoders_reduction(2, helix_data[:4000], helix_data[4000:], 3, helix_data)
dr_ae_mt = autoencoders_reduction(20, mnist_data_train, mnist_data_test, 784, mnist_data)
dr_ae_of = autoencoders_reduction(20, olivetti_[:300], olivetti_[300:], 4096, olivetti_)

# =====================|| Accuracy calculation of reduced dataset PCA ||=========================

K_NN(1, dr_pca_sr[:4000], labels_sr[:4000] , dr_pca_sr[4000:], labels_sr[4000:])
K_NN(1, dr_pca_bsr[:4000], labels_bsr[:4000] , dr_pca_bsr[4000:], labels_bsr[4000:])
K_NN(1, dr_pca_hx[:4000], labels_hx[:4000] , dr_pca_hx[4000:], labels_hx[4000:])
K_NN(1, dr_pca_mt[:6000], mnist_labels_data[:6000] , dr_pca_mt[6000:], mnist_labels_data[6000:])
K_NN(1, dr_pca_of[:300], olivetti_labels[:300] , dr_pca_of[300:], olivetti_labels[300:])

# =====================|| Accuracy calculation of reduced dataset Kernel PCA||=========================

K_NN(1, dr_Kpca_sr[:4000], labels_sr[:4000] , dr_Kpca_sr[4000:], labels_sr[4000:])
K_NN(1, dr_Kpca_bsr[:4000], labels_bsr[:4000] , dr_Kpca_bsr[4000:], labels_bsr[4000:])
K_NN(1, dr_Kpca_hx[:4000], labels_hx[:4000] , dr_Kpca_hx[4000:], labels_hx[4000:])
K_NN(1, dr_Kpca_mt[:6000], mnist_labels_data[:6000] , dr_Kpca_mt[6000:], mnist_labels_data[6000:])
K_NN(1, dr_Kpca_of[:300], olivetti_labels[:300] , dr_Kpca_of[300:], olivetti_labels[300:])

# =====================|| Accuracy calculation of reduced dataset LLE||=========================

K_NN(1, dr_lle_sr[:4000], labels_sr[:4000] , dr_lle_sr[4000:], labels_sr[4000:])
K_NN(1, dr_lle_bsr[:4000], labels_bsr[:4000] , dr_lle_bsr[4000:], labels_bsr[4000:])
K_NN(1, dr_lle_hx[:4000], labels_hx[:4000] , dr_lle_hx[4000:], labels_hx[4000:])
K_NN(1, dr_lle_mt[:6000], mnist_labels_data[:6000] , dr_lle_mt[6000:], mnist_labels_data[6000:])
K_NN(1, dr_lle_of[:300], olivetti_labels[:300] , dr_lle_of[300:], olivetti_labels[300:])

# =====================|| Accuracy calculation of reduced dataset Auto-Encoders||=========================

K_NN(1, dr_ae_sr[:4000], labels_sr[:4000] , dr_ae_sr[4000:], labels_sr[4000:])
K_NN(1, dr_ae_bsr[:4000], labels_bsr[:4000] , dr_ae_bsr[4000:], labels_bsr[4000:])
K_NN(1, dr_ae_hx[:4000], labels_hx[:4000] , dr_ae_hx[4000:], labels_hx[4000:])
K_NN(1, dr_ae_mt[:6000], mnist_labels_data[:6000] , dr_ae_mt[6000:], mnist_labels_data[6000:])
K_NN(1, dr_ae_of[:300], olivetti_labels[:300] , dr_ae_of[300:], olivetti_labels[300:])

# =====================|| END||=========================
