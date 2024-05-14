import numpy as np
import glob
from matplotlib import image
import cv2
import matplotlib.pyplot as plt

def load_images(directory):
    images = []
    for filename in glob.glob(directory+'*.jpg'):
        img = np.array(image.imread(filename))
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gimg)

        height, width = gimg.shape
        
    return images

def plot_reduced_data(X):
    plt.figure(figsize=(12,12))
    plt.scatter(X[:,0], X[:,1], s=60, alpha=.5)
    for i in range(len(X)):
        plt.text(X[i,0], X[i,1], str(i),size=15)
    plt.show()