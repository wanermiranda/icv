#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import sys
import glob 
from timer import Timer
import threading
import time

base_scale = 1 # used to reduce the amount of data threated
slide_window_width = 0 # to be initilized further
stride = 10 # step to slide 
NORM = 1
last_crop = 0
maxThreads = 4

queryIndex = 1

class WindowSlider (threading.Thread):
    def __init__(self, threadID, query, target, column):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.query = query
        self.target = target
        self.column = column
        self.best = -1

    def run(self):  
        y = self.column      
        slide_window_height = slide_window_width
        x = slide_window_width
        target_height, target_width  = self.target.shape[:2]

        while( x < target_width):                        
            calcY = y-slide_window_height
            calcX = x-slide_window_width
            crop = self.target[calcY: y, calcX: x]  
            
            MSDiff = getMeanSquareDiff(crop, self.target)
            (value, angle) = find_nearest(angledImages, MSDiff)
            diff = (MSDiff - value)**2

            if ((self.best >= diff) or (self.best == -1)):
                self.best = diff
                self.crop = crop
                self.angle = angle
                print x

            x = x + stride
            
            if (x > target_width):                
                x = target_width 

def find_nearest(array,value):
    idx = np.abs(np.subtract.outer(array, value)).argmin(0)
    return (array[idx], idx)

def getSquareCenterCrop(img):
    (h, w) = img.shape[:2]
    
    centerH = int(h/2)
    centerW = int(w/2)

    blockSize = h if (h < w) else w
    print "block size " + str(blockSize)    
    query_color = cv2.resize(query_color,None,fx=2.0, fy=2.0, interpolation = cv2.INTER_CUBIC)
    
    return crop

def getMeanSquareDiff(img, target):
    (m, n) = img.shape[:2]    
    summ = 0.0
    for j in range(m):
        for i in range(n):
            if target[j,i] != 0:
                summ += np.subtract(img[j][i], target[j][i], dtype=np.float64) ** 2

    return summ/(m*n)



def rotateImg (img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
     
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

#def quad_diff(img, target):


def image_hist(img):
    colors_hist = np.zeros((3,256)) 
    for row in img:
        for column in row:
            for i in range(3):
                color = column[i]
                colors_hist[i][color] = colors_hist[i][color] +1
    maxB = colors_hist[0].max()
    maxG = colors_hist[1].max()
    maxR = colors_hist[2].max()
    for idx in range(256):    
        colors_hist[0][idx] *= NORM/maxB
        colors_hist[1][idx] *= NORM/maxG
        colors_hist[2][idx] *= NORM/maxR
    return colors_hist


def dist_bin(A, B): 
    diff = np.zeros((3,256))
    for row in range(256):
        for band in range(3):
            diff[band][row] = (A[band][row] - B[band][row])**2
            diff[band][row] = (A[band][row] - B[band][row])**2
            diff[band][row] = (A[band][row] - B[band][row])**2
    
    return diff.sum()


dataset_query = '/home/gorigan/datasets/icv/tp1/imagens/query/'
dataset_target_sem_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/sem_ruido/'
dataset_target_com_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/com_ruido/'

queryList = ["001_apple_obj.png"
			,"002_dumbphone_obj.png"
			,"003_japan_obj.png"
			,"004_yen_obj.png"
			,"005_bottle_obj.png"
			,"006_shoe_obj.png"
			,"007_kay_obj.png "
			,"008_starbucks_obj.png"
			,"009_coca_obj.png"]

query_color = cv2.imread(dataset_query + queryList[queryIndex-1])
query_color = cv2.resize(query_color,None,fx=base_scale, fy=base_scale, interpolation = cv2.INTER_CUBIC)
height, width = query_color.shape[:2]
print "Query Image -> h: " + str(height) + " w:" + str(width)

print "Pre-processing image"
query_mono = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
query_mono = getSquareCenterCrop(query_mono)
height, width = query_mono.shape[:2]
print "Query Center Image -> h: " + str(height) + " w:" + str(width)

#plt.imshow(query_mono, cmap = plt.get_cmap('gray'))
#plt.show()        

angledImages = []
for idx in range(41):
    angle= idx
    img  = rotateImg(query_mono, angle)
    diff = getMeanSquareDiff(query_mono, img)
    angledImages.append(diff);    
    if (((angle) % 10) == 0):
        print "Angle " + str(angle)
        print diff
        sys.stdout.flush()


print "End pre-processing image"

print 
'''(value, angle) = find_nearest(angledImages, 666.0)
print str(value) + " at " + str(angle) + "o"'''
slide_window_width   =  width
slide_window_height  = height

for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
    print target_image_path
    target_color = cv2.imread(target_image_path)
    target_mono = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)    
    target_mono = cv2.resize(target_mono,None,fx=base_scale, fy=base_scale, interpolation = cv2.INTER_CUBIC)    
    target_height, target_width = target_mono.shape[:2]
    print "h: " + str(target_height) + " w:" + str(target_width)

    y = slide_window_height    
    local_diff = -1 
    local_crop = None
    while( y < target_height):
        threads = []            
        for t_index in range(maxThreads): 
            l_thread =  WindowSlider(t_index, query_mono, target_mono, y)
            l_thread.start()            
            threads.append(l_thread)
            y = y + stride
            print "Line :" + str(y)
            if (y > target_height):
                y = target_height
        
        for t in threads:
            t.join()  

        for t in threads:
            if ((t.best <= local_diff) or (local_diff == -1)):                                        
                local_diff = t.best
                local_crop = t.crop
                print "Thread: "  + str(t.threadID) + " Value: "+ str(local_diff)


    plt.imshow(local_crop, cmap = plt.get_cmap('gray'))
    plt.show()        
