#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import sys
import glob 
from timer import Timer
import threading
import time

slide_window_width = 140 # the height is to be calculed based on the proportion of the query image
stride = 10 # step to slide 
NORM = 1
last_crop = 0
maxThreads = 4
queryIndex = 3

class WindowSlider (threading.Thread):
    def __init__(self, threadID, name, counter, query_hist, column):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter        
        self.query_hist = query_hist
        self.column = column
        self.best = 9999

    def run(self):  
        y = self.column      
        slide_window_height = slide_window_width
        x = slide_window_width
        while( x < img_width):                        
            calcY = y-slide_window_height
            calcX = x-slide_window_width
            crop = img[calcY: y, calcX: x]  
            crop_hist = image_hist(crop)                                    
            diff = dist_bin(crop_hist, self.query_hist)                                    
            if self.best >= diff:
                self.best = diff
                self.crop = crop
            x = x + stride
            if (x > img_width):                
                x = img_width 

def getSquareCenterCrop(img):
    (h, w) = img.shape[:2]
    
    centerH = int(h/2)
    centerW = int(w/2)

    blockSize = h if (h > w) else w
    crop = img[ centerH - blockSize: centerH + blockSize, 
                centerW - blockSize: centerW + blockSize] 
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
    colors_hist[0] *= NORM/colors_hist[0].max()
    colors_hist[1] *= NORM/colors_hist[1].max()
    colors_hist[2] *= NORM/colors_hist[2].max()
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
height, width = query_color.shape[:2]
print height 
print width
slide_window_height = slide_window_width #/ (float(width) / height)

print "Pre-processing image"
query_mono = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
query_mono = getSquareCenterCrop(query_mono)
angledImages = []

for angle in range(359):
    angledImages.append(rotateImg(query_mono, angle))
    if ((angle % 30) == 0):
        print angle
        print getMeanSquareDiff(query_mono, angledImages[angle])
        plt.imshow(angledImages[angle], cmap = plt.get_cmap('gray'))
        plt.show()  

print "End pre-processing image"
'''
for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
    print target_image_path
    img = cv2.imread(target_image_path)
    img_height, img_width = img.shape[:2]    
    print "h: " + str(img_height) + " w:" + str(img_width)
    res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Display Window',res)         ## Show image in the window

    y = slide_window_height    
    local_diff = 9999 
    local_corp = None
    crop_hist = 0
    while( y < img_height):
        threads = []            
        for t_index in range(maxThreads): 
            l_thread =  WindowSlider(t_index, "Thread-" + str(t_index), t_index, query_hist, y)
            l_thread.start()            
            threads.append(l_thread)
            y = y + stride
            x = slide_window_width
            print y, x
            if (y > img_height):
                y = img_height

        
        for t in threads:
            t.join()  

        for t in threads:
            if (t.best <= local_diff):                                        
                local_diff = t.best
                local_corp = t.crop
                print "Thread: "  + str(t.counter) + " Value: "+ str(local_diff)


    plt.imshow(local_corp)
    plt.show()        

    image_hist(img)
    #image_hist_cv2(img)
    #print img
    cv2.waitKey(0)                           ## Wait for keystroke
    cv2.destroyAllWindows()   '''
