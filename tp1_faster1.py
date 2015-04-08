#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import math 
import cv2
import sys
import glob 
from timer import Timer
import threading
import time

base_scale = 0.4 # used to reduce the amount of data threated
slide_window_width = 0 # to be initilized further
stride = 5 # step to slide 
NORM = 1
last_crop = 0
maxThreads = 4

queryIndex = 1


def quad_it(img, base_hist, best, blocks):
    (height, width) = img.shape[:2]
    result_diff = best
    result_crop = None    
    dim_y = height / blocks
    dim_x = width / blocks
    print "(h,w)" + str(dim_y) + ":" + str(dim_x)    
    for row in range(blocks): 
        for col in range(blocks): 
            y1 = row * dim_y 
            y2 = (row + 1) * dim_y
            x1 = col * dim_x 
            x2 = (col + 1) * dim_x

            #print "diff crop " + str(y1) + ":" + str(y2) + "," + str(x1) + ":" + str(x2)

            crop = img[ (y1): (y2), (x1):(x2)]                                 
            crop_diff = dist_bin(image_hist(crop), base_hist)
            print crop_diff  
            if ((crop_diff <= result_diff) or (result_diff == -1)):
                result_diff = crop_diff
                result_crop = crop
                #plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                #plt.show()
    
    return result_diff, result_crop

class WindowSlider (threading.Thread):
    def __init__(self, threadID, query, target, column, best, query_hist):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.query = query
        self.target = target
        self.column = column
        self.best = best
        self.crop = None
        self.query_hist = query_hist

    def run(self):  
        y = self.column              
        x = slide_window_width
        target_height, target_width  = self.target.shape[:2]

        while( x < target_width):                        
            calcY = y-slide_window_height
            calcX = x-slide_window_width
            crop = self.target[calcY: y, calcX: x]  
            block_hist = image_hist(crop)
            MSDiff = dist_bin(block_hist, query_hist)
            #(value, angle) = find_nearest(angledImages, MSDiff)
            diff = MSDiff #(MSDiff - value)**2

            #if ( (calcY > 208) and (calcX > 235) ):
            #   print diff
            #   print calcY
            #   print calcX
            #   plt.imshow(crop, cmap = plt.get_cmap('brg_r'))
            #   plt.show()


            if ((self.best >= diff) or (self.best == -1)):
                self.crop = crop
                self.best = diff                
                print calcY
                print calcX
                print diff 


                #self.angle = angle

            x = x + stride
            
            if (x > target_width):                
                x = target_width 

def find_nearest(array,value):
    idx = np.abs(np.subtract.outer(array, value)).argmin(0)
    return (array[idx], idx)


def getMeanSquareDiff(img, target):
    (m, n) = img.shape[:2]    
    summ = 0.0
    for j in range(m):
        for i in range(n):
            if target[j,i] != 0:
                summ += np.subtract(img[j][i], target[j][i], dtype=np.float64) ** 2

    return summ/(m*n)

def getMeanSquareDiff_BGR(query, target):
    (m, n) = query.shape[:2]    
    summ = 0.0
    for j in range(m):
        for i in range(n):
            if (    ( (target[j][i][0] + target[j][i][1] + target[j][i][2]) == 0)
                 or ( (target[j][i][0] + target[j][i][1] + target[j][i][2]) == (3*255))
                 or ( (query[j][i][0] + query[j][i][1] + query[j][i][2]) == 0) 
                 or ( (query[j][i][0] + query[j][i][1] + query[j][i][2]) == (3*255.0))): 
                summ += 0
            else:
                    summ += math.sqrt(np.subtract(query[j][i][0], target[j][i][0], dtype=np.float64) ** 2)
                    summ += math.sqrt(np.subtract(query[j][i][1], target[j][i][1], dtype=np.float64) ** 2)
                    summ += math.sqrt(np.subtract(query[j][i][2], target[j][i][2], dtype=np.float64) ** 2)

    return summ/(3*(m*n))




def rotateImg (img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
     
    M = cv2.getRotationMatrix2D(center, angle, 1)
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


starting = time.time()
dataset_query = '/home/gorigan/datasets/icv/tp1/imagens/query/'
dataset_target_sem_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/sem_ruido/'
dataset_target_com_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/com_ruido/'

queryList = ["001_apple_obj.png"
			,"002_dumbphone_obj.png"
			,"003_japan_obj.png"
			,"004_yen_obj.png"
			,"005_bottle_obj.png"
			,"006_shoe_obj.png"
			,"007_kay_obj.png"
			,"008_starbucks_obj.png"
			,"009_coca_obj.png"]

query_color = cv2.imread(dataset_query + queryList[queryIndex-1])
print dataset_query + queryList[queryIndex-1]
height, width = query_color.shape[:2]
print "Query Image -> h: " + str(height) + " w:" + str(width)
query_hist = image_hist(query_color)
print "dist_bin check: " + str(not dist_bin(query_hist, query_hist))

for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
    print target_image_path
    target_color = cv2.imread(target_image_path)        
    target_height, target_width = target_color.shape[:2]

    print "Target Image -> h: " + str(target_height) + " w:" + str(target_width)

    best_diff = -1
    best_crop = target_color    
    
    for block_sz in range(1,10):
        print "Block sz:" + str(block_sz)
        best_diff, result_crop = quad_it(target_color, query_hist, best_diff, block_sz * 3)
        if (result_crop != None):
            best_crop = result_crop
            if (best_diff < 10): break


print "TIME: " + str(time.time() - starting) + " SECS"
print best_diff
plt.imshow(cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB))
plt.show()        


