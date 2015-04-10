#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import math 
import cv2
import sys
import glob 
from timer import Timer
from rotate import *
import threading
import time

base_scale = 0.5 # used to reduce the amount of data threated
slide_window_width = 0 # to be initilized further
stride = 5 # step to slide 
NORM = 1
last_crop = 0
maxThreads = 4
queryIndex = 3
rotation_factor = 15
angle_range = 360 / rotation_factor


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


class WindowSlider (threading.Thread):
    def __init__(self, threadID, query, target, column, best ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.query = query
        self.target = target
        self.column = column
        self.best = best
        self.crop = None
    

    def run(self):  
        y = self.column              
        x = slide_window_width
        target_height, target_width  = self.target.shape[:2]

        while( x < target_width):                        
            calcY = y-slide_window_height
            calcX = x-slide_window_width
            crop = self.target[calcY: y, calcX: x]  
            MSDiff = getMeanSquareDiff(crop, self.query, self.best)
            
            diff = MSDiff 

            if ((self.best >= diff) or (self.best == -1)):
                self.crop = crop
                self.best = diff                
                #print calcY
                #print calcX
                #print diff 


            x = x + stride
            
            if (x > target_width):                
                x = target_width 

def showImage (img):
    ##plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img, cmap = plt.cm.Greys_r)
    plt.show()

def find_nearest(array,value):
    idx = np.abs(np.subtract.outer(array, value)).argmin(0)
    return (array[idx], idx)


def getMeanSquareDiff(img, target, best):
    (m, n) = img.shape[:2]    
    summ = 0.0
    for j in range(m):
        for i in range(n):
            if target[j,i] != 0:
                summ += np.subtract(img[j][i], target[j][i], dtype=np.float64) ** 2
                if (summ > best):
                    break

    return summ/(m*n)

def getMeanSquareDiff_BGR(query, target, best):
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
                if (summ > best):
                    break

    return summ/(3*(m*n))



def rotateImg (img, angle):
    (h,w) = img.shape[:2]
    image_rotated = rotate_image(img, angle)
    image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(                
                h,
                w,
                math.radians(angle)
            )
        )
    return image_rotated_cropped
'''def rotateImg (img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
     
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

#def quad_diff(img, target):'''


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


def dist_bin(A, B, best): 
    diff = np.zeros((3,256))
    summ = 0
    for row in range(256):        
            diff[0][row] = (A[0][row] - B[0][row])**2
            diff[1][row] = (A[1][row] - B[1][row])**2
            diff[2][row] = (A[2][row] - B[2][row])**2
            summ = summ  + diff[0][row] + diff[1][row] + diff[2][row]
            if ((best > -1) and (summ > best)): 
                break
    
    return summ




rotate_diff = -1
rotate_crop = None
best_query = None
best_angle = 0
starting = time.time()
for angle in range(angle_range):

    query_color = cv2.imread(dataset_query + queryList[queryIndex-1])
    query_color = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
    query_color = rotateImg(query_color, angle*rotation_factor)

    print dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'
    for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
        print target_image_path
        target_color = cv2.imread(target_image_path)                    
        target_color = cv2.resize(target_color,None,fx=base_scale, fy=base_scale, interpolation = cv2.INTER_CUBIC)
        target_height, target_width = target_color.shape[:2]
        target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)    

        last_local_diff = rotate_diff 
        last_local_crop = rotate_crop

        factor = 0.0
        for factor_index in range(1,6):
            # Readjusting the query size and the slide window to match X% of the targeted image
            factor = 0.1 * factor_index
            query_color = cv2.imread(dataset_query + queryList[queryIndex-1])
            query_color = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
            query_color = rotateImg(query_color, angle*rotation_factor)            

            query_height, query_width = query_color.shape[:2]
            query_base_scale = (factor * target_height)/query_height

            query_color = cv2.resize(query_color,None,fx=query_base_scale, fy=query_base_scale, interpolation = cv2.INTER_CUBIC)
            query_height, query_width = query_color.shape[:2]

            slide_window_width   =  query_width
            slide_window_height  =  query_height


            
            print "Target Dimensions"
            print target_width
            print target_height
            print "Angle " + str(angle*rotation_factor)
            #showImage(target_color)
            
            print "Slide Window Dimensions"
            print slide_window_width
            print slide_window_height
            #showImage(query_color)

            y = slide_window_height    
            local_diff = last_local_diff 
            local_crop = last_local_crop

            while( y < target_height):
                threads = []            
                for t_index in range(maxThreads): 
                    l_thread =  WindowSlider(t_index, query_color, target_color, y, local_diff)
                    l_thread.start()            
                    threads.append(l_thread)
                    y = y + stride
                    #print "Line :" + str(y)
                    if (y > target_height):
                        y = target_height
                
                for t in threads:
                    t.join()  

                for t in threads:
                    if ( ((t.best <= local_diff) or (local_diff == -1)) and (t.best != -1)):                        
                        if (t.crop is not None):
                            print "Thread: "  + str(t.threadID) + " Value: "+ str(t.best)
                            local_diff = t.best
                            local_crop = t.crop
                            h, w = local_crop.shape[:2]


            if ((last_local_diff > local_diff) or (last_local_diff == -1)):
                last_local_crop = local_crop
                last_local_diff = local_diff
                best_query = query_color
                print "INTERMEDIATE TIME: " + str(time.time() - starting) + " SECS - FACTOR: " + str(factor) + " - ANGLE: " + str(angle*rotation_factor)
                print "DIFF: " + str(local_diff) 
                #showImage(local_crop)
                #showImage(best_query)
            else: 
                break
         

        if ((rotate_diff > last_local_diff) or (rotate_diff == -1)):
            rotate_crop = last_local_crop 
            rotate_diff = last_local_diff 
            best_angle = angle*rotation_factor
            print "INTERMEDIATE TIME: " + str(time.time() - starting) + " SECS - ANGLE: " + str(angle*rotation_factor)
            print "DIFF: " + str(rotate_diff) 
            #showImage(rotate_crop)
            #showImage(best_query)

print "TIME: " + str(time.time() - starting) + " SECS "
print "DIFF: " + str(rotate_diff) 
print "ANGLE: " + str(best_angle) 
showImage(rotate_crop)
print "Query "
showImage(best_query)
