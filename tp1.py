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
queryIndex = 7

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
            MSDiff = dist_bin(block_hist, self.query_hist, self.best)
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




def main (args):
    starting = time.time()
    query_color = cv2.imread(dataset_query + queryList[queryIndex-1])
    query_color = cv2.resize(query_color,None,fx=base_scale, fy=base_scale, interpolation = cv2.INTER_CUBIC)
    height, width = query_color.shape[:2]
    print "Query Image -> h: " + str(height) + " w:" + str(width)
    #plt.imshow(cv2.cvtColor(query_color, cv2.COLOR_BGR2RGB))
    #plt.show()   

    height, width = query_color.shape[:2]
    print "Query Center Image -> h: " + str(height) + " w:" + str(width)

    slide_window_width   =  width * 0.1
    slide_window_height  = height * 0.1 

    print "Slide Window: "
    print 'slide_window_width   ' +str( width)
    print 'slide_window_height  ' +str(height)

    query_hist = image_hist(query_color)


    for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
        print target_image_path
        target_color = cv2.imread(target_image_path)    
        target_color = cv2.resize(target_color,None,fx=base_scale, fy=base_scale, interpolation = cv2.INTER_CUBIC)
        target_height, target_width = target_color.shape[:2]

        print target_width
        print target_height

        y = slide_window_height    
        local_diff = -1 
        local_crop = None
        '''
        # Test Block - Sandbox
        

        calcY = 232-(slide_window_height/2)
        calcX = 256-(slide_window_width/2)

        y =  232 + (slide_window_height/2)
        x =  256 + (slide_window_width/2)

        local_crop = target_color[calcY: y, calcX: x]  

        MSDiff = getMeanSquareDiff_BGR(local_crop, query_color)
        #(value, angle) = find_nearest(angledImages, MSDiff)
        #diff = (MSDiff - value)**2

        print MSDiff
        plt.imshow(local_crop, cmap = plt.get_cmap('gray'))
        plt.show()        

        # Test Block - Sandbox
        '''
        while( y < target_height):
            threads = []            
            for t_index in range(maxThreads): 
                l_thread =  WindowSlider(t_index, query_color, target_color, y, local_diff, query_hist)
                l_thread.start()            
                threads.append(l_thread)
                y = y + stride
                print "Line :" + str(y)
                if (y > target_height):
                    y = target_height
            
            for t in threads:
                t.join()  

            for t in threads:
                if ( ((t.best <= local_diff) or (local_diff == -1)) and (t.best != -1)):
                    print "Thread: "  + str(t.threadID) + " Value: "+ str(t.best)
                    if (t.crop != None):
                        local_diff = t.best
                        local_crop = t.crop
                        h, w = local_crop.shape[:2]
                        print h 
                        print w                    
                        #if (62 >= local_diff):
                            #plt.imshow(local_crop, cmap = plt.get_cmap('brg_r'))
                            #plt.show()

    print "TIME: " + str(time.time() - starting) + " SECS"
    plt.imshow(cv2.cvtColor(local_crop, cv2.COLOR_BGR2RGB))
    plt.show()        



    print "Post-processing image"
    '''
    diff_angle = -1
    for idx in range(120):
        angle= idx
        img  = rotateImg(query_color, angle)
        diff = getMeanSquareDiff_BGR(local_crop, img)    
        if ( (diff <= diff_angle) or (diff_angle == -1)):
            diff_angle = diff
            print "Angle " + str(angle)
            print diff
            sys.stdout.flush()
            #plt.imshow(img, cmap = plt.get_cmap('gray'))
            #plt.show()        

    '''
    print "End Post-processing image"
