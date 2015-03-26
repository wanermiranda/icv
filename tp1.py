#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import sys
import glob 
from timer import Timer

MIN_MATCHES = 4	
slide_window_width = 60 # the height is to be calculed based on the proportion of the query image
stride = 5 # step to slide 
NORM = 1

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
    '''for i in range(3):
        plt.plot(colors_hist[i])
        plt.xlim([0,256])
    plt.show()'''

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

queryIndex = 1
query = cv2.imread(dataset_query + queryList[queryIndex-1])
query_hist = image_hist(query)
print query_hist
height, width = query.shape[:2]
print height 
print width
slide_window_height = slide_window_width #/ (float(width) / height)


for target_image_path in glob.glob(dataset_target_sem_ruido + '00'+str(queryIndex)+'*.png'): 
    print target_image_path
    img = cv2.imread(target_image_path)
    img_height, img_width = img.shape[:2]    
    print "h: " + str(img_height) + " w:" + str(img_width)
    res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Display Window',res)         ## Show image in the window

    y = slide_window_height
    x = slide_window_width
    last_diff = 30 
    last_crop = None
    diff = 0
    crop_hist = 0
    while( y < img_height):
        while( x < img_width):                        
            calcY = y-slide_window_height
            calcX = x-slide_window_width
            '''if ((calcY % slide_window_height) == 0):
                print calcY, y, calcX, x'''
            crop = img[calcY: y, calcX: x]
            #print "hist"
            #with Timer() as t:
            crop_hist = image_hist(crop)
            #print "=> elasped crop_hist: %s s" % t.secs                
            #print "diff"            
            #with Timer() as t:
            diff = dist_bin(crop_hist, query_hist)            
            #print "=> elasped dist_bin: %s s" % t.secs
            if (diff <= last_diff):                
                print "Crop " + str(diff)                
                print calcY, y, calcX, x
                last_diff = diff
                last_crop = crop
                print "Uai!"	          
            x = x + stride
            if (x > img_width):                
                x = img_width 
        y = y + stride
        x = slide_window_width
        if (y > img_height):
            y = img_height


    plt.imshow(last_crop)
    plt.show()        

    '''image_hist(img)
    #image_hist_cv2(img)
    #print img
    cv2.waitKey(0)                           ## Wait for keystroke
    cv2.destroyAllWindows()   '''
