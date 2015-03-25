#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import sys
import glob 

MIN_MATCHES = 4	
slide_window = 300 # H x H square window 
stride = 20 # step to slide 
NORM = 1

def image_hist(img):
    colors_hist = np.zeros((3,256)) 
    for row in img:
        for column in row:
            for i in range(3):
                color = column[i]
                colors_hist[i][color] = colors_hist[i][color] +1
    colors_hist[0] *= NORM/colors_hist[0].max()
    #np.linalg.norm(colors_hist[0],ord=1)
    #np.linalg.norm(colors_hist[1],ord=1)
    #np.linalg.norm(colors_hist[2],ord=1)

    colors_hist[1] *= NORM/colors_hist[1].max()
    colors_hist[2] *= NORM/colors_hist[2].max()
    return colors_hist
    '''for i in range(3):
        plt.plot(colors_hist[i])
        plt.xlim([0,256])
    plt.show()'''

def dist_bin(A, B): 
    diff = np.zeros((3,256))
    diff[0] = (A[0] - B[0])**2

    diff[1] = (A[1] - B[1])**2

    diff[2] = (A[2] - B[2])**2
    
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

query = cv2.imread(dataset_query + queryList[7])
query_hist = image_hist(query)
print query_hist
height, width = query.shape[:2]
print height 
print width


for target_image_path in glob.glob(dataset_target_sem_ruido + '008*.png'): 
    print target_image_path
    img = cv2.imread(target_image_path)
    img_height, img_width = img.shape[:2]    
    print "h: " + str(img_height) + " w:" + str(img_width)
    res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Display Window',res)         ## Show image in the window

    y = slide_window
    x = slide_window
    last_diff = 30 
    while( y < img_height):
        while( x < img_width):            
            print y-slide_window, y, x-slide_window, x
            crop = img[y-slide_window: y, x-slide_window: x]
            crop_hist = image_hist(crop)
            diff = dist_bin(crop_hist, query_hist)
            print "Crop " + str(diff)
            if (diff <= last_diff):
		last_diff = diff
		print "Uai!"
	        plt.imshow(crop)
		plt.show()
	    x = x + stride
            if ( x > img_width):                
                x = img_width   
                        
        y = y + stride
	x = slide_window
        if ( y > img_height):
            y = img_height


    
    cv2.waitKey(0)                           ## Wait for keystroke
    cv2.destroyAllWindows()   

    '''image_hist(img)
    #image_hist_cv2(img)
    #print img
    cv2.waitKey(0)                           ## Wait for keystroke
    cv2.destroyAllWindows()   '''
