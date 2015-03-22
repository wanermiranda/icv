#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
import glob 

MIN_MATCHES = 4

def show_image(img):
    plt.imshow(img)
    if len(img.shape)<3:
        plt.gray()
    plt.axis("off")
    plt.show()


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    Keep only matches that have distance ratio to 
    second closest point less than 'ratio'.
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)    
    return p1, p2, kp_pairs

def match_images(detector, matcher, target, kp1, desc1):
	kp2, desc2 = detector.detectAndCompute(target, None)
	raw_matches = matcher.knnMatch(desc1,desc2, k=2)	
	#raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 5)	
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)		
	print p2.size
	return (p2.size >= MIN_MATCHES)




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

query_image = queryList[0]
query_image_path =  dataset_query + query_image
print query_image_path
query_image = cv2.imread(query_image_path,3)
show_image(query_image)


detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)
kp1, desc1 = detector.detectAndCompute(query_image, None)

for target_image_path in glob.glob(dataset_target_com_ruido + '*.png'): 
	print target_image_path
	target = cv2.imread(target_image_path,3)
	if match_images(detector, matcher, target, kp1, desc1): 
		print 'Matched'


