#!/bin/python
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os

def show_image(img):
    plt.imshow(img)
    if len(img.shape)<3:
        plt.gray()
    plt.axis("off")
    plt.show()


def determinant(H):
    return (H[0,1] *  H[1,1]) - (H[1,0] *  H[0,1])

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



img1 = cv2.imread('box.png',0)
img2 = cv2.imread('box_in_scene.png',0)


print("Read img1 of size {} and img2 of size {}.".format(img1.shape, img2.shape))
#print("Image 1:")
#show_image(img1)
#print("Image 2:")
#show_image(img2)

detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)
detector.detectAndCompute(img1, None)

kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)


p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)


H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3)

print H
print(determinant(H))



h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
vis[:h1, :w1] = img1
vis[:h2, w1:w1+w2] = img2
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)



show_image(vis)


p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)


# plot the matches
color = (0, 255, 0)
for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
    if inlier:
        cv2.circle(vis, (x1, y1), 2, color, -1)
        cv2.circle(vis, (x2, y2), 2, color, -1)
        cv2.line(vis, (x1, y1), (x2, y2), color)

show_image(vis)

