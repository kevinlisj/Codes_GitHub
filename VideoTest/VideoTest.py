# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:11:40 2018

@author: lishuaijun
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os, sys
#% matplotlib inline

#%%
# Set to your datasets folder, and including the svm model
path = 'E:\AnacondaProject\HOG'
retval = os.getcwd()
print ('****Please set to your datasets folder, and including the test video folder: .\\VideoTest\\')
print ("Original Work Dir:%s" % retval)
os.chdir(path)
retval = os.getcwd()
print("Change to Current Dir:%s" % retval)
#%% Read in HOG Feature

hog = cv2.HOGDescriptor()
hog.load('.\myHogDector3.bin')

#%%
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('.\\VideoTest\\test_0003.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = 20
size = (int(cap.get(3)),int(cap.get(4)))
#print('Video Size:', size)
# Create a Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vd_out = cv2.VideoWriter()
vd_out.open('.\VideoTest\output.mp4', fourcc, fps, size, True)

ret, frame = cap.read()
#print(frame.shape)
while(ret):
    
    #cv2.imshow('Frame',frame)
    rects, wei = hog.detectMultiScale(frame, winStride=(4, 4),padding=(0, 0), scale=1.1)
    #print('HOG.detectMultiScale() Detected:', rects.shape, 'Rectangles')
    print(rects)
    detect_frame = frame.copy()
    
    for (x, y, w, h) in rects:
        cv2.rectangle(detect_frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.imshow('Detected Frame',detect_frame)
    
    vd_out.write(detect_frame)
    #cv2.imshow('Frame',frame)
    # Press Q on keyboard to  exit
    ret, frame = cap.read()
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

#%%
# When everything done, release the video capture objectq
cap.release()
vd_out.release()
print('Finished')
# Closes all the frames
cv2.destroyAllWindows()