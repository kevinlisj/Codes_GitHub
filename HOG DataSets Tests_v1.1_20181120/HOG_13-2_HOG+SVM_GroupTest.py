
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os, sys
#from sklearn import svm, datasets, externals 

# Set to your datasets folder, and including the svm model
path = 'E:\DataSets\INRIAPerson'
retval = os.getcwd()
print ('****Please set to your datasets folder, and including the svm model')
print ("Original Work Dir:%s" % retval)
os.chdir(path)
retval = os.getcwd()
print("Change to Current Dir:%s" % retval)


# In[48]:


#测试整个test集中的图片

# 读取测试样本的函数
def load_test_images(dirname, amout = 9999):
    img_list = []
    file = open(dirname)
    img_name = file.readline()
    while img_name != '':  # 文件尾     
        img=cv2.imread(img_name.strip('\n'))  #注意出去换行符
        img_list.append(img)
        img_name = file.readline()
        
        amout -= 1
        if amout <= 0: # 控制读取图片的数量
            break
    return img_list

# 读取test集中的正样本
test_gradient_list = []
test_list = load_test_images(r'.\Test\test_img_list_largeonly.Lst')
print('*************Start to Load POSITIVE TEST IMAGES from List*************')
#print('TestList:',len(test_list),'Test Sample 0:',test_list[0].dtype,test_list[0].shape)
#print('TestList:',len(test_list),'Test Sample 8:',test_list[8].dtype,test_list[8].shape)
print('*************Complete Loading POSITIVE TEST IMAGES from List*************')
print('*****',len(test_list),' Images loaded into test_list[]')


# In[49]:


# 加载训练好的HOG模型
hog = cv2.HOGDescriptor()
#hog.load('.\myHogDector1.bin')
#hog.load('myHogDector1_Default_Baolin.bin')
hog.setSVMDetector(hog.getDefaultPeopleDetector()) 
hog.save('DefaultPeopleDetector.bin')
hog.load('DefaultPeopleDetector.bin')


# In[50]:


def overlapping_area(detection_1, detection_2):
#     Function to calculate overlapping area'si
#     `detection_1` and `detection_2` are 2 detections whose area of overlap needs to be found out.
#     Each detection is list in the format ->
#     [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
#     The function returns a value between 0 and 1,which represents the area of overlap.
#     0 is no overlap and 1 is complete overlap.
#     Area calculated from ->
#     http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles  
    
    # Calculate the x-y co-ordinates of the rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[2]
    x2_br = detection_2[0] + detection_2[2]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[3]
    y2_br = detection_2[1] + detection_2[3]
    
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[2] * detection_1[3]
    area_2 = detection_2[2] * detection_2[3]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=0.05):
### This function performs Non-Maxima Suppression.
#     detections consists of a list of detections.
#     Each detection is in the format ->
#     [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
#     If the area of overlap is greater than the `threshold`,the area with the lower confidence score is removed.
#     The output is a list of detections.
    
    if (len(detections) == 0):
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],reverse=True)
    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the 
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]

    return new_detections



# In[51]:


# 遍历所有图片，进行多尺度检测，并保存结果
i = 0
cnt_FN = 0
cnt_TP = 0

print("****** Start Testing Positive Test Images ******")
for img in test_list:
    rects, wei = hog.detectMultiScale(img, winStride=(4, 4),padding=(0, 0), scale=1.2)
    i = i+1
    # Perform Non Maxima Suppression
    rects_nms = nms(rects, threshold=0.02)  # 0.02的时候，就只剩一个框了  
    if np.array(rects_nms).shape[0] == 0 :
        cnt_FN = cnt_FN + 1
    else:
        cnt_TP = cnt_TP + 1
    #print('#',i,'Detected:', rects.shape, 'Rectangles','After NMS, Reduce to:',np.array(rects_nms).shape,'TP:',cnt_TP,'FN:',cnt_FN)

print("****** Complete Testing Positive Test Images ******")
Recall = (cnt_TP/(cnt_TP+cnt_FN)*100)
print('Results: #Positive Tests=',cnt_TP+cnt_FN, ', #TP=',cnt_TP,', #FN=', cnt_FN)
print('Recall/Sensitivity - (TP/TP+FN): %6.3f %%' %Recall)


# In[46]:


# 读取test集中的负样本
test_gradient_list = []
test_list = load_test_images(r'.\Test\neg.Lst')

print('*************Start to Load NEGATIVE TEST IMAGES from List*************')
#print('TestList:',len(test_list),'Test Sample 0:',test_list[0].dtype,test_list[0].shape)
#print('TestList:',len(test_list),'Test Sample 8:',test_list[8].dtype,test_list[8].shape)
print('*************Complete Loading NEGATIVE TEST IMAGES from List*************')
print('*****',len(test_list),'Images loaded into test_list[]************')


# In[47]:


# 遍历所有图片，进行多尺度检测，并保存结果
i = 0
cnt_FP = 0
cnt_TN = 0

print("****** Start Testing NEGATIVE TEST IMAGES ******")
for img in test_list:
    rects, wei = hog.detectMultiScale(img, winStride=(4, 4),padding=(0, 0), scale=1.2)
    i = i+1
    # Perform Non Maxima Suppression
    rects_nms = nms(rects, threshold=0.02)  # 0.02的时候，就只剩一个框了  
    #print(rects_nms)
    if rects_nms:
        cnt_FP = cnt_FP + 1
        print('#',i,'Detected:', rects.shape, 'Rectangles','After NMS, Reduce to:',np.array(rects_nms).shape,'TN:',cnt_TN,'FP:',cnt_FP)
    else:
        cnt_TN = cnt_TN + 1
        #print('#',i,'Detected: NO Rectangles','After NMS, Reduce to NULL.','TN:',cnt_TN,'FP:',cnt_FP)
        
print("****** Complete NEGATIVE TEST IMAGES ******")
print('Results: #Negative Tests=',cnt_TN+cnt_FP,', #TN=',cnt_TN,', #FP=', cnt_FP)
Specificity = cnt_TN/(cnt_TN+cnt_FP)*100
print('Specificity - (TN/TN+FP): %6.3f %%'%Specificity)


# In[52]:


# 统计最终结果：
print('******** Summary of all the Statistics ********')
print('Positive Test:',cnt_TP+cnt_FN,', #TP=',cnt_TP,', #FN=', cnt_FN)
print('Negative Test:',cnt_TN+cnt_FP,', #TN=',cnt_TN,', #FP=', cnt_FP)

#print('#TP=',cnt_TP,'#FN=', cnt_FN,'Sensitivity/Recall (TP/TP+FN): ',cnt_TP/(cnt_TP+cnt_FN)*100,'%')
#print('#TN=',cnt_TN,'#FP=', cnt_FP,'Sensitivity (TN/TN+FP):',cnt_TN/(cnt_TN+cnt_FP)*100,'%')

Accuracy = (cnt_TP+cnt_TN)/(cnt_TP+cnt_FN+cnt_TN+cnt_FP)*100
print('Accuracy - (TP+TN)/(ALL):  %6.3f %%' %Accuracy)
Precision = (cnt_TP)/(cnt_TP+cnt_FP)*100
print('Precision - (TP)/(TP+FP):  %6.3f %%' %Precision)
Recall = cnt_TP/(cnt_TP+cnt_FN)*100
print('Recall/Sensitivity - (TP/TP+FN):  %6.3f %%'%Recall)
Specificity = cnt_TN/(cnt_TN+cnt_FP)*100
print('Specificity - (TN/TN+FP):  %6.3f %%' %Specificity)
F1_Score = 2*Precision*Recall/(Precision+Recall)
print('F1 Score - (2PR/P+R):  %6.3f %%' %F1_Score)

