
# coding: utf-8

# ## Test Cropped Positive Test Images
# With HOG feature descriptor, and classify using svm (skilearn)

# In[33]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os, sys
from sklearn import svm, datasets, externals 

# Set to your datasets folder, and including the svm model
path = "E:\CodeProjects\INRIAPerson"
retval = os.getcwd()
print ('****Please set to your datasets folder, and including the svm model')
print ("Original Work Dir:%s" % retval)
os.chdir(path)
retval = os.getcwd()
print("Change to Current Dir:%s" % retval)


# In[19]:


# Load SVM model
model = externals.joblib.load('HOG_trained_model.m')
print(model)


# In[27]:


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


test_list = load_test_images(r'.\Test\test_img_list_crop.Lst')
print('TestList:',len(test_list),'Test Sample 0:',test_list[0].dtype,test_list[0].shape)
print('TestList:',len(test_list),'Test Sample 8:',test_list[8].dtype,test_list[8].shape)
# 发现有的图片读进来是空的，可能还是格式问题，用格式工厂转换一下PNG解决
# 读进来的图片大小是有不同的，处理后，统一都是图片格式134x70的
#print(test_list[8])

### Test: Read-in one Image and Show, 
# img = test_list[0]
#
# plt.imshow(img)
# plt.axis("off")#去除坐标轴
# plt.show()
# print(img.shape)


# ## Test

# In[32]:


# wsize: 处理图片大小，通常64*128; 输入图片尺寸>= wsize
def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    hog = cv2.HOGDescriptor()
    # hog.winSize = wsize
    for i in range(len(img_lst)):
        if img_lst[i].shape[1] >= wsize[1] and img_lst[i].shape[0] >= wsize[0]:
            roi = img_lst[i][(img_lst[i].shape[0] - wsize[0]) // 2: (img_lst[i].shape[0] - wsize[0]) // 2 + wsize[0],                   (img_lst[i].shape[1] - wsize[1]) // 2: (img_lst[i].shape[1] - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_lst.append(hog.compute(gray))
    return gradient_lst

test_gradient_list = []

computeHOGs(test_list, test_gradient_list)
print('****Complete HOG feature computation, and stored in test_gradient_list ')
print('test_gradient_list:',len(test_gradient_list),np.array(test_gradient_list).shape)


# In[39]:


### 测试一张test的图片
print('Predict:',model.predict(np.array(test_gradient_list[5]).reshape(-1,3780)))

# 测试整个test的图片集
prediction = []
CorrentCnt = np.int(0)
IncorrectCnt = np.int(0) 
IncorrectLabel=[]

print('****Load-in features, start prediction...')
for i in range(len(test_gradient_list)):
    prediction.append(model.predict(np.array(test_gradient_list[i]).reshape(-1,3780)))


for i in range(len(prediction)):
    if prediction[i]==1:
        CorrentCnt = CorrentCnt + 1
    else:
        IncorrectCnt = IncorrectCnt + 1
        IncorrectLabel.append(i)
    #print(prediction[i])
print('****Prediction Complete, Results:')
print('**Count of Correct Prediction:',CorrentCnt)
print('**Count of InCorrect Prediction:',IncorrectCnt)
print('**Count of Total Test:',len(prediction))
print('**Accuracy (Test Pos as Pos):',CorrentCnt/len(prediction))
print('Incorrect Label:',IncorrectLabel)


# ## Comments
# 正样本测试都OK，负样本读进来，用hog.detect检测会报错，应尝试直接用feature分类来检测
