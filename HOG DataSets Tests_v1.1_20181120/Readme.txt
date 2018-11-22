FileName:  HOG_13-2_HOG+SVM_GroupTest
Version: v1.2
Comments: 能够读取数据库的正样本，负样本，用OpenCV的hog.detectMultiSacle()来进行多尺度滑窗检测，并输出统一格式的统计结果。

如下：
******** Summary of all the Statistics ********
Positive Test: 288 , #TP= 218 , #FN= 70
Negative Test: 453 , #TN= 412 , #FP= 41
Accuracy - (TP+TN)/(ALL):  85.020 %
Precision - (TP)/(TP+FP):  84.170 %
Recall/Sensitivity - (TP/TP+FN):  75.694 %
Specificity - (TN/TN+FP):  90.949 %
F1 Score - (2PR/P+R):  79.707 %