
# coding: utf-8

# # PART 2: Metrics

# Постановка задачи:Необходимо реализовать функции accuracy_score, precision_score, recall_score, lift_score, f1_score (названия функций должны совпадать). Каждая функция должна иметь 3 обязательных параметра def precision_score(y_true, y_predict, percent=None). Добавлять свои параметры нельзя.
# 
# Нельзя использовать готовые реализации этих метрик Если percent=None то метрика должна быть рассчитана по порогу вероятности >=0.5 Если параметр percent принимает значения от 1 до 100 то метрика должна быть рассчитана на соответствующем ТОПе 1 - верхний 1% выборки 100 - вся выборка y_predict - имеет размерность (N_rows; N_classes)

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import csv
from collections import defaultdict
import time
get_ipython().run_line_magic('matplotlib', 'inline')
file = np.loadtxt('HW2_labels.txt', delimiter = ',')
y_predict, y_true = file[:,:2],file[:,-1]
print(y_predict.shape, y_true.shape)


# In[10]:


def threshold(y_predict,percent):
    l = int(len(y_predict[:,1])*percent/100.0)
    result = np.sort(y_predict[:,1])[-l]
    return result

def accuracy_score(y_true, y_predict, percent=None):
    if percent == None:
        y_predict_clf = np.array([1 if y_predict[i,1]>=0.5 else 0 for i in range(y_predict.shape[0])])
    else:
        threshold_ = threshold(y_predict, percent)
        y_predict_clf = np.array([1 if y_predict[i,1]>=threshold_ else 0 for i in range(y_predict.shape[0])])
    result = 1 - (np.sum(np.abs(y_true - y_predict_clf))/len(y_true))
    return result


def precision_score(y_true, y_predict, percent=None):
    if percent == None:
        y_predict_clf = np.array([1 if y_predict[i,1]>=0.5 else 0 for i in range(y_predict.shape[0])])
    else:
        threshold_ = threshold(y_predict, percent)
        y_predict_clf = np.array([1 if y_predict[i,1]>=threshold_ else 0 for i in range(y_predict.shape[0])])
    True_Positive = np.sum(np.array([1 if y_true[i]==1 and y_predict_clf[i]==1 else 0 for i in range(len(y_true))]))
    False_Positive = np.sum(np.array([1 if y_true[i]==0 and y_predict_clf[i]==1 else 0 for i in range(len(y_true))]))
    result = True_Positive/(True_Positive + False_Positive)
    return result

def recall_score(y_true, y_predict, percent=None):
    if percent == None:
        y_predict_clf = np.array([1 if y_predict[i,1]>=0.5 else 0 for i in range(y_predict.shape[0])])
    else:
        threshold_ = threshold(y_predict, percent)
        y_predict_clf = np.array([1 if y_predict[i,1]>=threshold_ else 0 for i in range(y_predict.shape[0])])
    True_Positive = np.sum(np.array([1 if y_true[i]==1 and y_predict_clf[i]==1 else 0 for i in range(len(y_true))]))
    False_Negative = np.sum(np.array([1 if y_true[i]==1 and y_predict_clf[i]==0 else 0 for i in range(len(y_true))]))
    result = True_Positive/(True_Positive + False_Negative)
    return result

def f1_score(y_true, y_predict, percent=None):
    result = 2*precision_score(y_true, y_predict, percent)*recall_score(y_true, y_predict, percent)/(precision_score(y_true, y_predict, percent) + recall_score(y_true, y_predict, percent))
    return result

def lift_score(y_true, y_predict, percent=None):
    lift = np.sum(y_true)/len(y_true)
    result = precision_score(y_true, y_predict, percent)/lift
    return result

