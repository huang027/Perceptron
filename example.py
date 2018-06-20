import pandas as pd
import numpy as np
import cv2
import random
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import tqdm as tqdm


def Train(trainset,train_labels):
    # 获取参数
    trainset_size = len(train_labels)

    # 初始化 w,b
    w = np.zeros((feature_length,1))
    b = 0

    study_count = 0                         # 学习次数记录，只有当分类错误时才会增加
    nochange_count = 0                      # 统计连续分类正确数，当分类错误时归为0
    nochange_upper_limit = 100000           # 连续分类正确上界，当连续分类超过上界时，认为已训练好，退出训练

    while True:
        nochange_count += 1
        if nochange_count > nochange_upper_limit:
            break

        # 随机选的数据
        index = random.randint(0,trainset_size-1)
        img = trainset[index]
        label = train_labels[index]


        # 计算yi(w*xi+b)
        yi = int(label != object_num) * 2 - 1                       # 如果等于object_num, yi= 1, 否则yi=1
        result = yi * (np.dot(img,w) + b)

        # 如果yi(w*xi+b) <= 0 则更新 w 与 b 的值
        if result <= 0:
            img = np.reshape(trainset[index],(feature_length,1))    # 为了维数统一，需重新设定一下维度

            w += img*yi*study_step                                  # 按算法步骤3更新参数
            b += yi*study_step

            study_count += 1
            if study_count > study_total:
                break
            nochange_count = 0

    return w,b
def Predict(testset,w,b ):
    predict = []
    for img in testset:
        result = np.dot(img,w) + b
        result = result > 0

        predict.append(result)

    return np.array(predict)


if __name__=='__main__':
    print('start read data')
    time_1=time.time()
    raw_data=pd.read_csv('G:\\data\\lihang_book_algorithm-master\\data\\train_binary.csv',header=0)
    data=raw_data.values
    imgs=data[0::,1::]
    labels=data[::,0]
    train_features,test_features,train_labels,test_labels=train_test_split(imgs,labels,test_size=0.33,random_state=100)
    time_2=time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')
    print('start train')
    study_step = 0.0001  # 学习步长
    study_total = 10000  # 学习次数
    feature_length = len(raw_data.columns)-1  # hog特征维度
    object_num = 0
    w,b=Train(train_features,train_labels)
    time_3=time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')
    print('start predicting')
    test_predict=Predict(test_features,w,b)
    time_4=time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')
    score=accuracy_score(test_labels,test_predict)
    print("The accruacy socre is ", score)






