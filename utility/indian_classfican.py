from libsvm.svmutil import *
from scipy.io import loadmat
import scipy
import numpy as np

import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd

import matplotlib.pyplot as plt  
import numpy as np
from scipy.io import loadmat
import spectral

def c1():
    # Construct problem in python format
    # Dense data
    mat_list = ["noisy","scat","LLRGTV","LRTDTV","grn_net","macnet","t3sc","sst","sert_base"]
    for name in mat_list:
        y_path = "/data1/jiahua/ly/test_data/india_all_method_mat/Indian_pines_144_gt.mat"
        x_path = "/data1/jiahua/ly/test_data/india_all_method_mat/"+name+".mat"
        output_image = loadmat(y_path).get("indian_pines_gt")
        output_image = output_image[0:128,0:128]
        if name == "noisy":
        # gt_name = "gt.mat"
            key = "data"
        elif name == "LLRGTV" or name == "LRTDTV":
            key = "output_image"
        else :
            key = "data"
        input_image = loadmat(x_path).get(key)

    # 除掉 0 这个非分类的类，把所有需要分类的元素提取出来
        need_label = np.zeros([output_image.shape[0],output_image.shape[1]])
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] != 0:
                #if output_image[i][j] in [1,2,3,4,5,6,7,8,9]:
                    need_label[i][j] = output_image[i][j]


        new_datawithlabel_list = []
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if need_label[i][j] != 0:
                    c2l = list(input_image[i][j])
                    c2l.append(need_label[i][j])
                    new_datawithlabel_list.append(c2l)

        new_datawithlabel_array = np.array(new_datawithlabel_list) 

        from sklearn import preprocessing
        # data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
        data_D = preprocessing.MinMaxScaler().fit_transform(new_datawithlabel_array[:,:-1])
        data_L = new_datawithlabel_array[:,-1]

        # 将结果存档后续处理
        import pandas as pd
        new = np.column_stack((data_D,data_L))
        new_ = pd.DataFrame(new)
        new_.to_csv("/home/jiahua/liuy/hsi_pipeline/result/"+name+".csv",header=False,index=False)
    # new_.to_csv('C:/Users/xiao/Desktop/indianpines.csv',header=False,index=False)
    
def c2():

    # 导入数据集切割训练与测试数据
    mat_list = ["noisy","scat","LLRGTV","LRTDTV","grn_net","macnet","t3sc","sst","sert_base"]

    for name in mat_list:
    
        data = pd.read_csv("/home/jiahua/liuy/hsi_pipeline/result/"+name+".csv",header=None)
        # data = pd.read_csv('C:/Users/xiao/Desktop/indianpines.csv',header=None)
        data = data.values

        data_D = data[:,:-1]
        data_L = data[:,-1]
        data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.5)

        # 模型训练与拟合

        clf = SVC(kernel='rbf',gamma=0.01,C=12)
        clf.fit(data_train,label_train)
        pred = clf.predict(data_test)
        accuracy = metrics.accuracy_score(label_test, pred)*100
        print(name," : ",accuracy) 


        # 存储结果学习模型，方便之后的调用
        joblib.dump(clf, "/home/jiahua/liuy/hsi_pipeline/result/"+name+".m")
        # joblib.dump(clf, "C:/Users/xiao/Desktop/Indianpines_MODEL.m")
        
        
# noisy  :  74.44250109313512
# scat  :  90.99256668124181
# LLRGTV  :  62.4836029733275
# LRTDTV  :  67.20594665500656
# grn_net  :  76.12592916484478
# macnet  :  76.4975951027547
# t3sc  :  84.4993441189331
# sst  :  82.68473983384347
# sert_base  :  88.06296458242238
        

def c3():

    # KSC
    # y_path = "E:/硕士文件/硕士论文/毕业论文一/高光谱数据集/indianpines_分类/Indian_pines_144_gt.mat"
    # x_path = "C:/Users/xiao/Desktop/DPHSIR/indian_pines/DPHSIR.mat"
    # output_image = loadmat(y_path).get("indian_pines_gt")
    # input_image = loadmat(x_path).get("indian_pines")
    # Dense data
    mat_list = ["noisy","scat","LLRGTV","LRTDTV","grn_net","macnet","t3sc","sst","sert_base"]

    for name in mat_list:
        y_path = "/data1/jiahua/ly/test_data/india_all_method_mat/Indian_pines_144_gt.mat"
        x_path = "/data1/jiahua/ly/test_data/india_all_method_mat/"+name+".mat"
        output_image = loadmat(y_path).get("indian_pines_gt")
        output_image = output_image[0:128,0:128]
        if name == "noisy":
        # gt_name = "gt.mat"
            key = "data"
        elif name == "LLRGTV" or name == "LRTDTV":
            key = "output_image"
        else :
            key = "output"
        input_image = loadmat(x_path).get(key)

        testdata = np.genfromtxt("/home/jiahua/liuy/hsi_pipeline/result/"+name+".csv",delimiter=',')
        # testdata = np.genfromtxt('C:/Users/xiao/Desktop/indianpines.csv',delimiter=',')
        data_test = testdata[:,:-1] #一定为全部波段  overall
        label_test = testdata[:,-1]

        # /Users/mrlevo/Desktop/CBD_HC_MCLU_MODEL.m
        clf = joblib.load( "/home/jiahua/liuy/hsi_pipeline/result/"+name+".m")
        # clf = joblib.load("C:/Users/xiao/Desktop/Indianpines_MODEL.m")
        predict_label = clf.predict(data_test)
        accuracy = metrics.accuracy_score(label_test, predict_label)*100 #OA
        kappa = metrics.cohen_kappa_score(label_test,predict_label) #kappa

        print(name,":",accuracy,kappa)  # 97.1022836308
        print()

        # 将预测的结果匹配到图像中
        new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
        k = 0
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] != 0 :
                    new_show[i][j] = predict_label[k]
                    k +=1 

        # print( new_show.shape)

        # 展示地物
    #     ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9))
    #     ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(9,9))
        save_path = "/home/jiahua/liuy/hsi_pipeline/result/pic/"+name+".png"
        spy_colors = np.array([[0, 0, 0],
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [255, 0, 255],
                            [0, 255, 255],
                            [200, 100, 0],
                            [0, 200, 100],
                            [100, 0, 200],
                            [200, 0, 100],
                            [100, 200, 0],
                            [0, 100, 200],
                            [150, 75, 75],
                            [75, 150, 75],
                            [75, 75, 150],
                            [255, 100, 100],
                            [100, 255, 100],
                            [100, 100, 255],
                            [255, 150, 75],
                            [75, 255, 150],
                            [150, 75, 255],
                            [50, 50, 50],
                            [100, 100, 100],
                            [150, 150, 150],
                            [200, 200, 200],
                            [250, 250, 250],
                            [100, 0, 0],
                            [200, 0, 0],
                            [0, 100, 0],
                            [0, 200, 0],
                            [0, 0, 100],
                            [0, 0, 200],
                            [100, 100, 0],
                            [200, 200, 0],
                            [100, 0, 100],
                            [200, 0, 200],
                            [0, 100, 100],
                            [0, 200, 200]], np.int16)
        spectral.save_rgb(save_path,new_show, colors=spy_colors )


if __name__ == '__main__':
    c1()
    c2()
    c3()