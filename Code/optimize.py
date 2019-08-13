from scipy import  optimize as opt
import numpy as np
import pandas as pd
import os
import time
import pickle
from WindPy import *

def optimize():
    #得到当前日期
    date=time.strftime("%Y%m%d")
    print(date)

    #得到当前日期下的指数成分股以及权重
    w.start()
    w.isconnected()
    oriData=w.wset("IndexConstituent","date=20170602;windcode=000300.SH;field=wind_code,i_weight").Data
    indexWeight=pd.DataFrame({'weight':oriData[1]},index=oriData[0])

    # 得到因子暴露矩阵
    facGroup=['value','return']
    facValue=pd.DataFrame(columns=facGroup)
    for facname in facGroup:
        with open('//10.50.48.26/multiFactor/'+facname+'factor_value'+date+'.bin','rb') as f:
            facValue[facname]=pd.merge()
    #得到行业矩阵
    dir=os.path.abspath('..')
    with open(dir+'/data_in/instrument','rb') as f:
        indusData=pickle.load(f)
    indexIndus=indusData.merge(indexWeight)
    print(indexIndus)

    #计算行业权重
    indusWeight=np.dot(indexWeight,indexIndus)
    #计算风格权重
    facWeight=np.dot(indexWeight,facValue)

    portfolio=[]'''组合成分股，已知'''
    #求解
    #得到组合行业矩阵
    portIndus=pd.merge(portfolio,indusData)
    #得到组合风格矩阵
    portFac=pd.merge(portfolio,facValue)