import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import sys
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from conf_factor import fct_class

e = 1e-10  # 非常接近0的值

# 加载因子IC数据
def Load_IC(dir,fct_class,start,end):
    ic_df=pd.DataFrame()
    for fct in fct_class:
        f = open(dir+'/data_out/factor_ic/'+fct+'/'+fct+'_rankic', 'rb')
        data_ic=pickle.load(f)
        data_ic=data_ic[(data_ic['Test_Date'] >=str(start))&(data_ic['Test_Date'] <=str(end))]
        ic_fct = pd.DataFrame(np.array([data_ic['Test_Date'],data_ic['0']]).T)
        ic_fct.columns = ['date', fct]
        if ic_df.empty:
            ic_df = ic_fct
        else:
            ic_df = ic_df.merge(ic_fct)

    ic_df['date'] = pd.to_datetime(ic_df['date'], format='%Y-%m-%d')
    ic_df.set_index('date', inplace=True)
    ic_df.tail()
    return ic_df

# 最大化IR法（unshrunk covariance）计算权重
def Caculate_Weight(fct_class_name,fct_class,n,dir,ic_df):
    store_path = dir + '/data_out/class_factor_weight/' + fct_class_name
    isExists = os.path.exists(store_path)
    if not isExists:
        os.makedirs(store_path)
    ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    for dt in ic_df.index:
        ic_dt = ic_df[ic_df.index < dt].tail(n)
        if len(ic_dt) < n:
            continue
        ic_forcov=[]
        for fct in fct_class:
            ic_forcov.append(ic_dt[fct].T.tolist())
        ic_cov_mat = np.mat(np.cov(ic_forcov).astype(float))
        inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
        weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
        weight = np.array(weight.reshape(len(weight), ))[0]
        ic_weight_df.ix[dt] = weight / np.sum(weight)

        '''ic_cov_mat=np.array(ic_cov_mat).reshape(4,4)
        IC = np.array(ic_dt.mean()).reshape(4,1)
        fun = lambda W: (-np.matmul(W.T, IC) / np.sqrt(np.matmul(np.matmul(W.T,ic_cov_mat),W)))  # 约束函数
        cons = ({'type': 'ineq', 'fun': lambda W: W - e})
        W0 = [[0.4],[0.4],[0.4],[0.4]]
        res = minimize(fun, W0, method='SLSQP', constraints=cons)
        ic_weight_df.ix[dt] = res.x'''

    ic_weight_df=ic_weight_df.dropna(axis=0,how='any')
    color = ['green', 'blue', 'orange', 'gray']
    for fct in fct_class:
        plt.plot(ic_weight_df.index, ic_weight_df[fct], color=color[fct_class.index(fct)])
    plt.legend()
    plt.title('Factor weight using sample covariance')
    plt.savefig(store_path+'/weight_maxIR.png')
    plt.close()

    return ic_weight_df

#IC加权法计算权重
def Caculate_Weight_IC(fct_class_name,fct_class,n,dir,ic_df):
    store_path = dir + '/data_out/class_factor_weight/' + fct_class_name
    isExists = os.path.exists(store_path)
    if not isExists:
        os.makedirs(store_path)
    ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    for dt in ic_df.index:
        ic_dt = ic_df[ic_df.index < dt].tail(n)
        if len(ic_dt) < n:
            continue
        ic_mean=[]
        for fct in fct_class:
            ic_mean.append(ic_dt[fct].mean())
        ic_weight_df.ix[dt] = ic_mean / np.sum(ic_mean)
    ic_weight_df = ic_weight_df.dropna(axis=0, how='any')
    color = ['green', 'blue', 'orange','gray']
    for fct in fct_class:
        plt.plot(ic_weight_df.index, ic_weight_df[fct], color=color[fct_class.index(fct)])
    plt.legend()
    plt.title('Factor weight using sample covariance')
    plt.savefig(store_path +  '/weight_IC.png')
    plt.close()
    return ic_weight_df

# 最大化IR法（Ledoit-Wolf shrink covariance）计算权重
def Caculate_Weight_LW(fct_class_name,fct_class,n,dir,ic_df):
    store_path = dir + '/data_out/class_factor_weight/' + fct_class_name
    isExists = os.path.exists(store_path)
    if not isExists:
        os.makedirs(store_path)
    ic_weight_shrink_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    e = 1e-10  # 非常接近0的值
    lw = LedoitWolf()
    for dt in ic_df.index:
        ic_dt = ic_df[ic_df.index < dt].tail(n)
        if len(ic_dt) < n:
            continue
        ic_cov_mat = lw.fit(ic_dt.values).covariance_
        inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
        weight=np.matmul(inv_ic_cov_mat,np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1))
        #weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
        weight = np.array(weight.reshape(len(weight), ))[0]
        ic_weight_shrink_df.ix[dt] = weight / np.sum(weight)

        '''IC = np.array(ic_dt.mean()).reshape(4,1)
        fun = lambda W: (-(np.matmul(W.T, IC) / np.sqrt(W.T * ic_cov_mat * W)))[0][0]  # 约束函数
        cons = ({'type': 'ineq', 'fun': lambda W: W - e})
        W0 = np.random.rand(len(fct_class), 1)
        res = minimize(fun, W0, method='SLSQP', constraints=cons)
        ic_weight_shrink_df.ix[dt] = res.x'''

    ic_weight_shrink_df=ic_weight_shrink_df.dropna(axis=0,how='any')

    color = ['green', 'blue', 'orange', 'gray']
    for fct in fct_class:
        # ic_weight_df[fct]=np.array(ic_weight_df[fct])/np.array(ic_weight_df['Col_sum'])
        plt.plot(ic_weight_shrink_df.index, ic_weight_shrink_df[fct], color=color[fct_class.index(fct)])
    plt.legend()
    plt.title('Factor weight using sample covariance')
    plt.savefig(store_path + '/weight_maxIR_LW.png')
    plt.close()

    return ic_weight_shrink_df

#输出合成后的因子值
def Caculate_weightFct(fct_class_name,fct_class,dir,ic_weight_df,weight_method):
  store_path = dir +'/class_factor_data/'+fct_class_name+'/'+weight_method
  isExists = os.path.exists(store_path)
  if not isExists:
    os.makedirs(store_path)
  for dt in ic_weight_df.index:
    factor_before_weight = pd.DataFrame()
    factor_weighted = pd.DataFrame()
    weight=np.array(ic_weight_df[ic_weight_df.index==dt])[0]
    for fct in fct_class:
        dt=str(dt).replace('-','')[0:8]
        with open(dir+'/single_factor_data/'+fct+'/factor_value'+dt+'.bin','rb') as f:
            factor_data=pickle.load(f)
            factor_data.columns=['Code',fct]
        if len(factor_before_weight)==0:
            factor_before_weight['Code'] = factor_data['Code']
            factor_before_weight[fct]=factor_data[fct]
        else:
            factor_before_weight=factor_before_weight.merge(factor_data)
    factor_weighted['Code'] =factor_before_weight['Code']
    factor_before_weight=factor_before_weight.drop('Code',1).values
    weight_result=np.dot(weight,factor_before_weight.T)
    factor_weighted['value']=weight_result

    f = open(store_path + '/factor_value' + dt+'.bin', 'wb')
    pickle.dump(factor_weighted, f)
    f.close()

func_caculate_factor={
    'weight_IC':Caculate_Weight_IC,
    'weight_maxIR':Caculate_Weight,
    'weight_maxIR_LW':Caculate_Weight_LW,
}

def bigfac_weight_main(fct_class_name,Date_begin=20080103,Date_end=20081231,weight_method='weight_IC',n=100):
  #确认存储路径
  factors=fct_class[fct_class_name]
  dir=os.path.abspath('..')

  ic_df=Load_IC(dir,factors,Date_begin,Date_end)
  ic_weight_df = func_caculate_factor[weight_method](fct_class_name,factors,n, dir,ic_df)
  Caculate_weightFct(fct_class_name,factors,dir,ic_weight_df,weight_method)
  print('success')

