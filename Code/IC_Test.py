import pandas as pd
import pickle
import sys

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


def GetTestDate(Date_begin,Date_end,Date_data):
    Date_data.columns = ['Date']
    start = Date_data.loc[Date_data['Date'] >= Date_begin].index.values[0]
    end = Date_data.loc[Date_data['Date'] <= Date_end].index.values
    Test_Date = Date_data.iloc[start:end[len(end) - 1]+1].reset_index(drop=False)['Date']
    return Test_Date
#ic和rankic计算
# 数据覆盖度分析-全a覆盖，中证500覆盖，沪深300覆盖
def caculate_ic(factor_name,factor_data,dir,test_period,Test_Date):

    result_path=dir + '/data_out/test_out/ic_test/' + factor_name + '_ictest'
    isExists = os.path.exists(result_path)
    if not isExists:
        os.makedirs(result_path)
    test_period_list = ['Test_Date']
    temp = np.arange(int(test_period))
    temp = [str(x) for x in temp]
    test_period_list.extend(temp)

    ic_series = pd.DataFrame(columns=test_period_list)
    rankic_series = pd.DataFrame(columns=test_period_list)
    regression_cof_t_se = pd.DataFrame(columns=test_period_list)
    regression_cof_t_se_abs = pd.DataFrame(columns=test_period_list)
    regression_cof_r_se = pd.DataFrame(columns=test_period_list)
    factor_return_se = pd.DataFrame(columns=test_period_list)
    cover_se=[]
    for t in range(len(Test_Date)-(test_period+1)*5):
        print(Test_Date[t])
        ic = []
        rankic = []
        regression_cof_t=[]
        regression_cof_t_abs=[]
        regression_cof_r=[]
        factor_return=[]
        ic.append(str(Test_Date[t]))
        rankic.append(str(Test_Date[t]))
        regression_cof_t.append(str(Test_Date[t]))
        regression_cof_t_abs.append(str(Test_Date[t]))
        regression_cof_r.append(str(Test_Date[t]))
        factor_return.append(str(Test_Date[t]))
        if (t+5)>=len(Test_Date):
            break
        data=factor_data.loc[factor_data['Trade_Date'] == Test_Date[t]]
        cover_se.append(len(data))
        for w in range(1,test_period+1):
           test_period_begin=factor_data.loc[factor_data['Trade_Date'] == Test_Date[t+5*(w-1)]].reset_index(drop=True)
           test_period_end = factor_data.loc[factor_data['Trade_Date'] == Test_Date[t+5*w]].reset_index(drop=True)
           data_merge=pd.merge(data, test_period_begin,on=['Code'])
           data_merge=pd.merge(data_merge,test_period_end,on=['Code'])

           fac = data_merge['value_x'].reset_index(drop=True)
           fac.fillna(fac.mean(),inplace=True)
           #回归分析
           C_yeild = (data_merge['price'] - data_merge['price_y']) / data_merge['price_y']
           fac_t = sm.add_constant(fac)
           reg=sm.OLS(C_yeild.astype(float), fac_t.astype(float)).fit()
           factor_return.append(reg.params['value_x'])
           regression_cof_t.append(reg.tvalues[1])
           regression_cof_t_abs.append(np.abs(reg.tvalues[1]))
           regression_cof_r.append(reg.rsquared)
           # ic-相关系数

           df = pd.DataFrame({'value': fac, 'PCTchange': C_yeild})
           ic.append(df.corr()['PCTchange']['value'])
           #rankic-秩相关系数
           rankic.append(df.corr('spearman')['PCTchange']['value'])
        regression_cof_t_se.loc[len(regression_cof_t_se)]=regression_cof_t
        regression_cof_r_se.loc[len(regression_cof_r_se)] = regression_cof_r
        regression_cof_t_se_abs.loc[len(regression_cof_t_se_abs)]=regression_cof_t_abs
        factor_return_se.loc[len(factor_return_se)]=factor_return
        ic_series.loc[len(ic_series)] = ic
        rankic_series.loc[len(rankic_series)] = rankic

    store_ic_path = dir + '/data_out/factor_ic/' + factor_name
    isExists = os.path.exists(store_ic_path)
    if not isExists:
        os.makedirs(store_ic_path)
    f = open(store_ic_path + '/'+factor_name + '_ic', 'wb')
    pickle.dump(ic_series, f)
    f.close()
    f = open(store_ic_path + '/'+ factor_name + '_rankic', 'wb')
    pickle.dump(rankic_series, f)
    f.close()
    f = open(store_ic_path + '/' + factor_name + '_t', 'wb')
    pickle.dump(regression_cof_t_se, f)
    f.close()
    f = open(store_ic_path + '/' + factor_name + '_rsquare', 'wb')
    pickle.dump(regression_cof_r_se, f)
    f.close()

    #计算ic和rankic均值序列 并写入二进制
    t_means = []
    r_means = []
    ic_win2=[]
    t_win2=[]
    t_abs_means=[]
    r_win=[]
    ic_means = []
    rankic_means = []
    ic_std = []
    ic_win = []
    rankic_std = []
    for cols in range(test_period):
        r_win.append((factor_return_se[str(cols)] > 0).astype(int).sum(axis=0)/len(factor_return_se[str(cols)]))
        t_win2.append((regression_cof_t_se_abs[str(cols)] > 2).astype(int).sum(axis=0)/len(regression_cof_t_se_abs[str(cols)]))
        t_abs_means.append(regression_cof_t_se_abs[str(cols)].mean())
        t_means.append(regression_cof_t_se[str(cols)].mean())
        r_means.append(regression_cof_r_se[str(cols)].mean())
        ic_means.append(ic_series[str(cols)].mean())
        ic_win.append((ic_series[str(cols)] > 0).astype(int).sum(axis=0)/len(ic_series[str(cols)]))
        ic_win2.append((ic_series[str(cols)] > 0.02).astype(int).sum(axis=0)/len(ic_series[str(cols)]))
        rankic_means.append(rankic_series[str(cols)].mean())
        ic_std.append(ic_series[str(cols)].std())
        rankic_std.append(rankic_series[str(cols)].std())

    ir = np.array(ic_means) / np.array(ic_std)
    rankir = np.array(rankic_means) / np.array(rankic_std)
    t_test=pd.DataFrame(np.array([r_win,t_means,t_abs_means,t_win2, r_means]))
    ic_test = pd.DataFrame(np.array([ic_means, ic_std, ic_win,ic_win2,rankic_means, ir, rankir ]))
    t_test.index=['r_win','t_means','t_abs_means','t_win2','r_means']
    ic_test.index=['ic_means','ic_std','ic_win','ic_win2','rankic_means','ir','rankir']
    t_test.to_excel(result_path+'/t_test.xlsx')
    ic_test.to_excel(result_path+'/ic_test.xlsx')

    # 时间处理成datetime格式，以方便画图
    Days = len(Test_Date)
    X_Date = []
    for d in range(Days):
        X_Date.append(datetime.datetime.strptime(str(Test_Date[d]), "%Y%m%d"))

    ic_c = np.cumsum(ic_series['0'])
    plt.plot(X_Date[0:(len(X_Date) - (test_period+1)*5)], ic_c, color='red', label='dif')
    plt.savefig(result_path+'/ic_c.png')
    plt.close('all')
    plt.plot(X_Date[0:(len(X_Date) - (test_period + 1) * 5)], cover_se, color='red', label='dif')
    plt.savefig(result_path + '/cover.png')



def IC_Test_main(factor_name,Date_begin=20090101,Date_end=20121231,test_period=6,fct_class='single',weight_method='weight_IC'):
    dir=os.path.abspath('..')
    #读出价格数据
    f = open(dir + '/data_in/price', 'rb')
    price_data=pickle.load(f)
    f.close()
    price_data.columns=['Code','Trade_Date','price','PCTchange']
    #获取测试时间
    Date_data = pd.read_table(dir + "/data_in/Trade_Date.txt", header=None)
    Test_Date=GetTestDate(Date_begin,Date_end,Date_data)
    factor_data=pd.DataFrame()
    #读出因子数据
    if fct_class=='class':
        read_path=dir+'/'+fct_class+'_factor_data/'+factor_name+'/'+weight_method
    else:
        read_path = dir+'/'+fct_class+'_factor_data/'+factor_name
    for t in range(len(Test_Date)):
        if t!=0:
            with open(read_path+'/factor_value'+str(Test_Date.iloc[t])+'.bin', 'rb') as f:
                factor_data_temp = pickle.load(f)
                factor_data_temp['Trade_Date'] = Test_Date.iloc[t].astype(np.int64)
            factor_data=pd.concat([factor_data,factor_data_temp],axis=0)
        else:
            with open(read_path+'/factor_value'+str(Test_Date.iloc[t])+'.bin', 'rb') as f:
                factor_data = pickle.load(f)
                factor_data['Trade_Date'] = Test_Date.iloc[t].astype(np.int64)
    factor_data.columns=['Code','value','Trade_Date']
    #合成函数需要的数据
    factor_data_final=pd.merge(price_data,factor_data,on=['Trade_Date' ,'Code'])
    sys.path.append(dir)
    #进行测试
    caculate_ic(factor_name,factor_data_final,dir,test_period,Test_Date)
    print('success')




