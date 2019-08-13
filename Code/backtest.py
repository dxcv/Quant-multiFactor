import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
import sys
import os
def GetTestDate(Date_begin,Date_end,Date_data):
    Date_data.columns = ['Date']
    start = Date_data.loc[Date_data['Date'] >= Date_begin].index.values[0]
    end = Date_data.loc[Date_data['Date'] <= Date_end].index.values
    Test_Date = Date_data.iloc[start:end[len(end) - 1]+1].reset_index(drop=False)['Date']
    return Test_Date
# 回测-组合回测、多空、对冲指数；计算sharp ratio、信息比率、最大回撤
def BackTest(factor_name,factor_data,dir,Trade_Date):

    #时间处理成datetime格式，以方便画图
    Days=len(Trade_Date)
    X_Date=[]
    for d in range(Days):
        X_Date.append(datetime.datetime.strptime(str(Trade_Date[d]), "%Y%m%d"))
    #初始化累计收益率为1,初始化累计收益率序列为5列的dataframe
    portfolio_return_d=np.zeros(5)
    portfolio_return_d_c=np.ones(5)
    portfolio_return_d_c_series=pd.DataFrame(columns=['Group1','Group2','Group3','Group4','Group5'])
    portfolio_return_m_series= pd.DataFrame(columns=['Group1', 'Group2', 'Group3', 'Group4', 'Group5'])
    port_best=pd.DataFrame()
    turnover=pd.DataFrame(columns=['Adjustment_Date','value'])

    for t in range(round(Days/21)):
        turnover_tuple = []
        print(t)
        #每21个交易日调仓，获取组合股票（暂时按照单因子值排序）
        data_adjust = factor_data.loc[factor_data['Trade_Date'] == Trade_Date[t*21]].reset_index(drop=True)
        turnover_tuple.append(X_Date[t*21])
        if t!=0:
            last_portfolio=portfolio
            last_p=last_portfolio[round(4 * count / 5):(round(5 * count / 5) - 1)]
            portfolio = data_adjust.sort_values(by='value',ascending=True).reset_index(drop=False)['Code']
            p=portfolio[round(4 * count / 5):(round(5 * count / 5) - 1)]
            overlap=len(pd.merge(last_p,p,on='Code'))
            turnover_tuple.append((len(last_p)-overlap)/len(last_p))
        else:
            portfolio = data_adjust.sort_values(by='value', ascending=True).reset_index(drop=False)['Code']
            turnover_tuple.append(0)
        turnover.loc[len(turnover)]=turnover_tuple
        #按日累计收益率
        for i in range(21):
            mean_pf = np.zeros(5)
            if (t*21 + 1 + i)==len(X_Date):
               break
            data_t = factor_data.loc[factor_data['Trade_Date'] == Trade_Date[t*21 + 1 + i]].reset_index(drop=True)
            count = len(portfolio)
            port_best = port_best.append(portfolio[round(4 * count / 5):(round(5 * count / 5) - 1)])
            #日组合平均收益率
            for groups in range(5):
              port = portfolio[round(groups * count / 5):(round((groups + 1) * count / 5) - 1)]
              data_mean=data_t.loc[data_t['Code'].isin(port)]['PCTchange']
              mean_pf[groups]=np.mean(data_mean)
              #组合至今累计收益率
              portfolio_return_d_c[groups]=(1+mean_pf[groups]/100)*portfolio_return_d_c[groups]
              portfolio_return_d[groups] = mean_pf[groups] / 100
            portfolio_return_d_c_series.loc[len(portfolio_return_d_c_series)]=portfolio_return_d_c
        portfolio_return_m_series.loc[len(portfolio_return_m_series)]=portfolio_return_d_c

    portfolio_return_m=pd.DataFrame(columns=['Group1', 'Group2', 'Group3', 'Group4', 'Group5'])

    for t in range(round(Days / 21)):
        if t!=0:
            a=np.array(portfolio_return_m_series.loc[t])/np.array(portfolio_return_m_series.loc[t-1])
            portfolio_return_m.loc[len(portfolio_return_m)]=a-1
        else:
            portfolio_return_m.loc[len(portfolio_return_m)] = np.array(portfolio_return_m_series.loc[t])
    #计算比率
    portfolio_len=len(portfolio_return_d_c_series)
    portfolio_sharp_ratio=[]
    Max_Drawdown=[]
    CAR=[]
    color_choose=['lawngreen','skyblue','steelblue','saddlebrown','purple','red']
    result_path = dir + '/data_out/test_out/backtest_out/' + factor_name + '_backtest'
    isExists = os.path.exists(result_path)
    if not isExists:
        os.makedirs(result_path)
    f = open(result_path + '/pct_fenzu', 'wb')
    pickle.dump(portfolio_return_d_c_series, f)
    f.close()
    annu_yeild=[]
    for g in range(5):
        temp_return=portfolio_return_m['Group'+str(g+1)]
        temp_return_c=portfolio_return_d_c_series['Group'+str(g+1)]
        # 计算组合年化超额收益
        a = temp_return_c[portfolio_len - 1]
        excess_return = a - 0.05
        annu_yeild.append(np.power(a,1/10)-1)
        CAR.append(np.power(excess_return,1/10)-1)
        #计算夏普比率
        mean=np.mean(temp_return)
        std = np.std(temp_return, ddof=1)
        sharp_ratio=(mean)/std
        portfolio_sharp_ratio.append(sharp_ratio*np.sqrt(12))
        #计算最大回撤
        np.maximum.accumulate(temp_return_c)
        j = np.argmax((np.maximum.accumulate(temp_return_c) - temp_return_c)/np.maximum.accumulate(temp_return_c))
        i = np.argmax(temp_return_c[:j])
        Max_Drawdown.append((temp_return_c[i] - temp_return_c[j])/temp_return_c[i])
        # 画出收益率曲线
        plt.plot(X_Date[0:portfolio_len],temp_return_c,color=color_choose[g], label='y'+str(g))

    turnover=pd.DataFrame(np.array(turnover))
    turnover.to_excel(result_path+'/turn_over.xlsx')
    back_test = pd.DataFrame(np.array([annu_yeild,CAR, portfolio_sharp_ratio, Max_Drawdown]))
    back_test.to_excel(result_path+'/back_test.xlsx')

    return_01=portfolio_return_d_c_series['Group5']-portfolio_return_d_c_series['Group1']

    plt.plot(X_Date[0:portfolio_len], return_01,color=color_choose[5], label='dif')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(result_path+'/Cumulative_return.png')
    plt.close()
    turnover.columns=['Adjustment_Date','value']
    plt.plot(turnover['Adjustment_Date'], turnover['value'], color=color_choose[5], label='dif')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(result_path + '/Turn_over.png')

def backtest_main(factor_name,Date_begin=20090101,Date_end=20121231,fct_class='single',weight_method='weight_IC'):
    dir = os.path.abspath('..')
    # 读出价格数据
    f = open(dir + '/data_in/price', 'rb')
    price_data = pickle.load(f)
    f.close()
    price_data.columns = ['Code', 'Trade_Date', 'price', 'PCTchange']
    # 获取测试时间
    Date_data = pd.read_table(dir + "/data_in/Trade_Date.txt", header=None)
    Test_Date = GetTestDate(Date_begin, Date_end, Date_data)
    factor_data = pd.DataFrame()
    # 读出因子数据
    if fct_class=='class':
        read_path=dir+'/'+fct_class+'_factor_data/'+factor_name+'/'+weight_method
    else:
        read_path = dir+'/'+fct_class+'_factor_data/'+factor_name
    for t in range(len(Test_Date)):
        if t != 0:
            with open(read_path+'/factor_value'+str(Test_Date.iloc[t])+'.bin', 'rb')  as f:
                factor_data_temp = pickle.load(f)
                factor_data_temp['Trade_Date'] = Test_Date.iloc[t].astype(np.int64)
            factor_data = pd.concat([factor_data, factor_data_temp], axis=0)
        else:
            with open(read_path+'/factor_value'+str(Test_Date.iloc[t])+'.bin', 'rb') as f:
                factor_data = pickle.load(f)
                factor_data['Trade_Date'] = Test_Date.iloc[t].astype(np.int64)
    factor_data.columns = ['Code', 'value', 'Trade_Date']

    # 合成函数需要的数据
    factor_data_final = pd.merge(price_data, factor_data, on=['Trade_Date','Code'])
    sys.path.append(dir)
    #进行单因子回测
    BackTest(factor_name,factor_data_final,dir,Test_Date)
    print('success')




