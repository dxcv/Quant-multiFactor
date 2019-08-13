import numpy as np
import pandas as pd
import pymysql
import pickle
import os
import time
dir=os.path.abspath('..')
'''
with open(dir+'/data_in/price','rb') as f:
    price=pickle.load(f)
price.columns=['Date','Code','Close','Change']
price=price.set_index('Date',drop=True)
Date=price.index.drop_duplicates().values[5:]
weekReturn=pd.DataFrame(columns=['Code','WeeklyReturn'])
for dateIndex in range(len(Date)):
    a = price.loc[Date[dateIndex - 5]].reset_index(drop=True)
    b = price.loc[Date[dateIndex]].reset_index(drop=True)
    res=pd.merge(a,b,on='Code')
    weekReturn['WeeklyReturn']=res['Close_y']/res['Close_x']-1
    weekReturn['Code']=res['Code']
    with open(dir+'/data_out/weekReturn/weekReturn_'+str(Date[dateIndex])+'.bin','wb') as f:
        pickle.dump(weekReturn,f)
    print(dateIndex)'''
date=time.strftime("%Y%m%d")
conn=pymysql.connect(host='10.50.48.26',user='root',password='password',database='ori_data')
cursor=conn.cursor()
#找到交易日
selectTradeday='select DISTINCT TRADE_DAYS from ASHARECALENDAR WHERE TRADE_DAYS<=%s ORDER BY TRADE_DAYS DESC LIMIT 5'
cursor.execute(selectTradeday,(date))
Tradeday=cursor.fetchall()
lastDate=str(Tradeday[-1])[2:-3]
print(lastDate)

select_order="select S_INFO_WINDCODE,S_DQ_CLOSE from ASHAREEODPRICES where TRADE_DT=%s OR TRADE_DT=%s"
cursor.execute(select_order,(date,lastDate))
Close=cursor.fetchall()
print(Close)
print('success')

