import pandas as pd
import numpy as np
import pickle
import os
import pymysql
import statsmodels.api as sm



conn=pymysql.connect(host='10.50.48.26',user='root',password='password',database='ori_data')
cursor=conn.cursor()
select_order="CREATE TABLE IF NOT EXISTS %s "
cursor.execute(select_order,('test'))

dir=os.path.abspath('..')
facUse=['value','growth']
weightMethod='IC'
#得到HS300成分股的收益率序列
with open(dir+'/HS300_mem','rb') as f:
    index_mem=pickle.load(f)
f.close()
index_mem=index_mem.set_index('Date',drop=True)
Date=index_mem.index.drop_duplicates().values
subfac=pd.DataFrame(columns=facUse,index=Date)
facret_oneday=pd.DataFrame(columns=facUse,index=np.arange(1,13))
facretMatrix=pd.DataFrame(columns=facUse,index=Date)
for date in Date:
    print(date)
    Mem_change=index_mem.loc[date].set_index('Code',drop=True)
    print(Mem_change)
    for back in range(1,12):
        for fac in facUse:
            with open('//10.50.48.26/multiFactor/'+fac+'/'+weightMethod+'/factor_value_'+(date-back*5)+'.bin','rb') as f:
                subfac[fac]=pickle.load(f)
        facret_oneday.loc[back]=(sm.OLS(Mem_change,subfac).fit().params)
    facret=facret_oneday.average()
    facretMatrix.loc[date]=facret
    with open('//10.50.48.26/factorReturn/return_'+date+'.bin','wb') as f:
        pickle.dump(facret_oneday,f)

halfLife=12
timeWindow=52
tw=np.arange(1,53)
weight=np.power(2,(tw-timeWindow)/halfLife)
weight=weight/np.sum(weight)
print(weight)
weightFac=np.dot(facretMatrix,weight)
facretMatrix.to_excel('facReturn.xlsx')










#得到HS300成分股的因子值
fac_name=['size','value']
