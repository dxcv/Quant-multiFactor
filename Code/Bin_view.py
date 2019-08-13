import pickle
import sys
import os
import pandas as pd
def GetTestDate(Date_begin,Date_end,Date_data):
    Date_data.columns = ['Date']
    start = Date_data.loc[Date_data['Date'] > Date_begin].index.values[0]
    end = Date_data.loc[Date_data['Date'] < Date_end].index.values
    Test_Date = Date_data.iloc[start:end[len(end) - 1]].reset_index(drop=False)['Date']
    return Test_Date
if __name__=="__main__":
    dir=os.path.abspath('..')
    path = sys.argv[1]
    f = open(dir+path, 'rb')
    data=pickle.load(f)
    print(data)
    f.close()


