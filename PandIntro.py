import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader.data as web
import datetime
import numpy as np

df = pd.read_csv('ZILL-Z77006_LPC.csv')
print(df.head())
df.set_index('Date', inplace = True)
df.to_csv('new.csv')
df = pd.read_csv('new.csv', index_col= 'Date')
df.columns = ['AUSTIN HPI']
print(df.head())
df.to_csv('new.csv', header = False)
df = pd.read_csv('new.csv', names = ['Date','Austin'])
df.rename(columns = {'Austin': 'A', 'Date':'D'}, inplace = True)
print(df.head())

l = [1,2,3,4]
print(l[-2])



info = {'day': [1,2,3,4,5,6],
        'Visitors': [44,55,66,77,88,99],
        'Bouncers': [100,110,120,130,140,150]}
df = pd.DataFrame(info)
df.set_index('day', inplace = True)
print(df[['Bouncers','Visitors']])
print(pd.DataFrame(np.array(df[['Bouncers','Visitors']])))


start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015, 8 ,22)
df = web.DataReader("XOM", "google", start, end)
style.use('fivethirtyeight')
df['High'].plot(linewidth = 2)
plt.legend()
plt.show()

