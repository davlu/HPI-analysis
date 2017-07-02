import pandas as pd
import quandl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
key = "W-zXc_azFuCuXBefqRzr"  #feel free to use

def state_list():
    url = 'https://simple.wikipedia.org/wiki/List_of_U.S._states'
    stateList = pd.read_html(url)
    return stateList

def mortgage_30yr():
    df = quandl.get('FMAC/MORTG',trim_start = "1975-01-01",authtoken = key)
    df['Value']= (df['Value']-df['Value'][0])/df['Value'][0] * 100
    df = df.resample('D').mean()
    df = df.resample('M').mean()
    df.columns = ['M30']
    return df

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=key)
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df
def grab_data():
    main_df = pd.DataFrame()
    stateList = state_list()
    for abbr in stateList[0][0][1:]:
        query= 'FMAC/HPI_' + abbr
        df = quandl.get(query, authtoken= key)
        df.columns = [str(abbr)]
        df[abbr] = (df[abbr]-df[abbr][0])/df[abbr][0] * 100
        if main_df.empty:
            main_df=df
        else:
            main_df = main_df.join(df)
    main_df.to_pickle('piccle.pickle')

def benchmark():
    df = quandl.get('FMAC/HPI_USA', authtoken= key)
    df['United States HPI'] = (df['Value']-df['Value'][0])/df['Value'][0] * 100   #percent change from start
    del df['Value']
    return df

def labels(fut_hpi, cur_hpi):
    if fut_hpi>cur_hpi:
        return 1
    else:
        return 0

def rol_mean(value):
    return np.mean(value)

HPI_data = pd.read_pickle('piccle.pickle')
GDP = gdp_data()

us_unemployment = us_unemployment()
HPI_m30 = mortgage_30yr()
HPI_Benchmark = benchmark()
Combined_HPI = HPI_data.join([HPI_m30,GDP,us_unemployment, HPI_Benchmark])
Combined_HPI.dropna(inplace=True)
Combined_HPI.to_pickle('picle.pickle')
print(HPI_data.describe())

HPI_correlation = HPI_data.corr()
HPI_correlation.set_index('TX',inplace = True)
grab_data()
benchmark = benchmark()
benchmark.plot(ax=ax1, color = 'k', linewidth = 10)

hpi_data = pd.read_pickle('picle.pickle').pct_change()
hpi_data.replace([np.inf,-np.inf], np.nan, inplace = True)
hpi_data['US_future_HPI'] = hpi_data['United States HPI'].shift(-1)
hpi_data.dropna(inplace= True)
hpi_data['Labels'] = list(map(labels,hpi_data['US_future_HPI'], hpi_data['United States HPI']))
hpi_data['applied m30'] = hpi_data['M30'].rolling(window=10,center = False).apply(rol_mean)


fig = plt.figure()
ax1 = plt.subplot2grid((2,1),(1,0))
ax2 = plt.subplot2grid((2,1),(0,0),sharex= ax1)
hpi_data = pd.read_pickle('piccle.pickle')
hpi_data['TXAK_CORR'] = hpi_data['TX'].rolling(window=12, center = False).corr()
hpi_data['TX12MA'] = hpi_data['TX'].rolling(window=12, center = False).mean()
hpi_data['TX_STD'] = hpi_data['TX'].rolling(window=12,center=False).std()
hpi_data.dropna(how='any',inplace= True)
print(hpi_data[['TX12MA','TX_STD']])
hpi_data.fillna(value= -9999,  inplace = True)
hpi_data[['TX','TX12MA']].plot(ax=ax1)
# hpi_data['TX_STD'].plot(ax=ax2, legend = 'STD')
hpi_data['TX'].plot(ax=ax1)
hpi_data['AK'].plot(ax=ax1)
hpi_data['TXAK_CORR'].plot(ax=ax2)
plt.legend()
plt.show()
 #testing df 
df1 = pd.DataFrame({'Year':[2001,2002,2003,2004],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]})


df3 = pd.DataFrame({'Year':[2001,2003,2004,2005],
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53]})

df4 = pd.merge(df1,df3,on='Year',how='inner')
print(df4)
df1.set_index('Year',inplace = True)
df3.set_index('Year',inplace = True)
joined = df1.join(df3, how = 'inner')