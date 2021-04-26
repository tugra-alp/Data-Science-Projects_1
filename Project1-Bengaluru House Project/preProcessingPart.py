#%% 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#%%
# link of dataset = 'https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data'
data = pd.read_csv("Bengaluru_House_Data.csv")
#%%   ---- DATA CLEANING ----
# Removing unnecessary features
data = data.drop(['area_type','society','balcony','availability'], axis = 'columns')
#%%
# data.isnull().sum()
data = data.dropna()
#%% 
# 4 BHK = 4 Bedroom - bhk (Bedrooms Hall Kitchen)
# tokenize example : data['size'][1].split(' ')[0]
data['BHK'] = (data['size'].apply(lambda x: x.split(' ')[0])).astype(float)
#%% to detect range values in total_sqft 
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
data[~data['total_sqft'].apply(is_float)].head()
#%%  to convert range values to single value (taking mean of them) in total_sqft
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
#convert_sqft_to_num('1120 - 1145') : an example to see how func works
df = data.copy()
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df[df.total_sqft.notnull()]
#%% ---------- Feature Engineering ------------  
df1 = df.copy()
#this new feature will help outlier detection
df1['per_ft_price'] = df1['price']*100000 / df1['total_sqft']

#%% Dim. Reduction for 'location' column. 
# Any location having less than 10 data points should be tagged as "other" location.
df1.location = df1.location.apply(lambda x: x.strip())
location_stats = df1['location'].value_counts(ascending=False)
less_than_10_loc = location_stats[location_stats <= 10]
df1.location = df1.location.apply(lambda x: 'other' if x in less_than_10_loc else x)
#%%
# There are some disproportion between size and total_sqft
# for per bedroom , 300 sq ft is expected. (300 sq feet = 27.8 m^2)
# i.e 2 bhk apartment is minimum 600 sqft.
df1[(df1.total_sqft/df1.BHK)<300].head() # i.e; BHK:6 total_sqft: 1020.0 is data error.
df2 = df1[~(df1.total_sqft/df1.BHK<300)] # without data errors ( 744 points removed)
#%% Outlier Removal Using Standard Deviation and Mean
# removing outliers for per location and according to their std & mean
def remove_pfp_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.per_ft_price)
        st = np.std(subdf.per_ft_price)
        reduced_df = subdf[(subdf.per_ft_price>(m-st)) & (subdf.per_ft_price<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3 = remove_pfp_outliers(df2)
#%% comparing house price 2 and 3 BHK with given a location to DETECT another outliers
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df3,"Rajaji Nagar")
#%% i.e: we should remove the price of 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.per_ft_price),
                'std': np.std(bhk_df.per_ft_price),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.per_ft_price<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df4 = remove_bhk_outliers(df3)
df4.shape # removed 2925 outlier
#%% plot same scatter after removing outliers
plot_scatter_chart(df4,"Rajaji Nagar")

#%% bathroom outliers
# total bath = total bed + 1 max (Anything above that is an outlier or a data error and can be removed)
df4[df4.bath>df4.BHK+2] # to see outliers
df5 = df4[df4.bath<=df4.BHK+1]
#%%  after preprocessing we dont need some features anymore
df6 = df5.drop(['size','per_ft_price'],axis='columns') #per_ft_price is used only outlier detection
#'BHK' feature instead of 'size'
#%% Use One Hot Encoding For Location col.
dummies = pd.get_dummies(df6.location)
df7 = pd.concat([df6,dummies.drop('other',axis='columns')],axis='columns')
#%%  saving pre-processed dataframe as csv file
finaldf = df7.drop('location',axis='columns')

#finaldf.to_csv("preProcessedData.csv")
