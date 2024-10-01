import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
import os
######################################################start conf plot
# import seaborn as sns
import matplotlib.pyplot as plt;
import matplotlib.ticker as plticker
import matplotlib.dates as md
from pandas.plotting import table
fig_size_l=[8,5]
fig_size=(fig_size_l[0], fig_size_l[1])# (10, 8)
# sns.set(rc={'figure.figsize': fig_size})
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.grid']=True
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rcParams["font.family"] = "DejaVu Sans" #"Times New Roman"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
######################################################end conf plot

file_location = os.path.dirname(os.path.abspath(__file__))

name_case='record_'+'01'
file = file_location+'/'+f'../outputs/{name_case}.h5'


df_meta=pd.read_hdf(file,key="/meta_data")
# Adding the area
file_area = file_location+'/'+'../config/ybus_area.h5'
with pd.HDFStore(file_area, mode='r') as hdf:
    keys = hdf.keys()
ybus={}
for i in keys:
    ybus[i.replace('/','').replace('-','_')]=pd.read_hdf(file_area,key=i)
df_list=[]
for key,value in ybus.items():
    temp=list(value.columns)
    temp=pd.DataFrame({"index":temp})
    temp['area']=key
    df_list.append(temp)
df_list=pd.concat(df_list)
df_list['index']=df_list['index'].str.split('.').str[0].str.lower()
df_list=df_list.drop_duplicates()
df_list.reset_index(drop=True,inplace=True)
df_meta=df_meta.merge(df_list,right_on='index',left_on='f_node_name',how='left').drop(columns='index')
df_meta.rename(columns={'area':'f_area'},inplace=True)
df_meta=df_meta.merge(df_list,right_on='index',left_on='t_node_name',how='left').drop(columns='index')
df_meta.rename(columns={'area':'t_area'},inplace=True)
df_meta['area']=df_meta['f_area']
df_meta['t_area']=df_meta['t_area'].fillna(df_meta['f_area'])
df_meta.loc[df_meta['area'] != df_meta['t_area'],'area']=np.nan
# df_meta.loc[df_meta['area'] != df_meta['t_area'],['area','f_area','t_area']]


df=pd.read_hdf(file,key="/a0")
df['value'] = df['value'].astype(str)
df.rename(columns={'unique_id':'measurement_mRID'},inplace=True)
df = df.pivot_table(index='timestamp', columns='measurement_mRID', values='value', aggfunc='first')
df = df.map(lambda x: complex(x))
# df.fillna(0)
df=df.dropna()
#### measurement types
print(f"measurement types: {df_meta['measurement_type'].unique()}")
df_pnv=df[df_meta.loc[df_meta['measurement_type']=='PNV','measurement_mRID'].to_list()]
df_magnitude = df_pnv.map(lambda x: abs(x))


area_color={
    "area_1":"#e0e4e7",
    "area_2":"#fee1cc",
    "area_3":"#ecdcd5",
    "area_4":"#f7cce3",
    "area_5":"#dfeed1",
    "area_6":"#ccdcfc"
}

# Create a figure with subplots, one below the other
def round_to_closest_minute(dt):
    # Calculate the number of seconds to round
    seconds = (dt - dt.replace(second=0, microsecond=0)).total_seconds()
    
    # If seconds are 30 or more, round up, otherwise round down
    if seconds >= 30:
        dt = dt + timedelta(minutes=1)
    
    return dt.replace(second=0, microsecond=0)
def get_time_from_messages(): # TODO: make it generic 
    df=pd.read_hdf(file,key="/message_record")
    return [
            round_to_closest_minute(min(df['simulation_time'])-timedelta(seconds=120)),
            round_to_closest_minute(min(df['simulation_time'])),
            round_to_closest_minute(max(df.loc[df['type']=='self_connection','simulation_time'])),
            ]
    # df[['type','simulation_time']]

times_to_plot=get_time_from_messages()
num_rows=len(  times_to_plot   )
fig, axes = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))
# Loop through each row of the DataFrame and create a subplot
for i, time in enumerate(times_to_plot):
    print(i)
    ax = axes[i]
    data=df_magnitude.loc[time]
    data=data.reset_index()
    data=data.merge(df_meta[['measurement_mRID','area']],on='measurement_mRID',how='right')
    data.columns = data.columns.astype(str)
    data=data.sort_values(by='area')
    data.reset_index(drop=True,inplace=True)
    data.dropna(inplace=True)
    # Plot data from the row on the current subplot
    for k,v in area_color.items():
        data_t=data.loc[data['area']==k].copy()
        ax.plot(data_t.index, data_t[str(time)],color=v, alpha=1, linewidth=20)#, marker='o')
    ax.grid()
    ax.set_ylabel('voltage (V)')
    ax2 = ax.twinx()
    ax2.set_ylabel(f'time {time}')
    ax2.set_yticks([])
# Adjust spacing between subplots
ax.set_xlabel('nodes')
legend_labels = [plt.Line2D([0], [0], marker='s', color='w', label=key, markersize=20, markerfacecolor=color) for key, color in area_color.items()]
plt.legend(handles=legend_labels, title='',loc='lower center',bbox_to_anchor=(0.5, -0.35),ncol=len(area_color)) # bbox_to_anchor=(-0.1, -0.1)
plt.tight_layout()
plt.show()
# Show the plot
plt.savefig(f'system_state_{name_case}.png')
plt.savefig(f'system_state_{name_case}.svg')


