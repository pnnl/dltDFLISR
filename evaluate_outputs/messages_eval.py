# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:17:15 2023

@author: bere240
"""
import os
import pandas as pd
######################################################start conf plot
# import my_conf as CONF
# import seaborn as sns
import matplotlib.pyplot as plt;
import matplotlib.ticker as plticker
import matplotlib.dates as md
from pandas.plotting import table
fig_size_l=[10,5]
fig_size=(fig_size_l[0], fig_size_l[1])# (10, 8)
# sns.set(rc={'figure.figsize': fig_size})
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.grid']=True
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
######################################################end conf plot

file_location = os.path.dirname(os.path.abspath(__file__))

#### getting the message record
name_case='record_'+'01'
file = file_location+'/'+f'../outputs/{name_case}.h5'
df=pd.read_hdf(file,key="/message_record")
df.columns
#### adding colors to the messages based on type
df['type'].unique() # get the types
# Define color map for different message types
color_map = {
    'self_connection': 'blue',
    'eval_reconnection_path': 'red',
    'reply_eval_reconnection_path': 'green'
}
color_map_legend = {
    'connection': 'blue',
    'request path evaluation': 'red',
    'return path requested': 'green'
}
#### making the message plot
plt.figure(figsize=(10, 6))
custom_agent_order = ['area_1','area_2','area_3','area_4','area_5','area_6']
for agent in custom_agent_order:
    agent_data = df[df['_from'] == agent]
    plt.scatter(agent_data['_from'], agent_data['simulation_time'],
                c=[color_map[msg_type] for msg_type in agent_data['type']],
                label=agent)
    agent_data = df[df['_to'] == agent]
    plt.scatter(agent_data['_to'], agent_data['simulation_time_received'],
                c=[color_map[msg_type] for msg_type in agent_data['type']],
                label=agent)
    #### Annotate with arrows
for index, row in df.iterrows():
    color="blue"
    if row['message_received']=="False":
        color='red'
    plt.annotate("", xy=(row['_to'], row['simulation_time_received']), xytext=(row['_from'], row['simulation_time']),
                 arrowprops=dict(arrowstyle="->", color=color))
    #### Get the tick labels of the x-axis
x_tick_labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]
    #### Add black vertical lines for agents
for i in range(len(x_tick_labels)):
    plt.axvline(x=i, color='black', linewidth=1)
# Inverting the yaxis
plt.gca().invert_yaxis()
# Move x-axis ticks to upper x-axis
plt.tick_params(axis='x', bottom=False, top=True,labeltop=True,labelbottom=False)
plt.ylabel('Simulation Time')
plt.title('Message Exchange Between Agents')
plt.tight_layout()
# plt.grid()
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=key, markersize=10, markerfacecolor=color) for key, color in color_map_legend.items()]
plt.legend(handles=legend_labels, title='',loc='lower center',bbox_to_anchor=(0.5, -0.12),ncol=len(color_map_legend)) # bbox_to_anchor=(-0.1, -0.1)
plt.tight_layout()
plt.savefig(f'messages_{name_case}.png')
plt.savefig(f'messages_{name_case}.svg')
# plt.show()
