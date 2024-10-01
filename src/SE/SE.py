# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:52:28 2022

@author: Monish Mukherjee and Fernando 
"""
import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2
from SE import WLS

import logging
log = logging.getLogger(__name__)

def get_voltages_state_estimation(Y_bus:pd.DataFrame,sensor:pd.DataFrame,print_results:bool=False,CB_error:bool=False,measurement_error:bool=False) -> tuple:
    """Perform state estimation to get the voltage states for a power distribution system network.

    This function estimates the voltage states (amplitude and angle) at various nodes of a power distribution system network using Weighted Least Squares (WLS) method. It processes sensor measurements, checks for bad data, and optionally prints the results.

    Args:
        Y_bus (pd.DataFrame): Admittance bus matrix of the power distribution system network.
        sensor (pd.DataFrame): DataFrame containing sensor measurements and their characteristics (i.e., type, location, variance, and measurement).
        print_results (bool, optional): If True, print detailed estimation results and bad data detection information. Defaults to False.
        CB_error (bool, optional): If True, introduces a error in circuit breaker measurements for testing purposes. Defaults to False.
        measurement_error (bool, optional): If True, introduces random measurement errors based on the variance of measurements. Defaults to True.

    Returns:
        tuple:
            - results_table (pd.DataFrame): DataFrame containing the estimated voltage amplitudes and angles at each node.
            - s_area (pd.DataFrame): DataFrame containing the processed sensor measurements and associated information.
            - bool: Boolean indicating whether bad data exists in the measurements (True if bad data exists, False otherwise).
            - s_u (np.ndarray): State vector obtained from the WLS estimation.
            - e_2 (np.ndarray): Residual vector from the WLS estimation.
    """  
    s_CB=sensor.loc[sensor['node1']!=sensor['node2']].copy(deep=True)
    # Fix flow of power direction
    s_CB.loc[s_CB['node1'].isin(Y_bus.columns),'measurement']=s_CB.loc[s_CB['node1'].isin(Y_bus.columns),'measurement']*-1.0

    s_area=sensor.loc[sensor['node1']==sensor['node2']].copy(deep=True)
    s_area.reset_index(drop=True,inplace=True)

    for index,row in s_CB.iterrows():
        if any([row['node1']==j for j in list(Y_bus.columns)]):
            row['node2']=row['node1']
        else:
            row['node1']=row['node2']
        if CB_error:
            row['measurement']=row['measurement']*-1 # grosse error for BC state test was -100
        temp=s_area.loc[(s_area['sensor_type']==row['sensor_type']) & \
                   (s_area['node1']==row['node1']),:].copy(deep=True)
        if len(temp) == 1:
            row['measurement']=row['measurement']+float(s_area.loc[(s_area['sensor_type']==row['sensor_type']) & \
                                                             (s_area['node1']==row['node1']),'measurement'])
            row['variance']=row['variance']+float(s_area.loc[(s_area['sensor_type']==row['sensor_type']) & \
                                                             (s_area['node1']==row['node1']),'variance'])
            s_area.loc[(s_area['sensor_type']==row['sensor_type']) & \
                       (s_area['node1']==row['node1']),:]  = list(row[row.keys()].values)
        else:
            s_area.loc[len(s_area.index)] = list(row[row.keys()].values)      
    #################### adding error #####################################
    if measurement_error:
        np.random.seed(0)
        s_area['measurement'] = s_area['measurement']  + np.random.normal(loc=0.0,scale=s_area['variance'])
    #################### Per Unit Voltage Angles Guess ####################
    node_names=list(Y_bus.columns)
    Theta_i_guess = np.zeros(len(node_names));
    for i in range(len(node_names)):
        if '.1' in node_names[i]:
            Theta_i_guess[i] = 0
        elif '.2' in node_names[i]:
            Theta_i_guess[i] = -2*np.pi/3
        elif '.3' in node_names[i]:
            Theta_i_guess[i] = 2*np.pi/3
    Theta_i = Theta_i_guess
    V_i_guess = Theta_i * 0.0 +1.0
    ####################### Dictionary corrections ########################
    dic_names_to_index=dict(zip(list(Y_bus.columns), list(Y_bus.index)))
    dic_sensor_type_to_index=dict(zip(['Vi','Pi','Qi'],[5,2,4]))
    ####################### Makes measurements and R matrix ###############
    z = s_area.copy(deep=True)
    z["node1"]=z["node1"].map(dic_names_to_index)

    z['node2']=0
    z["sensor_type"]=z["sensor_type"].map(dic_sensor_type_to_index)

    z = z.reset_index(drop=True)
    z['sensor_index']=list(z.index)
    z = z[['sensor_index','sensor_type','measurement','node1','node2','variance','sensor_name']]
    
    R_diag =  np.diag(z['variance'])
    
    iter_max = 25  # maximum number of iteration for WLS to converge
    threshold = 1e-7
    V_wls, iter_number, f_val, s_u, e_2 = WLS.state_estimation(Y_bus, \
                                z.drop(columns=['variance','sensor_name','area'],errors='ignore').to_numpy(), \
                                z['measurement'].to_numpy(), R_diag, iter_max, \
                                threshold, V_i_guess, Theta_i_guess)
    ######################### Bad Data Detection ##############################
    Voltage_SE = np.abs(V_wls);
    Theta_SE = (180/np.pi)*np.angle(V_wls);
    Theta_i  = (180/np.pi)*(Theta_i);
    Nm = len(z)                
    Ns = 2*len(Y_bus)-3;           
    k = Nm-Ns;                         
    a = 0.01;                # Specified probability     0.01               
    p = 1-a;                 # probability                   
    X2ka = chi2.ppf(p, df=k);     # Chi-square distribution         

    f = sum(f_val)
    s_area=s_area.assign(f_val=f_val)
    results_table = pd.DataFrame({'Node':list(Y_bus.columns),'Voltage_SE':Voltage_SE,'Theta_SE':Theta_SE})
    if print_results:
        print(f"No. of Measurments: {Nm} \nNo. of State Varriables: {Ns} \nNo. Degree of Freedom {k} \nNumber of iterations {iter_number}")
        print(f'The chi test comparison: {X2ka}')
        if f<X2ka:
            print('====================================')
            print(' No Bad Data exists in measurements')
            print('====================================')
        else:
            print('=======================================')
            print(' There exists Bad Data in measurements')
            print('=======================================')
            measurement_index = np.argmax(np.sqrt(f_val))
            print(s_area.iloc[measurement_index,:].to_markdown())
            print('------------------------')
    return results_table, s_area, f<X2ka, s_u, e_2

def split(a:list, n:int) -> list:
    """Split a list into `n` parts as evenly as possible.

    Args:
        a (list): The list to be split.
        n (int): The number of parts to split the list into.

    Returns:
        list of lists: A list containing `n` sublists, with each sublist containing a portion of the original list `a`.
    """
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)) 

def combine(a:list,n:int) -> list:
    """Combine a flat list into a list of sublists, where each sublist has a specified length `n`.

    Args:
        a (list): The list to be combined into sublists.
        n (int): The length of each sublist.

    Returns:
        list of lists: A list containing sublists of length `n` from the original list `a`.
    """
    lst=list(a)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

if __name__=="__main__":
    logging.basicConfig(filename="SE_gridappsd.log",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    logging.info("Start of file")
    
    path = "~/grid/dltdflisr/config/"
    sensor=pd.read_hdf(path+"sensors.h5")
    
    
    #### Testing all areas by time
    measurements=pd.read_hdf(path+"measurements.h5")
    test_false_gros_error=False
    testing_detection_non_gros_error=False
    CB_test = True
    test_detection_of_gross_error_voltage=False
    test_detection_of_gross_error_pi_CB=False
    test_detection_of_gross_error_qi_CB=False
    test_detection_of_gross_error_pi=False
    test_detection_of_gross_error_qi=False
    CB_error_multiple_at_once=False
    
    for aa in range(6,7):
        Y_bus=pd.read_hdf(path+"ybus_area.h5", key='area-'+str(aa))# working for all 
        sensor_a=sensor.loc[sensor['node1'].isin(list(Y_bus.columns)) | sensor['node2'].isin(list(Y_bus.columns))].copy(deep=True)
        
        logging.info(f"\n\n Start of area         {str(aa)} \n\n")
        
        if test_false_gros_error:
            for time in measurements.index:
                c_m=measurements.loc[time,:].copy(deep=True)
                c_m=pd.DataFrame({"sensor_name":c_m.index,"measurement":c_m}).reset_index(drop=True)
                sensor_a.drop(columns="measurement",inplace=True)
                sensor_a=sensor_a.merge(c_m,on='sensor_name')
    
                sensor_a.reset_index(drop=True,inplace=True)
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                
                # capacitor_bus=['BUS2038','BUS3079']
                # capacitor_bus=[nn+'.'+str(k) for nn in capacitor_bus for k in [1,2,3]]
                # s.loc[(s['sensor_type']=='Qi') & (s['node1'].isin(capacitor_bus)),'measurement']=0.00026253233563062713
            
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False,measurement_error=True)
                if not No_Bad_Data:
                    logging.info(f"Gross error detected but non is present {time}")
        
        for time in measurements.index:#[1:2]:
            c_m=measurements.loc[time,:].copy(deep=True)
            c_m=pd.DataFrame({"sensor_name":c_m.index,"measurement":c_m}).reset_index(drop=True)
            sensor_a.drop(columns="measurement",inplace=True)
            sensor_a=sensor_a.merge(c_m,on='sensor_name')

            sensor_a.reset_index(drop=True,inplace=True)
            
            if test_detection_of_gross_error_voltage:
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                index=s.loc[s['sensor_type']=='Vi'].index
                ss=s.copy(deep=True)
                for hhh in index:
                    s_m=s.loc[s.index.isin([hhh]),'measurement']
                    s_var=(s.loc[s.index.isin([hhh]),'variance'])**0.5*3
                    for _ in range(1000):
                        s.loc[s.index.isin([hhh]),'measurement']=s_m+s_var*_*0.1
                        results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                        # logging.info(No_Bad_Data)
                        if not No_Bad_Data:
                            logging.info(f"{hhh} Gross error detected vi expected error times {_*0.1}")
                            break
                    s=ss.copy(deep=True)
            
            if test_detection_of_gross_error_pi_CB:
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                index=s.loc[(s['sensor_type']=='Pi') & (s['measurement'].abs()>=1e-7)  & (s['node1']!=s['node2']) ].index
                ss=s.copy(deep=True)
                
                if CB_error_multiple_at_once:
                    index=combine(index,3)
                else:
                    index=combine(index,1)
                
                for hhh in index:
                    s_m=s.loc[s.index.isin(hhh),'measurement']
                    s_var=(s.loc[s.index.isin(hhh),'variance'])**0.5*3
                    for _ in range(1000):
                        s.loc[s.index.isin(hhh),'measurement']=s_m+s_var*_*0.5 #p_error*_
                        results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                        if not No_Bad_Data:
                            logging.info(f"{hhh} Gross error detected pi CB expected error times {_*0.5}")
                            break
                    s=ss.copy(deep=True)
            
            if test_detection_of_gross_error_pi:
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                index=s.loc[(s['sensor_type']=='Pi') & (s['measurement'].abs()>=1e-7)  & (s['node1']==s['node2']) ].index
                ss=s.copy(deep=True)
                
                if CB_error_multiple_at_once:
                    index=combine(index,3)
                else:
                    index=combine(index,1)
                
                for hhh in index:
                    s_m=s.loc[s.index.isin(hhh),'measurement']
                    s_var=(s.loc[s.index.isin(hhh),'variance'])**0.5*3
                    for _ in range(60,1000):#range(1000):
                        s.loc[s.index.isin(hhh),'measurement']=s_m+s_var*_*0.5 #p_error*_
                        results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                        if not No_Bad_Data:
                            logging.info(f"{hhh} Gross error detected pi expected error times {_*0.5}")
                            break
                    s=ss.copy(deep=True)
                
                    
            if test_detection_of_gross_error_qi_CB:
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                index=s.loc[(s['sensor_type']=='Qi') & (s['measurement'].abs()>=1e-7)  & (s['node1']!=s['node2']) ].index
                ss=s.copy(deep=True)
                
                if CB_error_multiple_at_once:
                    index=combine(index,3)
                else:
                    index=combine(index,1)
                
                for hhh in index:
                    s_m=s.loc[s.index.isin(hhh),'measurement']
                    s_var=(s.loc[s.index.isin(hhh),'variance'])**0.5*3
                    for _ in range(1000):
                        s.loc[s.index.isin(hhh),'measurement']=s_m+s_var*_*0.5 #p_error*_
                        results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                        if not No_Bad_Data:
                            logging.info(f"{hhh} Gross error detected qi CB expected error times {_*0.5}")
                            break
                    s=ss.copy(deep=True)
            
            if test_detection_of_gross_error_qi:
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                s.reset_index(drop=True,inplace=True)
                index=s.loc[(s['sensor_type']=='Qi') & (s['measurement'].abs()>=1e-7)  & (s['node1']==s['node2']) ].index
                ss=s.copy(deep=True)
                
                if CB_error_multiple_at_once:
                    index=combine(index,3)
                else:
                    index=combine(index,1)
                
                for hhh in index:
                    s_m=s.loc[s.index.isin(hhh),'measurement']
                    s_var=(s.loc[s.index.isin(hhh),'variance'])**0.5*3
                    for _ in range(50,1000):#range(1000):
                        s.loc[s.index.isin(hhh),'measurement']=s_m+s_var*_*0.5 #p_error*_
                        results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                        if not No_Bad_Data:
                            logging.info(f"{hhh} Gross error detected qi CB expected error times {_*0.5}")
                            break
                    s=ss.copy(deep=True)
            
            if testing_detection_non_gros_error:
                #### Testing for adding measurements
                temp_sense=sensor_a.loc[  (sensor_a['sensor_type'].isin(['Pi','Qi'])) | (sensor_a['sensor_name'].isin(list(sensor_a['sensor_name'][0:3]))) ].copy(deep=True)
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,temp_sense,print_results=False,CB_error=False)
                s_u_with_error=s_u
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,sensor_a,print_results=False,CB_error=False)
                if not(s_u < s_u_with_error):
                    logging.info(f"Problem in s_u detecting state; time  {str(time)};  area {str(aa)}")
                temp_sense=sensor_a.loc[  (sensor_a['sensor_type'].isin(['Pi','Qi'])) | (sensor_a['sensor_name'].isin(list(sensor_a['sensor_name'][0:3]))) ].copy(deep=True)
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,temp_sense,print_results=False,CB_error=False)
                s_u_with_error=s_u
                temp_sense=sensor_a.loc[  (sensor_a['sensor_type'].isin(['Pi','Qi'])) | (sensor_a['sensor_name'].isin(list(sensor_a['sensor_name'][0:6]))) ].copy(deep=True)
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,temp_sense,print_results=False,CB_error=False)
                if not(s_u < s_u_with_error):
                    logging.info(f"Problem in s_u detecting state; 0:3 vs 0:6; time  {str(time)};  area {str(aa)}")
                
            if CB_test:
                #### CB error test by changing state
                # Voltage only on CB nodes
                s=sensor_a.copy(deep=True)
                cbn=s.loc[s['node1']!=s['node2'],['node1','node2']]
                cbn=np.unique(cbn.to_numpy().flatten())
                cbn=cbn[pd.Series(cbn).isin(Y_bus.columns)]
                s.drop( s.loc[ (s['sensor_type']=='Vi') & (~s['node1'].isin(cbn))].index , inplace=True)
                
                results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,s,print_results=False,CB_error=False)
                f_val_no_error=s_area['f_val'].sum()
                
                cdns=[[cbn[i],cbn[i+1],cbn[i+2]] for i in np.arange(0,len(cbn),3) ]
                for CB in cdns:
                    ss=s.copy(deep=True)
                    if ss.loc[ (ss['sensor_type'].isin(['Pi','Qi'])) & ( (ss['node1'].isin(CB)) | (ss['node2'].isin(CB))  ) , 'measurement'].abs().sum() < 1e-7:
                        ss.loc[ (ss['sensor_type'].isin(['Pi','Qi'])) & ( (ss['node1'].isin(CB)) | (ss['node2'].isin(CB))  ) , 'measurement'] = -8.8e-06
                    else:
                        ss.loc[ (ss['sensor_type'].isin(['Pi','Qi'])) & ( (ss['node1'].isin(CB)) | (ss['node2'].isin(CB))  ) , 'measurement'] = 0.0
                    results_table,s_area,No_Bad_Data,s_u,e_2=get_voltages_state_estimation(Y_bus,ss,print_results=False,CB_error=False)
                    if s_area['f_val'].sum() < f_val_no_error:
                        logging.info(f"Problem in f_val detecting state of CB {str(CB)}; time  {str(time)};  area {str(aa)}; \n measurements original \n {str(s.loc[ (s['sensor_type'].isin(['Pi','Qi'])) & ( (s['node1'].isin(CB)) | (s['node2'].isin(CB))  ) , 'measurement'].values)} \n list of CBS {str(cdns)} ")
    
    logging.info("\n\n End loging")
    logging.shutdown()