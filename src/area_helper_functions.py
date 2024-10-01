import os
import pandas as pd
import numpy as np
import networkx as nx
from SE import SE

from typing import List, Dict, Union

import helper as HP

import logging
log = logging.getLogger(__name__)

file_location = os.path.dirname(os.path.abspath(__file__))
#### mrid of CB switch
def CB_switch_mrid(configuration: dict, meta_data: pd.DataFrame) -> None:
    """Associates circuit breaker (CB) switch mRID (measurement Resource ID) information with the possible connections in the area agent configuration.

    This function processes the connection data and the provided metadata to map the correct mRID to the relevant 
    connections between nodes. It filters metadata for specific conditions, creates a unique identifier based on 
    node names, and merges this data with the connection information in the configuration dictionary.

    Args:
        configuration (dict): A dictionary containing the area agent configuration. This function modifies the 
                              'connections' key under 'area_agent_configuration' to include mRID information.
                              The expected structure is:

                              * 'connections': A dictionary that includes:

                                - 'connections': DataFrame containing node connection information.
        meta_data (pd.DataFrame): A DataFrame containing metadata related to the circuit breakers, which must 
                                  include the following columns:

                                  - 'mRID': Measurement Resource ID.
                                  - 'f_node_name': From node name.
                                  - 't_node_name': To node name.
                                  - 'rc4_2021': Type of equipment (e.g., 'LoadBreakSwitch').
                                  - 'measurement_type': Type of measurement (e.g., 'Pos' for position).
                                  - 'phase': Phase of the measurement (e.g., 'A').

    Returns:
        None: The function modifies the `configuration` dictionary in place by adding a new key 
              'possible_connections_mrid' under 'connections' with the updated DataFrame that includes mRID information.
    """
    df=configuration['area_agent_configuration']['connections']['connections']
    df['node1']=df['node1'].str.split('.').str[0].str.lower()
    df['node2']=df['node2'].str.split('.').str[0].str.lower()
    df=df.drop_duplicates()
    df.reset_index(drop=True,inplace=True)
    df=df.copy()

    meta_data.loc[meta_data['measurement_type']=='Pos']
    dff=meta_data.loc[(meta_data['rc4_2021']=='LoadBreakSwitch') &
                  (meta_data['measurement_type']=='Pos') &
                  (meta_data['phase']=='A'),['mRID','f_node_name','t_node_name']]
    
    dff['set']=dff.apply(lambda row: '_'.join(sorted([row['f_node_name'], row['t_node_name']])), axis=1)
    df['set']=df.apply(lambda row: '_'.join(sorted([row['node1'], row['node2']])), axis=1)

    df=df.merge(dff[['mRID','set']],on='set',how='left')
    configuration['area_agent_configuration']['connections']['possible_connections_mrid']=df

#### get additional area specific data for area agent creation
def additional_area_data(configuration: dict) -> dict:
    """Processes and organizes additional area-specific data for creating area agents.

    This function reads various data files specified in the `configuration` dictionary, processes them, and 
    updates the dictionary with structured data for area agents. It handles sensor data, connection data, 
    Ybus matrices, and maximum load data, and organizes these into a format suitable for further use.

    Args:
        configuration (dict): A dictionary containing file paths and other configurations related to 
                              area agents. The dictionary keys must include:

                              - 'area_agent_configuration': A dictionary with keys:

                                - 'sensors': File path for sensor data (HDF5 format).
                                - 'connections': File path for connection data (HDF5 format).
                                - 'ybus': File path for Ybus data (HDF5 format).
                                - 'max_area_load': File path for maximum area load data (HDF5 format).

    Returns:
        dict: The updated `configuration` dictionary with the following additional keys:

              - 'connections': A dictionary with the following dataframes:

                * 'possible_connections': Unique connections between areas.
                * 'area_connections_base': Connections involving only areas.
                * 'substation_area_connection': Connections involving substations.
                * 'interconnections': List of unique node connections.
                * 'connections': Original connections data.
              - 'ybus_reduced': A dictionary with reduced Ybus matrices for each key.
              - 'max_area_load': DataFrame with aggregated maximum load data, including additional columns for sensor names and node information.

        ybus (dict): pd.Dataframe of each ybus area.
    """
    # sensor
    temp=configuration['area_agent_configuration']['sensors']
    configuration['area_agent_configuration']['sensors']=pd.read_hdf(temp)
    # connection
    temp=configuration['area_agent_configuration']['connections']
    df_a_c=pd.read_hdf(temp)
    area_connections_base=df_a_c.copy()
    substation_area_connection=df_a_c.copy()
    area_connections_base=area_connections_base.loc[area_connections_base['area1']>0]
    substation_area_connection=substation_area_connection.loc[substation_area_connection['area1']<0]
    interconnections=df_a_c[['node1','node2']].to_numpy().flatten()
    possible_connections=df_a_c[["area1","area2"]].drop_duplicates().reset_index(drop=True)
    configuration['area_agent_configuration']['connections'] = {'possible_connections':possible_connections,
                                                                'area_connections_base':area_connections_base,
                                                                'substation_area_connection':substation_area_connection,
                                                                'interconnections':interconnections,
                                                                'connections':df_a_c}
    # Ybus and Ybus reduction
    temp=configuration['area_agent_configuration']['ybus']
    with pd.HDFStore(temp, mode='r') as hdf:
        keys = hdf.keys()
    ybus={}
    for i in keys:
        ybus[i.replace('/','').replace('-','_')]=pd.read_hdf(temp,key=i)
    # get order reduction ybus
    ybus_reduced={}
    ybus_equivalent_load={}
    for key,value in ybus.items():
        df_reduce, area_nodes = create_eq_ybus_df(value,interconnections,int(key.split('_')[-1]) )
        ybus_reduced[key]=df_reduce
        ybus_equivalent_load[key]=area_nodes
    configuration['area_agent_configuration']["ybus_reduced"]=ybus_reduced

    # max load
    temp=configuration['area_agent_configuration']['max_area_load']
    max_area_load=pd.read_hdf(temp)
    max_area_load['phase']=max_area_load['node1'].str[-1:]
    max_area_load=max_area_load.groupby( by=['area','sensor_type','phase'] ).sum()
    max_area_load.reset_index(inplace=True)
    for key, value in ybus_equivalent_load.items():
        a=key[-1:]
        for v in value:
            p=v[-1:]
            max_area_load.loc[(max_area_load['area']==int(a)) & (max_area_load['phase']==p),['node1','node2']]=v
    max_area_load['sensor_name']=max_area_load['sensor_type']+'_area_'+max_area_load['node1']
    configuration['area_agent_configuration']['max_area_load']=max_area_load

    return ybus


def get_new_expected_CB(cb: str, areas_disconnected: list, possible_connections: pd.DataFrame) -> str:
    """Generates a new circuit breaker status string based on areas that are disconnected.

    This function updates the given circuit breaker status string by setting the status of circuit breakers 
    corresponding to disconnected areas to "0" (indicating open). It then returns the updated status string.

    Args:
        cb (str): A string representing the current circuit breaker status, where each character corresponds 
                  to the status of a specific breaker.
        areas_disconnected (list): A list of areas that are currently disconnected from the network.
        possible_connections (pd.DataFrame): A DataFrame containing the possible connections in the network, 
                                             with columns 'area1' and 'area2' representing connections between areas.

    Returns:
        str: The updated circuit breaker status string with breakers in disconnected areas set to "0".
    """
    cbl=[char for char in cb]
    cb_opened=possible_connections.loc[possible_connections['area1'].isin(areas_disconnected) | possible_connections['area2'].isin(areas_disconnected)].index
    for i in cb_opened:
        cbl[i]="0"
    temp=''.join(cbl)
    return temp

def identify_CB_status_is_radial(possible_connections: pd.DataFrame, cd_new_for_path: str) -> bool:
    """Determines if the circuit breaker status, represented by a given configuration string, is radial in respect to a substation.

    This function evaluates the connectivity of the network based on the circuit breaker status described in 
    `cd_new_for_path`. It checks if each area is connected to the substations in a way that would be classified 
    as radial, meaning each area is directly connected to a substation without alternative paths.

    Args:
        possible_connections (pd.DataFrame): A DataFrame containing the possible connections in the network, 
                                             with columns 'area1' and 'area2' representing connections between areas.
        cd_new_for_path (str): A string representing the circuit breaker status, where each character corresponds 
                               to the status of a specific breaker.

    Returns:
        bool: `True` if the network configuration is radial with respect to substations, `False` otherwise.
    """
    indices_of_ones = [i for i, char in enumerate(cd_new_for_path) if char == '1']
    s_state=possible_connections.loc[possible_connections.index.isin(indices_of_ones)]
    temp=s_state[['area1','area2']].copy()
    substations=temp[temp<0].stack().drop_duplicates().astype(int).tolist()
    areas=temp[temp>0].stack().drop_duplicates().astype(int).tolist()
    if len(substations)==0:
        radial = True
        return radial
    G = nx.from_pandas_edgelist(s_state, 'area1', 'area2')
    for a in areas:
        n=0
        for s in substations:
            temp=list(nx.all_simple_paths(G, source=a, target=s))
            if len(temp)>0:
                n=n+1
        if n > 1 :
            radial = False
            return radial        
    radial = True
    return radial

# Function to get all neighbors, including indirect neighbors
def get_all_neighbors(G: nx.Graph, area: int) -> List[int]:
    """Retrieves all neighbors of a given area in a network, including indirect neighbors.

    This function uses a breadth-first search approach to identify not only the direct neighbors but also all 
    indirectly connected nodes in the network. It explores the entire connected component of the specified area 
    in the graph.

    Args:
        G (nx.Graph): A NetworkX graph object representing the network, with nodes and edges.
        area (int): The node identifier for which neighbors are to be found.

    Returns:
        List[int]: A list of node identifiers representing all neighbors of the specified area, including indirect neighbors.
    """
    all_neighbors = set()
    queue = [area]

    while queue:
        current_node = queue.pop()
        neighbors = list(nx.all_neighbors(G, current_node))
        for neighbor in neighbors:
            if neighbor not in all_neighbors:
                all_neighbors.add(neighbor)
                queue.append(neighbor)

    return list(all_neighbors)

def get_area_connected_neighbors(possible_connections: pd.DataFrame,
                                  cd_new_for_path: str,
                                  path: List[int]) -> List[int]:
    """Retrieves the list of neighbors connected to a specified area based on updated circuit breaker status.

    This function identifies the neighbors of a given area within a network, considering the updated circuit breaker 
    status. It uses the provided connections data and the updated CB status to construct a graph and determine 
    all connected neighbors.

    Args:
        possible_connections (pd.DataFrame): A DataFrame containing possible connections between areas with attributes such as 'area1', 'area2', and status.
        cd_new_for_path (str): A string representing the updated circuit breaker status for the path.
        path (List[int]): A list of node identifiers representing the path. The first node in this list is used to determine the area for which neighbors are to be identified.

    Returns:
        List[int]: A list of node identifiers representing the neighbors of the specified area.
    """
    indices_of_ones = [i for i, char in enumerate(cd_new_for_path) if char == '1']
    s_state=possible_connections.loc[possible_connections.index.isin(indices_of_ones)]
    temp=s_state[['area1','area2']].copy()
    substations=temp[temp<0].stack().drop_duplicates().astype(int).tolist()
    areas=temp[temp>0].stack().drop_duplicates().astype(int).tolist()
    G = nx.from_pandas_edgelist(s_state, 'area1', 'area2')
    area=path[0]
    neighbors_list = list(nx.all_neighbors(G,area))
    neighbors_list = get_all_neighbors(G,area)
    return neighbors_list


# Identify CB change for a path 
def identify_new_CB(system_status: str,
                    possible_connections: pd.DataFrame,
                    path: List[int]) -> Dict:
    """Identifies circuit breaker (CB) changes for a given path based on the system status and possible connections.

    This function determines which circuit breakers need to be changed (opened or closed) to accommodate a given 
    path, based on the current system status and the status of possible connections. It returns a dictionary containing
    details about the path, circuit breaker changes, and affected areas.

    Args:
        system_status (str): A string representing the current system status, including circuit breaker states.
        possible_connections (pd.DataFrame): A DataFrame containing possible connections between areas, with attributes such as status.
        path (List[int]): A list of node identifiers representing the path for which CB changes are to be identified.

    Returns:
        Dict[str, bool, str, list, list]:
            - "path": The given path as a list of node identifiers.
            - "substation_radial": A boolean indicating if the CB status is radial.
            - "cb_path": A string representing the updated CB status for the path.
            - "cbs_to_close": A list of indices of circuit breakers that need to be closed.
            - "areas_involved": A dictionary of areas involved in the path and their connected neighbors.
    """
    # path=all_paths['-2'][0]
    # cb=prior_system_status.split("_")[-1]
    # cb_new=get_new_expected_CB(cb,areas_disconnected,possible_connections)
    cb_new=system_status.split("_")[-1]
    indices_of_ones = [i for i, char in enumerate(cb_new) if char == '1']
    possible_connections['status']=int(0)
    possible_connections.loc[possible_connections.index.isin(indices_of_ones),["status"]]=int(1)

    to_close=[]
    for i in range(len(path)):
        temp=possible_connections.loc[possible_connections['area1'].isin(path[i:i+2]) & possible_connections['area2'].isin(path[i:i+2]) & (possible_connections['status']==0)].index
        if len(temp)>0:
            to_close.append(temp[0])
    
    cbl=[char for char in cb_new]
    for t in to_close:
        cbl[t]="1"
    cd_new_for_path=''.join(cbl)
    return {"path":path,"substation_radial":identify_CB_status_is_radial(possible_connections,cd_new_for_path),
            "cb_path":cd_new_for_path,"cbs_to_close":to_close,"areas_involved":get_area_connected_neighbors(possible_connections,cd_new_for_path,path)}
    
# Calculate path costs
def calculate_path_cost(graph: nx.Graph, path: List[int]) -> float:
    """Calculates the total cost of a path in a graph.

    This function computes the cost of traversing a given path in the provided graph. It sums the costs of the edges 
    that make up the path. The graph is expected to have edge attributes named 'Cost' which represent the cost of each edge.

    Args:
        graph (nx.Graph): A NetworkX graph object containing edge attributes 'Cost'.
        path (List[int]): A list of node identifiers representing the path in the graph.

    Returns:
        float: The total cost of traversing the given path.
    """
    cost = sum(graph[path[i]][path[i+1]]['Cost'] for i in range(len(path) - 1))
    return cost

def reconnection_paths(prior_system_status: str,
                       areas_disconnected: List[int],
                       possible_connections: pd.DataFrame,
                       area_number: int) -> Dict:
    """Computes potential reconnection paths and their associated costs based on the current system status.

    This function analyzes reconnection paths for a specified area based on the prior system status, disconnected 
    areas, and possible connections. It updates the costs of connections, builds a graph representation, and computes 
    all simple paths between the area and potential reconnection targets. It also calculates the cost of each path.

    Args:
        prior_system_status (str): A string representing the prior system status, including circuit breaker states.
        areas_disconnected (List[int]): A list of area numbers that are currently disconnected.
        possible_connections (pd.DataFrame): A DataFrame containing possible connections between areas, with 'area1', 'area2', and edge attributes.
        area_number (int): The area number for which reconnection paths are to be evaluated.

    Returns:
        Dict: A dictionary where:
            - Keys are strings representing targets and their path costs (e.g., 'target' and 'target_cost').
            - Values are lists of simple paths or path costs, depending on the key.
    """
    not_available_areas=[i for i in areas_disconnected if i!=area_number]
    
    cb=prior_system_status.split("_")[-1]
    cb_new=get_new_expected_CB(cb,areas_disconnected,possible_connections) 
    indices_of_ones = [i for i, char in enumerate(cb_new) if char == '1']
    possible_connections['Cost']=100.0
    possible_connections.loc[possible_connections.index.isin(indices_of_ones),["Cost"]]=0.0
    s_state=possible_connections.loc[~possible_connections['area1'].isin(not_available_areas) & ~possible_connections['area2'].isin(not_available_areas)]

    G = nx.from_pandas_edgelist(s_state, 'area1', 'area2',edge_attr='Cost')
    target=[int(i) for i in s_state.min(axis=1) if i<0]
    all_paths={}
    for i in target:
        temp = list(nx.all_simple_paths(G, source=area_number, target=i))
        if len(temp) > 0:
            all_paths[str(i)]=temp
            cost=[]
            for j in temp:
                cost.append(calculate_path_cost(G,j))
            all_paths[str(i)+"_cost"]=cost
    return all_paths

def area_at_fault(prior_system_status: str,
                   areas_disconnected: List[int],
                   possible_connections: pd.DataFrame,
                   area_number: int) -> Union[bool, str]:
    """Determines if an area is at fault based on system status, disconnected areas, and possible connections.

    This function evaluates whether a specific area (identified by `area_number`) is at fault. It checks the number 
    of disconnected areas, validates the provided data, and analyzes the connections to determine if the area 
    is at fault. The analysis uses graph-based path finding to identify connections between the area and the disconnected 
    nodes.

    Args:
        prior_system_status (str): A string representing the prior system status, where the last part indicates circuit breaker status.
        areas_disconnected (List[int]): A list of area numbers that are currently disconnected.
        possible_connections (pd.DataFrame): A DataFrame containing possible connections between areas, with 'area1' and 'area2' columns.
        area_number (int): The area number to be checked for fault status.

    Returns:
        Union[bool, str]: Returns `True` if the area is at fault, `False` if it is not, or an error message string if there are issues with the input data.
    """
    if len(areas_disconnected)==1:
        return True
    cb=prior_system_status.split("_")[-1]
    if len(cb) != len(possible_connections):
        return f"Error on the provided input data 1. area_number {area_number}. areas_disconnected {areas_disconnected}"
    if area_number not in areas_disconnected:
        return f"Error on the provided input data 2. area_number {area_number}. areas_disconnected {areas_disconnected}"
    if (area_number > possible_connections.max().max()) or (area_number < possible_connections.min().min()):
        return f"Error on the provided input data 3. area_number {area_number}. areas_disconnected {areas_disconnected}"
    indices_of_ones = [i for i, char in enumerate(cb) if char == '1']
    s_state=possible_connections[possible_connections.index.isin(indices_of_ones)]
    # s_state.loc[s_state['area1'].isin(areas_disconnected) | s_state['area2'].isin(areas_disconnected)]
    G = nx.from_pandas_edgelist(s_state, 'area1', 'area2')
    target=[int(i) for i in s_state.min(axis=1) if i<0]
    all_paths={}
    for i in target:
        temp = list(nx.all_simple_paths(G, source=area_number, target=i))
        if len(temp) > 0:
            all_paths[str(i)]=temp
    at_fault_analyses=[]
    for substation,paths in all_paths.items():
        for i in paths:
            intersection = list(set(i) & set(areas_disconnected))
            if len(intersection)>1 and len(i)>2:
                at_fault_analyses.append(False)
            else:
                at_fault_analyses.append(True)
    if all(at_fault_analyses):
        return True
    else:
        return False

def area_disconnected(df:pd.DataFrame) -> bool:
    """ Evaluate if area disconnected based on the sensor for that time dataframe 

    Args:
        df (dataframe): sensors for that time

    Returns:
        disconnected (boolean): is or is not disconnected 
    """
    df=df.loc[df['sensor_type']=="Vi"]
    mean_vi=df['measurement'].mean()
    if mean_vi < 0.1:
        disconnected=True
    else:
        disconnected=False
    return disconnected


def create_equivalent_ybus(ybus:np.ndarray, buses_to_keep:np.ndarray) -> np.ndarray:
    r"""Creates an equivalent ybus matrix by replacing the specified buses with equivalent admittances.

    The relationship between current and voltage is given by:

    $$ I = Y_{bus} V $$
    
    Where:

    $$\left[\begin{array}{c}I_{Study} \\ I_{External}\end{array}\right]=\left[\begin{array}{ll}Y_{S,S} & Y_{S,E} \\ Y_{E,S} & Y_{E,E}\end{array}\right] \left[\begin{array}{c}V_{Study} \\ V_{External}\end{array}\right]$$

    And:
    
    $$I_{study}=(Y_{s,s}-Y_{s,e}Y_{e,e}^{-1}Y_{e,s})V_{study}+Y_{s,e}Y_{e,e}^{-1}I_{external}$$

    Args:
        ybus (np.ndarray): The original ybus matrix.
        buses_to_keep (np.ndarray): Array of indices of the buses to keep.

    Returns:
        np.ndarray: The equivalent ybus matrix.
    """
    # TODO: requires modification/improvement the focus of this effort is not on model reduction and this approach has considerable limitations.
    # Extract the submatrices
    y_ss = ybus[np.ix_(buses_to_keep, buses_to_keep)]
    y_se = ybus[np.ix_(buses_to_keep, ~np.isin(np.arange(ybus.shape[0]), buses_to_keep))]
    y_ee = ybus[np.ix_(~np.isin(np.arange(ybus.shape[0]), buses_to_keep), ~np.isin(np.arange(ybus.shape[0]), buses_to_keep))]
    y_es = ybus[np.ix_(~np.isin(np.arange(ybus.shape[0]), buses_to_keep), buses_to_keep)]
    # Compute the equivalent admittances
    y_ee_inv = np.linalg.inv(y_ee)
    y_eq_ss = y_ss - y_se @ y_ee_inv @ y_es
    return y_eq_ss

def create_eq_ybus_df(df:pd.DataFrame,interconnections:pd.DataFrame,area:int):
    """Creates an equivalent Y-bus matrix DataFrame by reducing the original DataFrame based on specified criteria.

    This function processes the input DataFrame `df` to generate a reduced version of the Y-bus matrix, focusing on 
    specific nodes derived from the intersection with `interconnections` and additional area-based nodes. It returns 
    both the reduced Y-bus DataFrame and a list of area-specific nodes.

    Args:
        df (pd.DataFrame): The original DataFrame representing the Y-bus matrix.
        interconnections (pd.DataFrame): DataFrame containing interconnection nodes to be preserved.
        area (int): An integer representing the area to be considered for node filtering.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            - pd.DataFrame: The reduced Y-bus matrix DataFrame with only the relevant nodes.
            - list[str]: A list of area-specific node names used in the reduction.
    """
    dfnp=df.to_numpy()
    columns=df.columns
    intersec=np.intersect1d(columns,interconnections)
    notOFinterest=pd.Series(columns)
    notOFinterest=notOFinterest[~notOFinterest.isin(intersec)]
    notOFinterest=notOFinterest.str[:-2]
    notOFinterest=notOFinterest.value_counts()
    notOFinterest=notOFinterest[notOFinterest>=3]
    # i=int(len(notOFinterest)*0.62)
    # i=int(len(notOFinterest)*0.98)
    i=int(len(notOFinterest)*0.5)

    notOFinterest=notOFinterest.reset_index()
    area_bus=notOFinterest.loc[i,'node']
    area_nodes=[area_bus+'.'+str(i) for i in range(1,4)]

    node_name_keep=np.append(intersec, area_nodes)
    node_index_keep=list()
    for i in node_name_keep:
        node_index_keep.append( np.where(columns == i)[0][0] )
    node_index_keep=np.array(node_index_keep)
    
    node_index_keep=np.sort(node_index_keep)
    
    df_reduce = create_equivalent_ybus(dfnp,node_index_keep)
    df_reduce = pd.DataFrame(df_reduce,columns=columns[node_index_keep])
    
    return df_reduce, area_nodes

################################################################################################################## SE portion
#### changing CB state
def get_results_model_reduction(Y_BUS: pd.DataFrame, area_connections: pd.DataFrame, sensor: pd.DataFrame, 
                                substation_area_connection: pd.DataFrame, sensors_area_max_load: pd.DataFrame) -> dict:
    """Prepares ybus model to execute the state estimation with model reduction.

    Performs state estimation (SE) on a power system model by modifying the Y-bus matrix and incorporating sensor data. 
    The function processes connections and sensor data to simulate circuit breaker (CB) state changes and determines 
    if a connection is feasible based on voltage magnitude thresholds.

    Args:
        Y_BUS (pd.DataFrame): The Y-bus admittance matrix of the system, which is modified in-place to reflect the 
                              current CB state. Rows and columns represent the nodes in the system.
        area_connections (pd.DataFrame): DataFrame containing node-to-node connections within the area. 
                                         Columns expected: ['node1', 'node2'].
        sensor (pd.DataFrame): DataFrame with sensor data for state estimation. This is concatenated with 
                               additional virtual sensors and reference voltages within the function.
        substation_area_connection (pd.DataFrame): DataFrame containing the connection information for substations. 
                                                   Used to define reference voltages.
        sensors_area_max_load (pd.DataFrame): DataFrame containing the maximum load sensor data for the area. 
                                              Expected columns include ['node1', 'node2', 'sensor_type', 'sensor_name'].

    Returns:
        dict: A dictionary containing:
              - 'V magnitude maximum' (float): The maximum voltage magnitude resulting from state estimation.
              - 'V magnitude minimum' (float): The minimum voltage magnitude resulting from state estimation.
              - 'connection_possible' (bool): A flag indicating whether the connection is feasible based on voltage limits.
    """
    #### Make Ybus
    Y_cb = 10000000 + 0j
    Y_BUS.fillna(0+0j,inplace=True)
    Y_BUS.index=Y_BUS.columns
    for i,j in zip(area_connections['node1'],area_connections['node2']):
        # print(i)
        # print(j)
        Y_BUS.loc[[i],[i]] += Y_cb
        Y_BUS.loc[[j],[j]] += Y_cb
        Y_BUS.loc[[i],[j]] -= Y_cb
        Y_BUS.loc[[j],[i]] -= Y_cb
    
    #### Process sensors
        # add virtual 0 sensors
    cc=np.array(Y_BUS.columns)
    cc=np.setdiff1d(cc,np.array(sensors_area_max_load['node2']))
    ccP=pd.DataFrame({'node1':cc,'node2':cc})
    ccQ=pd.DataFrame({'node1':cc,'node2':cc})
    ccP['sensor_type']='Pi'
    ccQ['sensor_type']='Qi'
    ccP['sensor_name']='Pi_'+cc
    ccQ['sensor_name']='Qi_'+cc
    cc=pd.concat([ccP,ccQ])
    cc['measurement']=0.0
    cc['variance']=1e-20
        # add reference voltage
    reference_dic=[]
    for i in substation_area_connection.loc[substation_area_connection['node2'].isin(Y_BUS.columns),'node2'].to_list():
        reference_dic.append({'sensor_name':'ref_'+i,'sensor_type':'Vi','node1':i,
                        'node2':i,'measurement':1.0,'variance':1e-20})
    df_v_ref=pd.DataFrame( reference_dic )
        # remove substation from cc
    cc=cc.loc[~(cc['node2'].isin(df_v_ref['node1']))]

    #### sensor for SE
    sensor=pd.concat([df_v_ref,sensors_area_max_load,cc])

    #### run SE
    results_table,s_area,No_Bad_Data,s_u,e_2=SE.get_voltages_state_estimation(Y_BUS.reset_index(drop=True),sensor,print_results=False,measurement_error=False)
    connection_possible=True
    if (results_table['Voltage_SE'].max() > 1.05 ) or (results_table['Voltage_SE'].min() < 0.095):
        connection_possible=False
    return {'V magnitude maximum':results_table['Voltage_SE'].max(),'V magnitude minimum':results_table['Voltage_SE'].min(),'connection_possible':connection_possible}#,'results':results_table}



if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    path=path+'/../config/'
    sensor=pd.read_hdf(path+"sensors.h5")
    x=area_disconnected(sensor)
    print(x)
    df_a_c=pd.read_hdf(path+"area_connections.h5")
    possible_connections=df_a_c[["area1","area2"]].drop_duplicates().reset_index(drop=True)
    prior_system_status="CB_state_101110110"
    areas_disconnected=[5,6]
    area_number=6
    x=area_at_fault(prior_system_status,areas_disconnected,possible_connections,area_number)
    print(x)
    area_number=5
    x=area_at_fault(prior_system_status,areas_disconnected,possible_connections,area_number)
    print(x)

    #### Testing the read
    import pandas as pd
    import os
    # read TOML      os.path.expanduser('~/grid/dltdflisr/config/input_config.toml')
    configuration = HP.load_toml(f'{file_location}/../config/input_config.toml')
    # configure logging
    HP.configure_logging(**configuration['logging_configuration'])
    # create sport environmental variables
    HP.os_variables(configuration['system_variables'])
    # 
    ybus = additional_area_data(configuration)

    base_path = "~/grid/dltdflisr"
    base_path = os.path.expanduser(base_path)
    meta=f"{base_path}/outputs/record_00.h5"
    meta_data=pd.read_hdf(meta,key='/meta_data')

    df=configuration['area_agent_configuration']['connections']['connections']
    df['node1']=df['node1'].str.split('.').str[0].str.lower()
    df['node2']=df['node2'].str.split('.').str[0].str.lower()
    df=df.drop_duplicates()

    meta_data.loc[meta_data['measurement_type']=='Pos']
    dff=meta_data.loc[(meta_data['rc4_2021']=='LoadBreakSwitch') &
                  (meta_data['measurement_type']=='Pos') &
                  (meta_data['phase']=='A'),['mRID','f_node_name','t_node_name']]
    
    dff['set']=dff.apply(lambda row: '_'.join(sorted([row['f_node_name'], row['t_node_name']])), axis=1)
    df['set']=df.apply(lambda row: '_'.join(sorted([row['node1'], row['node2']])), axis=1)

    df=df.merge(dff[['mRID','set']],on='set',how='left')
    configuration['area_agent_configuration']['connections']['possible_connections_mrid']=df

    #### testing reconnection path
    prior_system_status='CB_state_101110110'
    areas_disconnected=[5, 6]
    possible_connections=df_a_c[["area1","area2"]].drop_duplicates().reset_index(drop=True)
    area_number=6
    oi=reconnection_paths(prior_system_status,areas_disconnected,possible_connections,area_number)
    ####
    # Identify CB change for a path 
    # system_status
    # possible_connections
    # path
    # ola=identify_new_CB()
