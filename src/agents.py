import os
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import traceback
from typing import Dict
from typing import OrderedDict
from typing import Tuple
import json
import logging
import numpy as np
import math
import gridappsd.topics as t
from gridappsd import DifferenceBuilder
from gridappsd.simulation import Simulation
from gridappsd.field_interface.context import LocalContext
from gridappsd.field_interface.interfaces import MessageBusDefinition
from gridappsd.field_interface.agents import CoordinatingAgent
from gridappsd.field_interface.agents import FeederAgent
from gridappsd.field_interface.agents import SwitchAreaAgent
from gridappsd.field_interface.agents import SecondaryAreaAgent
from cimgraph.models.distributed_area import DistributedArea
from cimgraph.data_profile.rc4_2021 import EnergyConsumer
from cimgraph.data_profile.rc4_2021 import LoadBreakSwitch
from cimgraph.data_profile.rc4_2021 import ACLineSegment
from cimgraph.data_profile.rc4_2021 import LinearShuntCompensator
from cimgraph import utils

import area_helper_functions as PSA
from io import StringIO
from MO import monitoring as MO
from MO import trekker as MT
import pickle
import helper as HP
import h5py
import random
random.seed(10) # any seed will work is just to make the behavior reproducible 

log = logging.getLogger(__name__)

#### constant variables
keys_to_check = ["command","from","to","content","simulation_time","type"]

class HardCodedTimeSleep:
    """A hard-coded configuration class with integer variables.

    The added time sleep on the code is needed to enable time for the new information to arrive from the simulation to the agents or for a command to take effect in the simulation. Or for coordination withing the agents.

    Attributes:
        interval_1 (int): Time interval in seconds. Method time_control after initialization to await distribution power system simulation and receive of messages by the actors.
        interval_2 (int): Time interval in seconds. Method update_cb_status the simulation needs time to implement a change.
        interval_3 (int): Time interval in seconds. Method on_upstream_message additional time for agents to finish current processing before proceeding.
    """
    interval_1 = 5 # time.sleep(5) time_control after initialization to await distribution power system simulation and receive of messages by the actors.
    interval_2 = 3 # time.sleep(3) update_cb_status the simulation needs time to implement a change.
    interval_3 = 1 # time.sleep(1) on_upstream_message additional time for agents to finish current processing before proceeding.

#### agent common functions
def make_empty_message_dict() -> Dict[str, str]:
    """Creates an empty message dictionary with predefined keys.

    This function initializes a dictionary where each key from a predefined list of keys is set to an empty string. 
    The list of keys to include is specified by `keys_to_check`, which should be defined in the surrounding scope.

    Returns:
        Dict[str, str]: A dictionary with keys from `keys_to_check`, each mapped to an empty string.
    """
    message={}
    for i in keys_to_check:
        message[i]=""
    return message

def test_df_are_equal(df:pd.DataFrame,dff:pd.DataFrame) -> bool:
    return (df.apply(lambda col: col.map(str)) == dff.apply(lambda col: col.map(str))).all().all()


#### be able to send configurations to output file
def is_picklable(obj) -> bool:
    """Checks if an object can be pickled.

    This function attempts to pickle the provided object. If the object can be pickled without raising exceptions,
    it returns `True`. Otherwise, it returns `False` if the object raises a `PicklingError`, `TypeError`, or 
    `AttributeError`.

    Args:
        obj: The object to check for picklability.

    Returns:
        bool: `True` if the object can be pickled, `False` otherwise.
    """
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False

def filter_picklable(data) -> 'Any':
    """Filters a data structure to retain only picklable elements.

    This function recursively traverses a data structure (which may include dictionaries, lists, and tuples) and 
    retains only the elements that are picklable. Elements that cannot be pickled are excluded from the resulting 
    structure. The function supports dictionaries, lists, and tuples as well as nested structures.

    Args:
        data: A data structure (dictionary, list, or tuple) to filter.

    Returns:
        Any: A filtered version of the input data structure, containing only picklable elements. The type of the
             returned structure matches the type of the input (dictionary, list, or tuple).
    """
    # Handle dictionaries
    if isinstance(data, dict):
        return {k: filter_picklable(v) for k, v in data.items() if is_picklable(v) or isinstance(v, (dict, list, tuple))}
    # Handle lists
    elif isinstance(data, list):
        return [filter_picklable(item) for item in data if is_picklable(item) or isinstance(item, (dict, list, tuple))]
    # Handle tuples
    elif isinstance(data, tuple):
        return tuple(filter_picklable(item) for item in data if is_picklable(item) or isinstance(item, (dict, list, tuple)))
    else:
        return data if is_picklable(data) else None

def measurement_description(area:DistributedArea,agentID:str) -> pd.DataFrame:
    """Provides a description of the mRID measurements.

    Provides in a dataframe format having every row representing a given measurement mRID the: 
        - FROM and TO node name and mRID;
        - rc4_2021 name;
        - phase specification (i.e., A, B, and C);
        - measurement_mRID;
        - Type of measurement;
    
    Args:
        area (DistributedArea): processed cim module for the area of interest
        agentID (str): agent ID for error logging

    Returns:
        pd.DataFrame: Descriptions of the measurement mRID
    """    
    measurements=list()
    def call(elements,name:str,terminals:int) -> None:
        for load in elements:
            if len(load.Terminals) == terminals:
                for m in load.Measurements:
                    dic=dict()
                    dic['mRID'] = load.mRID
                    dic['name'] = load.name
                    dic['rc4_2021']=name
                    dic['f_node_mRID']=load.Terminals[0].ConnectivityNode.mRID
                    dic['f_node_name']=load.Terminals[0].ConnectivityNode.name
                    if terminals == 2:
                        dic['t_node_mRID']=load.Terminals[1].ConnectivityNode.mRID
                        dic['t_node_name']=load.Terminals[1].ConnectivityNode.name
                    else:
                        dic['t_node_mRID']=None
                        dic['t_node_name']=None
                    dic['phase'] = m.phases[-1] # Getint the last character of the string 
                    dic['measurement_mRID'] = m.mRID
                    dic['measurement_type'] = m.measurementType
                    measurements.append(dic)
            else:
                log.error(f"measurement_description {name} agentID: {agentID}")
    call(area.graph[area.cim.EnergyConsumer].values(),'EnergyConsumer',1)
    call(area.graph[area.cim.LoadBreakSwitch].values(),'LoadBreakSwitch',2)
    call(area.graph[area.cim.ACLineSegment].values(),'ACLineSegment',2)
    call(area.graph[area.cim.LinearShuntCompensator].values(),'LinearShuntCompensator',1)
    df=pd.DataFrame(measurements)
    return df

class SampleFeederAgent(FeederAgent):
    """A class representing a feeder agent in the distribution power system simulation.

    This class extends the `FeederAgent` base class and includes methods for managing circuit breakers, processing 
    messages, controlling simulation time, and handling sensor measurements. It interacts with other agents, updates 
    internal mappings, and saves collected data to a file at the end of the simulation.
    """
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict,
                 simulation_id: str,
                 simulation,
                 agent_config: Dict) -> None:
        """Initializes the SampleFeederAgent with the given parameters and registers the agent.

        Args:
            upstream (MessageBusDefinition): The message bus definition for upstream communication.
            downstream (MessageBusDefinition): The message bus definition for downstream communication.
            config (Dict): A dictionary containing configuration parameters.
            area (Dict): A dictionary representing the area related to the agent.
            simulation_id (str): An identifier for the simulation.
            simulation (object): The simulation instance.
            agent_config (Dict): Additional configuration specific to the agent.
        """        
        self._latch = False
        self._message_next_seconds=None
        self._time_control=None
        self._time_granted_smaller_than_a_second=timedelta(seconds=0)
        self._timestamp_set_on_centralized_measurement=set()
        self._feeder_simulation_time=None
        self._timestamp_set_on_measurement=set()
        super().__init__(upstream, downstream, config, area, simulation_id)

        utils.get_all_data(self.feeder_area)
        self.measurements_info=measurement_description(self.feeder_area,self.agent_id)

        PSA.CB_switch_mrid(agent_config,self.measurements_info)

        self.new_cb_s_state=None
        self._perform_self_time_control_call = False
        self.system_status=agent_config['feeder_agent_configuration']['initial_CB_state']

        self._compleat_simulation_configuration=agent_config
        self.agent_config=agent_config['feeder_agent_configuration']
        self.measurement_log=None
        self.agent_dic={}
        self.message_record=[] # structure "command" "from" "to" "content" "simulation_time" "type"
        log.debug("Spawning Feeder Agent")
        if simulation is not None:
            self.simulation = simulation
            self.simulation_id = simulation.simulation.simulation_id
            self.simulation.simulation.add_onmeasurement_callback(self.on_centralized_measurement)

    def monitoring_to_file(self) -> None:
        """Saves the collected information to a file at the end of the simulation.

        This method writes collected data to an HDF5 file that was created at the beginning of the simulation. It performs
        the following operations:
        - Saves temporal sensor measurement data in the HDF5 file, organized with metadata for sensors (e.g., a1, a2, a3, ... an).
        - Creates a DataFrame from the `message_record` and saves it to the HDF5 file under the key 'message_record'.
        - Pickles an overall configuration dictionary and stores it in the HDF5 file under the key 'configuration_toml'.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It performs file operations to save collected data and configuration.
        """
        self.measurement_log.close_data_processor()
        df=pd.DataFrame(self.message_record)
        df.to_hdf(self.measurement_log.file_name, key='/message_record', mode='a', index=False, complevel=self.measurement_log.complevel)

        # Recursively filter out non-picklable items like 'DynamicInlineTableDict'
        filtered_dict = filter_picklable(self._compleat_simulation_configuration)
        # Pickle the filtered dictionary
        data_serialized=pickle.dumps(filtered_dict)
        with h5py.File(self.measurement_log.file_name, 'a') as h5file:
            h5file.create_dataset('configuration_toml', data=np.void(data_serialized))
    
    def compute_time_for_next_event(self) -> None:
        """Computes and adjusts the time step for the next event in the simulation.

        This method calculates the appropriate time step for the next event based on the current simulation messages. It 
        ensures that the simulation progresses efficiently by adjusting the time step to be as large or small as needed.
        """
        df=pd.DataFrame(self.message_record)
        self._message_next_seconds = None
        if len(df)>0:
            df=df.loc[(df['message_received']=="True") & (df['simulation_time_received']>self._feeder_simulation_time)]
            if len(df)>0:
                min_time=df['simulation_time_received'].min()
                time_difference = min_time - self._feeder_simulation_time
                self._message_next_seconds = time_difference #.total_seconds()
            else:
                self._message_next_seconds = None
    
    def time_control(self) -> None:
        """Manages the simulation time control, adjusting the time steps for the simulation as needed.

        Initially, this method runs the simulation in 1-second increments while attempting to discover the simulation 
        time via asynchronous messages. Once the simulation time is known, it computes the time for the next event and 
        adjusts the simulation time steps accordingly. The method ensures that the simulation time steps are appropriately 
        large or small, while handling a minimum time step of 1 second for the distribution power simulation.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It performs internal adjustments to manage the simulation time 
                and time steps.
        """
        #### get second precision on the simulation control
        if (self._feeder_simulation_time is None) and (self._time_control is None):
            if len(self._timestamp_set_on_centralized_measurement) != 0:
                self._time_control = max(self._timestamp_set_on_centralized_measurement) # Get the maximum datetime
            self.simulation.simulation.resume_pause_at(1)
        elif (self._feeder_simulation_time is None):
            temp=max(self._timestamp_set_on_centralized_measurement)
            if temp > self._time_control:
                self._feeder_simulation_time = temp
            else:
                self.simulation.simulation.resume_pause_at(1)
        #### Start normal time control operation
        else:
            self.compute_time_for_next_event()
            maximum_step=timedelta(seconds=self.agent_config['temporal']['maximum_delta'])
            if self._message_next_seconds is not None:
                maximum_step = min([maximum_step,self._message_next_seconds])
            self._feeder_simulation_time = self._feeder_simulation_time + maximum_step
            temp=self._time_granted_smaller_than_a_second + maximum_step
            if temp.seconds >= 1:
                self._time_granted_smaller_than_a_second = timedelta(microseconds=temp.microseconds)
                self.simulation.simulation.resume_pause_at(temp.seconds)
                time.sleep(HardCodedTimeSleep.interval_1)
            #### Tell switch area agents that the new simulation time
            message=make_empty_message_dict()
            message['command']='change_s_time'
            message['content']={'time':HP.comp_time(self._feeder_simulation_time)}
            message['from']=self.agent_id
            message['type']=self._perform_self_time_control_call
            self.publish_downstream(message)


    def on_centralized_measurement(self, sim: Simulation, timestamp: str, measurements: Dict[str, Dict]) -> None:
        """Handles centralized sensor measurements for logging and simulation time management.

        This method receives and processes sensor measurements from a centralized source, regardless of their location. 
        It updates the sensor logs and manages simulation time based on the received data. This method is called by the 
        simulation system whenever new measurements become available.

        Args:
            sim (Simulation): The simulation instance that is providing the sensor measurements and simulation time.
            timestamp (str): A string representing the time at which the measurements were taken.
            measurements (Dict[str, Dict]): A dictionary where keys are sensor IDs and values are dictionaries containing 
                                            measurement data. Each inner dictionary holds the specific measurement details 
                                            for the respective sensor.

        Returns:
            None: This method does not return any values. It performs internal updates and logging based on the received 
                measurements and simulation time.
        """
        m_key='start_monitoring'
        m_value='value'
        m_mrid='measurement_mrid'
        if (m_key in self.agent_config) and (int(timestamp) >= int(self.agent_config[m_key])) and (self.measurement_log is None):
            log.info("Making the monitoring")
            temp=self.agent_config['monitoring']
            temp['meta_data']=self.measurements_info # add meta data for measurements 
            self.measurement_log=MO.DataProcessor(**temp) # create monitoring object
        human_time = HP.human_time(timestamp)
        if self.measurement_log is not None:
            for key, value in measurements.items():
                if ('angle' in value) and ('magnitude' in value):
                    value[m_value] = HP.pol_to_cart(value['magnitude'],value['angle'])
                self.measurement_log.receive_data(value[m_mrid],human_time,value[m_value])
        if human_time not in self._timestamp_set_on_centralized_measurement:
            self._timestamp_set_on_centralized_measurement.add(human_time)
            if self._feeder_simulation_time is not None:
                log.info(f"FF ID:{self.agent_id} time centralized: {human_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, agent time: {self._feeder_simulation_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, unix time: {HP.comp_time(human_time)}")
            else:
                log.debug(f"FF ID:{self.agent_id} time centralized: {human_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, unix time: {HP.comp_time(human_time)}")

    def on_measurement(self, headers: Dict, message) -> None:
        """Processes asynchronous measurement messages from sensors in the distribution power system simulation.

        On the first reception of a measurement message, the message is saved to a JSON file for understanding the 
        structure of the data. For subsequent messages, it does nothing.
        """
        if not self._latch:
            with open(f"{os.environ.get('OUTPUT_DIR')}/feeder.json", "w", encoding="UTF-8") as file:
                temp={}
                temp['message']=message
                temp['headers']=headers
                file.write(json.dumps(temp))
            self._latch = True      

    def on_upstream_message(self, headers: Dict, message) -> None:
        """Placeholder method for handling upstream message.

        Not used given currently their is no upstream message.

        Args:
            headers (Dict): A dictionary containing headers associated with the request.
            message (Dict): The request message to be processed.
        """

    def update_area_id_name_dict(self) -> None:
        """Updates internal dictionaries that map `SampleSwitchAreaAgent` IDs to names and vice versa.

        This method refreshes two internal dictionaries: one that maps `SampleSwitchAreaAgent` IDs to their corresponding 
        names and another that maps names to their corresponding IDs. This ensures that the mappings are current and 
        accurate.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It updates internal dictionaries used for mapping IDs and names.
        """
        temp=self.agent_dic
        area_id_names={}
        names_area_id={}
        for key, value in temp.items():
            area_id_names[key]=value['name']
            names_area_id[value['name']]=key
        self._area_id_names=area_id_names
        self._names_area_id=names_area_id
    
    def test_status_of_all_agents(self) -> None:
        """Verifies if all registered instances of `SampleSwitchAreaAgent` have informed the feeder agent of completion.

        This method checks the status of all registered instances to confirm whether they have communicated to the 
        feeder agent that they have finished their processing. If all instances have reported completion, the power 
        system simulation can proceed.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It performs internal checks to determine if the simulation 
                can proceed based on the status of all agents.
        """
        number=0
        for key, value in self.agent_dic.items():
            number += value['status']
        if number == len(self.agent_dic.keys()):
            for nested_dict in self.agent_dic.values():
                nested_dict['status'] = 0 # sets all back to 0
            if self._perform_self_time_control_call:
                if self.new_cb_s_state is not None:
                    self.update_cb_status(self.new_cb_s_state)
                    self.new_cb_s_state = None
                self.time_control()

    def on_downstream_message(self, headers: Dict, message) -> None:
        """Processes asynchronous messages from the downstream message bus.

        This method handles messages received from the downstream message bus. Messages with specific structures are 
        processed according to the command message and message type. Messages that do not match the expected structure 
        are logged with the instance ID, message header, and message content.

        Args:
            headers (Dict): A dictionary containing headers associated with the incoming message. The headers may 
                            include metadata such as instance ID and message type.
            message (Dict): A dictionary containing the content of the incoming message. The message is processed 
                            based on its structure and type.

        Returns:
            None: This method does not return any values. It performs processing and logging based on the message 
                received.
        """
        if all(key in message for key in keys_to_check):
            command = message['command']
            if not ((message['from']).encode('utf-8') == (self.agent_id).encode('utf-8')):
                log.debug(f"Command {command} downstream. FF ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
            if command == 'forward':
                self.message(message)
            elif command == 'register':
                if message['from'] not in self.agent_dic:
                    self.agent_dic[message['from']] = message['content'] # status is set and unset constantly 
                    if len(self.agent_area_dict['switch_areas']) == len(self.agent_dic):
                        message=make_empty_message_dict()
                        message['command']='registered_switch_area_agent'
                        message['content']={'agents':self.agent_dic,'measurements':self.measurements_info.to_json()}
                        message['from']=self.agent_id
                        self.publish_downstream(message)
                        self.update_area_id_name_dict()
                else:
                    raise ValueError(f"Agent with the same name attempted to get registered. FF ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
            elif command == 'get_event_done':
                if message['from'] in self.agent_dic:
                    self.agent_dic[message['from']]['status'] = 1 # Set agent done
                    if (self._feeder_simulation_time is not None):
                        self.test_status_of_all_agents()
                else:
                    raise ValueError(f"command get_event_done but agent not registered. FF ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
            elif (command == 'change_s_state') and (message['type']):
                self.new_cb_s_state=message['content']
                log.debug(f"FF ID:{self.agent_id}. change_s_state message: {message}")
            elif (command == "message") or (command == "registered_switch_area_agent"):
                """Do nothing"""
            else:
                log.info(f"Unknown message command. FF ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
        else:
            log.debug(f"Downstream FF ID:{self.agent_id}. Headers: {headers}, message: {message}")

    def on_request(self, message_bus, headers: Dict, message) -> None:
        """Placeholder method for handling synchronous communication requests between instances.

        This method is intended to be implemented for processing synchronous requests received via a message bus. 
        It currently serves as a placeholder and does not perform any operations. The method is designed to handle 
        requests with associated headers and messages.

        Args:
            message_bus (_type_): The message bus or communication channel used to receive the request.
            headers (Dict): A dictionary containing headers associated with the request.
            message (Dict): The request message to be processed.
        """
        log.debug(f"FF ID:{self.agent_id} Received request message_bus {message_bus}")
        log.debug(f"FF ID:{self.agent_id} Received headers from request: {headers}")
        log.debug(f"FF ID:{self.agent_id} Received message from request: {message}")
        reply_to = headers['reply-to']
        message_bus.send(reply_to, {'data':'this is a response'})
    
    def message(self,message):
        """Processes and forwards messages of type 'forwarding', including handling communication delays and message loss.

        This method handles incoming messages by recording them internally and forwarding them to the appropriate 
        recipients. It takes into account communication delays and potential message loss during the forwarding process.

        Args:
            message (Dict): A dictionary containing the message details. This includes information necessary for 
                            processing, handling delays, and managing message loss.

        Returns:
            None: This method does not return any values. It processes and forwards the message as described.
        """
        # simulation only function 
        m_from=self._area_id_names[message['from']]
        m_to=self._area_id_names[message['to']]
        message["_from"]=m_from
        message["_to"]=m_to
        specific=self.agent_config['message_config']['communication_area_agent_specific']
        default=self.agent_config['message_config']['communication_area_agent_default']
        if m_from+"_"+m_to in specific:
            temp=m_from+"_"+m_to
            delay=specific[temp]["message_delay"]
            lost=specific[temp]["message_lost"]
        else:
            delay=default["message_delay"]
            lost=default["message_lost"]
        # message["delay"]=delay
        # message["lost"]=lost
        message["command"]="message"
        message['simulation_time'] = HP.human_time(message['simulation_time'])
        message["simulation_time_received"] = message['simulation_time']
        while True:
            # message["simulation_time_received"]=(datetime.strptime(message["simulation_time_received"], "%Y-%m-%d %H:%M:%S.%f")+timedelta(seconds=delay)).strftime("%Y-%m-%d %H:%M:%S.%f")
            message_delay=timedelta(seconds=random.uniform(delay[0],delay[1]))
            message["simulation_time_received"]=message["simulation_time_received"]+message_delay
            if random.random() > lost:
                message["message_received"]="True"
                self.message_record.append(message.copy())
                break
            else:
                message["message_received"]="False"
                self.message_record.append(message.copy())
                # message["simulation_time_received"]=(datetime.strptime(message["simulation_time_received"], "%Y-%m-%d %H:%M:%S.%f")+timedelta(microseconds=300)).strftime("%Y-%m-%d %H:%M:%S.%f") # 50 microseconds delay for optic fiber 5 miles distance
                message["simulation_time_received"]=message["simulation_time_received"]+timedelta(microseconds=300) # 50 microseconds delay for optic fiber 5 miles distance
        try:
            message['simulation_time'] = HP.comp_time(message['simulation_time'])
            message["simulation_time_received"] = HP.comp_time(message["simulation_time_received"])
            self.publish_downstream(message)
        except Exception as e:
            raise ValueError(f"Fail to send message. FF ID:{self.agent_id} --- Error: {e} --- Message: {message}")
    
    def update_cb_status(self,CB:str) -> None:
        """Updates the status of circuit breakers based on the received CB string.

        This method compares the current status of circuit breakers with the provided CB string. If the status of 
        any circuit breaker differs from what is specified in the CB string, the method sends asynchronous messages 
        to open or close the relevant circuit breakers accordingly with open_CB and close_CB methods.

        Args:
            CB (str): A string representing the desired status of circuit breakers. The format of this string dictates 
                    which circuit breakers need to be opened or closed.
        """
        df=self._compleat_simulation_configuration['area_agent_configuration']['connections']['possible_connections_mrid']
        CB=CB.split("_")[-1]
        cb=self.system_status.split("_")[-1]
        any_change=False
        time.sleep(HardCodedTimeSleep.interval_2)
        for char, old_chart, (_, row) in zip(CB, cb, df.iterrows()):
            if char != old_chart:
                log.debug(f"FF ID:{self.agent_id}, state s change, char: {char}, old_chart: {old_chart}, row: {row} ")
                any_change=True
                step=1
                self._feeder_simulation_time += timedelta(seconds=step)
                self.simulation.simulation.resume_pause_at(step)
                time.sleep(HardCodedTimeSleep.interval_2)
                if char == "1":
                    self.close_CB(row['mRID'])
                else:
                    self.open_CB(row['mRID'])
                time.sleep(HardCodedTimeSleep.interval_2)
                self._feeder_simulation_time += timedelta(seconds=step)
                self.simulation.simulation.resume_pause_at(step)
                time.sleep(HardCodedTimeSleep.interval_2)
        if any_change:           
            message=make_empty_message_dict()
            message['command']='change_s_state'
            message['content']=CB
            message['from']=self.agent_id
            message['type']=False
            self.publish_downstream(message)

    def open_CB(self,CB_mrid:str) -> None:
        """Sends an asynchronous message to open a circuit breaker.

        This method constructs and sends an asynchronous message to open the circuit breaker identified by the 
        provided MRID (Measurement Reference ID) string.

        Args:
            CB_mrid (str): The MRID of the circuit breaker to be opened. This string uniquely identifies the circuit 
                        breaker in the distribution network.

        Returns:
            None: This method does not return any values. It sends an asynchronous message to open the specified 
                circuit breaker.
        """
        difference_builder = DifferenceBuilder(self.simulation_id)
        difference_builder.add_difference(CB_mrid, "Switch.open", 1, 0)
        self.simulation.gapps.send(t.simulation_input_topic(self.simulation_id),difference_builder.get_message())
    
    def close_CB(self,CB_mrid:str) -> None:
        """Sends an asynchronous message to close a circuit breaker.

        This method constructs and sends an asynchronous message to close the circuit breaker identified by the 
        provided MRID (Measurement Reference ID) string.

        Args:
            CB_mrid (str): The MRID of the circuit breaker to be closed. This string uniquely identifies the circuit 
                        breaker in the distribution network.

        Returns:
            None: This method does not return any values. It sends an asynchronous message to close the specified 
                circuit breaker.
        """
        difference_builder = DifferenceBuilder(self.simulation_id)
        difference_builder.add_difference(CB_mrid, "Switch.open", 0, 1)
        self.simulation.gapps.send(t.simulation_input_topic(self.simulation_id),difference_builder.get_message())


class SampleSwitchAreaAgent(SwitchAreaAgent):
    """A sample implementation of a switch area agent for managing and processing messages in a distribution network.

    This class extends `SwitchAreaAgent` and provides specific implementations for handling asynchronous messages, 
    evaluating reconnection paths, managing circuit breaker status, and registering the agent. It includes methods 
    for processing messages, updating sensor data, and communicating with other agents.
    """
    def __init__(self,
                 upstream: MessageBusDefinition,
                 downstream: MessageBusDefinition,
                 config: Dict,
                 area: Dict,
                 simulation_id: str,
                 name: str,
                 agent_config: Dict) -> None:
        """Initializes the SampleSwitchAreaAgent with the given parameters and registers the agent.

        Args:
            upstream (MessageBusDefinition): The message bus used for receiving messages from upstream.
            downstream (MessageBusDefinition): The message bus used for sending messages downstream.
            config (Dict): Configuration dictionary for initializing the agent.
            area (Dict): Dictionary containing information about the area managed by the agent.
            simulation_id (str): Identifier for the simulation.
            name (str): Name of the agent.
            agent_config (Dict): Additional configuration dictionary for the agent.
        """
        self._feeder_simulation_time = None
        self._latch = False
        self._timestamp_set_on_measurement=set()
        self.counter = 0
        self.last_update = 0
        self._location = ""
        self.source_received = False
        self.bus_info = {}
        self.branch_info = {}
        self.mrid_map = {}
        self.area = area
        super().__init__(upstream, downstream, config, area, simulation_id)
        # utils.get_all_data(self.switch_area)
        # self.measurements_info=measurement_description(self.switch_area,self.agent_id)
        self.sensor_log=None
        self.message_record=[]
        self.areas_disconnected=[]
        self.agent_dic=None # obtain by communication
        self._compleat_simulation_configuration=agent_config
        self.area_agent_configuration = agent_config['area_agent_configuration']
        self.update_status=False
        self.system_status=agent_config['feeder_agent_configuration']['initial_CB_state']
        self.name=name

        self.register()

    def register(self) -> None:
        """Sends a specific registration message for the feeder agent.

        This method sends a registration message to register the feeder agent. It is called only once during the 
        instance creation to ensure that the agent is properly registered.
        """
        message = make_empty_message_dict()
        message['command'] = 'register'
        message['from'] = self.agent_id
        message['content'] = {'status':0,'name':self.name,
            'switch_boundary':self.area['boundary_switches']}
        self.publish_upstream(message)

    def process_messages(self) -> None:
        """Processes messages to update the knowledge of area connections and manage circuit breaker changes.

        This method self-evaluates the instance's message record to process incoming messages. It updates the internal 
        knowledge of area connections and reconnection path evaluations. Based on the processed information, it calls 
        the `manage_change_cb` method to decide whether to change the status of circuit breakers.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It performs internal updates and may trigger actions.
        """
        df=pd.DataFrame(self.message_record)
        if len(df)>0:
            df=df.loc[(df['message_received']=="True") & (df['simulation_time_received']<=self._feeder_simulation_time)]
            if len(df)>0:
                # areas disconnected 
                self.process_disconnected_messages(dff=df.loc[df['type']=='self_connection'].copy())
                # evaluate reconnection path voltage eval
                self.eval_reconnection_message_eval(dff=df.loc[df['type']=='eval_reconnection_path'].copy())
                # process received reconnection path voltage eval --> change CB if accepted
                if self._FLISR_variables['messages_returns_from_request_v_eval'] and self._FLISR_variables['area_is_disconnected'] and self._FLISR_variables['not_sent_cb_change']:
                    self.menage_change_cb(dff=df.loc[df['type']=='reply_eval_reconnection_path'].copy()) 

    def on_measurement(self, headers: Dict, message: Dict) -> None:
        """Processes asynchronous measurement messages from sensors in the distribution power system simulation.

        On the first reception of a measurement message, the message is saved to a JSON file for understanding the 
        structure of the data. For subsequent messages, if the simulation time and sensor log object are available, 
        the message is processed by the sensor log object.

        Args:
            headers (Dict): A dictionary containing headers associated with the measurement message. Headers may 
                            include metadata such as sensor ID and timestamp.
            message (Dict): A dictionary containing the measurement data from the sensor. The structure of the data 
                            is used for processing and logging.

        Returns:
            None: This method does not return any values. It performs actions based on the reception and processing 
                of measurement messages.
        """
        m_value='value'
        m_mrid='measurement_mrid'
        if not self._latch:
            aa=self.agent_id.split('.')[-1]
            with open(f"{os.environ.get('OUTPUT_DIR')}/switch_{aa}.json", "w", encoding="UTF-8") as file:
                temp={}
                temp['message']=message
                temp['headers']=headers
                file.write(json.dumps(temp))
            self._latch = True
        else:
            if (self.sensor_log is not None) and (self._feeder_simulation_time is not None):
                for key, value in message.items():
                    if ('angle' in value) and ('magnitude' in value):
                        value[m_value] = HP.pol_to_cart(value['magnitude'],value['angle'])
                    self.sensor_log.receive_data(value[m_mrid],self._feeder_simulation_time,value[m_value])

    def on_upstream_message(self, headers: Dict, message: Dict) -> None:
        """Processes asynchronous messages from the upstream message bus.

        This method handles messages received from the upstream message bus. Messages with specific structures are 
        processed according to the command message and message type. Messages that do not match the expected structure 
        are logged with the instance ID, message header, and message content.

        Args:
            headers (Dict): A dictionary containing headers associated with the incoming message. The headers may 
                            include metadata such as instance ID and message type.
            message (Dict): A dictionary containing the content of the incoming message. The message is processed 
                            based on its structure and type.

        Returns:
            None: This method does not return any values. It performs processing and logging based on the message 
                received.
        """
        if all(key in message for key in keys_to_check):
            command = message['command']
            # log.debug(f"Command {command}. FA ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
            if (command == "message") and (message["to"] == self.agent_id):
                message['simulation_time'] = HP.human_time(message['simulation_time'])
                message["simulation_time_received"] = HP.human_time(message["simulation_time_received"])
                if (message['type'] == 'eval_reconnection_path') or (message['type'] == 'reply_eval_reconnection_path'):
                    message['content']=pd.read_json(StringIO(message['content']))
                self.message_record.append(message)
            elif (command=="registered_switch_area_agent"):
                self.agent_dic=message["content"]['agents']
                nested_dict=self.agent_dic
                self.agent_dic_ID_to_name = {outer_key: inner_dict['name'] for outer_key, inner_dict in nested_dict.items()}

                self.measurements_info=pd.read_json(StringIO(message["content"]['measurements']))
                if self.sensor_log is None:
                    self.sensor_log=MT.DataTrekker(self.measurements_info,self.area_agent_configuration['sensors'])
            elif (command=='change_s_time'):
                if self._feeder_simulation_time is None:
                    self._feeder_simulation_time=HP.human_time( message["content"]['time'] )
                    self.initiate_FLISR_variables()
                self._feeder_simulation_time=HP.human_time( message["content"]['time'] )
                if message['type']:
                    self.update_sensors()
                    self.process_messages()
                    self.test_if_disconnected()
                    time.sleep(HardCodedTimeSleep.interval_3)
                    self.send_event_done_message()
            elif (command=='change_s_state') and not(message['type']):
                self.system_status=message['content']
                log.debug(f"FA ID:{self.agent_id}. change_s_state message: {message}")
            elif (command=="forward") or (command=="register") or (command=="get_event_done") or (command=="message"):
                """Do nothing"""
            else:
                log.info(f"Unknown message command. FA ID:{self.agent_id} --- Headers: {headers} --- Message: {message}")
        else:
            log.debug(f"Upstream FA ID:{self.agent_id}. Header: {headers}. Message: {message}")

    def on_downstream_message(self, headers: Dict, message: Dict) -> None:
        """Placeholder method for handling asynchronous messages from downstream

        Args:
            headers (Dict): message header
            message (Dict): message
        """        
        log.debug(f"FA ID:{self.agent_id} Received message headers from downstream message bus: {headers}")
        log.debug(f"FA ID:{self.agent_id} Received message from downstream message bus: {message}")

    def on_request(self, message_bus, headers:Dict, message:Dict) -> None:
        """Placeholder method for handling synchronous communication requests between instances.

        This method is intended to be implemented for processing synchronous requests received via a message bus. 
        It currently serves as a placeholder and does not perform any operations. The method is designed to handle 
        requests with associated headers and messages.

        Args:
            message_bus (_type_): The message bus or communication channel used to receive the request.
            headers (Dict): A dictionary containing headers associated with the request.
            message (Dict): The request message to be processed.
        """
        log.debug(f"FA ID:{self.agent_id} Received request message_bus {message_bus}")
        log.debug(f"FA ID:{self.agent_id} Received headers from request: {headers}")
        log.debug(f"FA ID:{self.agent_id} Received message from request: {message}")
        reply_to = headers['reply-to']
        message_bus.send(reply_to, {"message":"max bha"})
    
    def send_event_done_message(self) -> None:
        """Sends a message to inform that the instance has completed making evaluations.

        This method creates and sends an asynchronous message to notify that the instance has finished its evaluation 
        process. The communication is asynchronous, so this method initiates the sending of the message but does not 
        wait for a response.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It handles the initiation of asynchronous communication.
        """
        message = make_empty_message_dict()
        message['command'] = 'get_event_done'
        message['from'] = self.agent_id
        message['content'] = {'status':1,'name':self.name}
        self.publish_upstream(message)
    
    #### FLISR   
    def update_sensors(self) -> None:
        """Updates the sensor measurement data with the most recent data received from the distribution system simulation.

        This method retrieves and processes the latest sensor measurement data from the distribution system simulation 
        and updates the internal records with this new information.

        Args:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any values. It updates the internal sensor measurement data with the 
                most recent data.
        """
        self.sensor_a=self.sensor_log.get_most_recent_data()

    def process_disconnected_messages(self,dff:pd.DataFrame) -> None:
        """Processes self-connection status messages from all instances to update the current knowledge of disconnected areas.

        This method receives a DataFrame containing self-connection status messages from all instances, including 
        time information. It updates the internal state to reflect the current status of areas that are disconnected 
        based on the received messages.

        Args:
            dff (pd.DataFrame): A DataFrame with self-connection status messages from all instances. The DataFrame 
                                includes time information and status updates for areas, which are used to update the 
                                internal knowledge of disconnected areas.

        Returns:
            None: This method does not return any values. It updates the internal state of the instance based on the 
                received messages.
        """
        if len(dff)==0:
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Area has no self_connection type messages.")
            return None
        dff=dff.sort_values(by='simulation_time', ascending=False)
        dff=dff.drop_duplicates(subset=['from'],keep="first")
        dff=dff.loc[dff['content']=='NO']
        temp=dff['from'].to_list()
        # update names
        temp = [self.agent_dic_ID_to_name.get(item, item) for item in temp]

        temp=[int(i.split("_")[-1]) for i in temp]
        if self._FLISR_variables['area_is_disconnected']:
            temp.append(int(self.name.split("_")[-1]))
        self.areas_disconnected=temp #### TODO: include flag if set of areas has change 
        if len(temp) > 0:
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. System area disconnected: {temp}")

    def eval_reconnection_message_eval(self,dff:pd.DataFrame) -> None:
        """Evaluates reconnection paths based on a DataFrame of evaluation requests from another instance.

        This method receives a DataFrame containing reconnection paths evaluation requests from another instance. 
        It processes the evaluation of these paths and sends an asynchronous message with the results back to the 
        requesting instance. The evaluated reconnection paths evaluation message is then removed from the instance's 
        queue of messages.

        Args:
            dff (pd.DataFrame): A DataFrame with reconnection paths evaluation requests from another instance. 
                                The DataFrame should contain the paths to be evaluated and any relevant parameters 
                                required for the evaluation.

        Returns:
            None: This method does not return any values. It performs asynchronous communication and message management.
        """
        if len(dff)==0:
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Area has no area evaluation voltage requests.")
            return None
        # there are and the self.message_record needs to them once analyses is performed 
        for index, row in dff.iterrows():
            temp=row['content'].copy()
            result=self.evaluate_reconnection_path(paths_eval_df=temp,local=False)
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Resulted v response:\n {result}")
            message_delay=self.area_agent_configuration['FLISR']['range_message_respond_of_paths']
            message_delay=timedelta(seconds=random.uniform(message_delay[0],message_delay[1]))
            message = make_empty_message_dict()
            message['command']="forward"
            message['from'] = self.agent_id
            message['to'] = row["from"]
            message['content'] = result.to_json()
            message['simulation_time'] = HP.comp_time(self._feeder_simulation_time+message_delay)
            message['type']='reply_eval_reconnection_path'
            self.publish_upstream(message)
            # outmessage={"command":"message","from":self.name,"to":row["from"],"content":result,'simulation_time':self._feeder_simulation_time+message_delay,'type':'reply_eval_reconnection_path'}
            # self.communication_manager.tell(outmessage.copy())
            del self.message_record[index] # removing message from record since it has been evaluated already 
    
    def menage_change_cb(self,dff:pd.DataFrame) -> None:
        """Manages the status change of circuit breakers based on evaluated connection paths from other instances.

        This method receives a DataFrame containing evaluated connection paths from other instances. It determines 
        whether to change the status of multiple circuit breakers on the distribution network based on the agreement 
        of feasibility among instances. Reconnection paths that are radial and require fewer circuit breaker changes 
        are prioritized.

        Args:
            dff (pd.DataFrame): A DataFrame with evaluated connection paths from other instances. The DataFrame 
                                should include information necessary for assessing the feasibility of the reconnection 
                                paths and deciding a reconnection path.

        Returns:
            None: This method does not return any values. It executes commands to change the status of circuit breakers 
                as needed based on the evaluation.
        """
        dff=dff.loc[dff["simulation_time"]>=self._FLISR_variables['FLISR_event_start']] # keeping only messages from current event 
        if len(dff)==0:
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Area has no reply_eval_reconnection_path.")
            return None
        dff=dff.sort_values(by='simulation_time', ascending=False)
        dff=dff.drop_duplicates(subset=['from'],keep="first")

        source=self._FLISR_variables['evaluated_reconnection_path'].copy()
        source.drop(columns=['V magnitude maximum','V magnitude minimum'],inplace=True)
        
        sport_local_eval=[]
        for index, row in dff.iterrows():
            if test_df_are_equal(source,row['content'].drop(columns=['V magnitude maximum','V magnitude minimum'])):
                sport_local_eval.append(row['from'])
        if len(sport_local_eval) >= (len(self.agent_dic)-1)*0.5:
            # local eval if consider true ... select and execute change
            source=self._FLISR_variables['evaluated_reconnection_path'].copy()
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. The local evaluation has been supported:\n {source}")
            source['max_v_delta']=source['V magnitude maximum']-source['V magnitude minimum']
            source.sort_values(by='max_v_delta', ascending=True,inplace=True)
            first=source.loc[(source['connection_possible']) & (source['substation_radial'])]
            if len(first)>=1:
                first_row = first.iloc[0]
                new_path=first_row['cb_path']
                log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Submit new radial path of: {new_path}")
                message_delay=self.area_agent_configuration['FLISR']['range_message_change_the_paths']
                message_delay=timedelta(seconds=random.uniform(message_delay[0],message_delay[1]))
                message = make_empty_message_dict()
                message['command']="change_s_state"
                message['from'] = self.agent_id
                message['content'] = "CB_state_"+new_path
                message['simulation_time'] = HP.comp_time(self._feeder_simulation_time+message_delay)
                message['type']=True
                self.publish_upstream(message)
                # self.communication_manager.tell({"command":"change_s_state","simulation_time":self._feeder_simulation_time+message_delay,"content":"CB_state_"+new_path,"from":self.name}) 
                self._FLISR_variables['not_sent_cb_change']=False
                return None
            second=source.loc[(source['connection_possible'])]
            if len(second)>=1:
                first_row = second.iloc[0]
                new_path=first_row['cb_path']
                log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Submit new path of: {new_path}")
                message_delay=self.area_agent_configuration['FLISR']['range_message_change_the_paths']
                message_delay=timedelta(seconds=random.uniform(message_delay[0],message_delay[1]))
                message = make_empty_message_dict()
                message['command']="change_s_state"
                message['from'] = self.agent_id
                message['content'] = "CB_state_"+new_path
                message['simulation_time'] = HP.comp_time(self._feeder_simulation_time+message_delay)
                message['type']=True
                self.publish_upstream(message)
                # self.communication_manager.tell({"command":"change_s_state","simulation_time":self._feeder_simulation_time+message_delay,"content":"CB_state_"+new_path,"from":self.name})
                self._FLISR_variables['not_sent_cb_change']=False
                return None
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. No changes given the possible solutions.")

    def initiate_FLISR_variables(self) -> None:
        """Initializes the fault location isolation and service restoration (FLISR) variables.

        This method sets up the variables used for the FLISR process, which involves isolating events and securing 
        a time limit for reconnection attempts. It also initializes other auxiliary variables necessary for 
        the cyclical FLISR process.

        This method does not take any arguments and does not return any values. 
        It is responsible for preparing the necessary state for subsequent FLISR operations.
        """
        self._FLISR_variables=dict()
        self._FLISR_variables['FLISR_event_start']=self._feeder_simulation_time
        self._FLISR_variables['time_to_FLISR']=self._feeder_simulation_time+timedelta(minutes=self.area_agent_configuration['FLISR']['initial_time_to_FLISR'])
        self._FLISR_variables['time_to_await_other_area_status_for_fault_identification']=self._feeder_simulation_time+timedelta(minutes=self.area_agent_configuration['FLISR']['time_to_await_other_area_status_for_fault_identification'])
        if self.update_status:
            self.update_status=False
        else:
            self.prior_system_status=self.system_status
        self._FLISR_variables['area_was_disconnected']=None
        self._FLISR_variables['area_is_disconnected']=None
        self._FLISR_variables['area_at_fault']=None
        self._FLISR_variables['evaluated_reconnection_path']=None
        self._FLISR_variables['past_requested_path_eval']=None
        self._FLISR_variables['messages_returns_from_request_v_eval']=False
        self._FLISR_variables['not_sent_cb_change']=True
    
    def test_if_disconnected(self) -> None:
        """Tests if the current instance is disconnected. 

        This method performs a self-evaluation to determine if the instance is currently connected or disconnected. 
        It will also send messages to other agents if there is a change in connection status or if an assessment 
        of a reconnection path is needed. The communication with other agents is asynchronous, and this method 
        initiates the sending of messages but does not wait for their responses.

        This method does not return any values or take any arguments.
        """
        if (self._FLISR_variables['area_is_disconnected'] == None) and (self._FLISR_variables['area_was_disconnected'] == None) and (self._FLISR_variables['area_at_fault'] == None):
            self.initiate_FLISR_variables()
        if (self._FLISR_variables['area_is_disconnected'] == True) and (self._FLISR_variables['area_was_disconnected'] == None) and (self._FLISR_variables['area_at_fault'] == None):
            self._FLISR_variables['area_was_disconnected']=True
        self._FLISR_variables['area_is_disconnected']=PSA.area_disconnected(self.sensor_a) #### Identify if area is disconnected
        log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Is area disconnected: {self._FLISR_variables['area_was_disconnected']}")

        if not(self._FLISR_variables['area_is_disconnected']) and (self._FLISR_variables['area_was_disconnected']==True):
            self.tell_all_other_area_agents('YES','self_connection',"range_message_change_of_state") # area is connected 
            self.initiate_FLISR_variables()
        elif not(self._FLISR_variables['area_is_disconnected']):
            self.initiate_FLISR_variables()
        elif (self._FLISR_variables['area_is_disconnected'] == True) and (self._FLISR_variables['area_was_disconnected'] == None) and (self._FLISR_variables['area_at_fault'] == None):
            self.tell_all_other_area_agents('NO','self_connection',"range_message_change_of_state") # area is NOT connected 
            self._FLISR_variables['area_was_disconnected']=True # Avoiding the possible problem 

        if (self._FLISR_variables['time_to_FLISR'] >= self._feeder_simulation_time) and (self._feeder_simulation_time>=self._FLISR_variables['time_to_await_other_area_status_for_fault_identification']):
            #### Discover is area is at fault 
            if (self._FLISR_variables['area_is_disconnected'] == True) and (self._FLISR_variables['area_was_disconnected'] == True) and (self._FLISR_variables['area_at_fault'] == None):
                temp=PSA.area_at_fault(self.prior_system_status,self.areas_disconnected,self.area_agent_configuration['connections']['possible_connections'],int(self.name.split("_")[-1]))
                if isinstance(temp, str):
                    log.info(f"FA ID:{self.agent_id}. Agent {self.name}, error on area_at_fault. Error ocurred on simulation time: {self._feeder_simulation_time}. Error message: {temp}")
                else:
                    log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Is area at fault: {temp}")
                    self._FLISR_variables['area_at_fault'] = temp
                CB=self.prior_system_status.split('_')[-1]
                df=self.area_agent_configuration['connections']['possible_connections'].copy()
                areas=self.areas_disconnected
                s_list = list(CB)
                for i in df.loc[(df['area1'].abs().isin(areas)) | (df['area2'].abs().isin(areas))].index:
                    s_list[i]="0"
                CB=''.join(s_list)
                message=make_empty_message_dict()
                message['command']='change_s_state'
                message['content']="CB_state_"+CB
                message['from']=self.agent_id
                message['type']=True
                self.publish_upstream(message)
                self.system_status="CB_state_"+CB
            #### Connections paths
            if (self._FLISR_variables['area_is_disconnected'] == True) and (self._FLISR_variables['area_was_disconnected'] == True) and (self._FLISR_variables['area_at_fault'] == False):
                temp=PSA.reconnection_paths(self.prior_system_status,self.areas_disconnected,self.area_agent_configuration['connections']['possible_connections'],int(self.name.split("_")[-1])) #### Identify the possible reconnection paths with cost given the number of needed changes
                log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Reconnection paths:\n {temp}")
                # Area there possible reconnection paths
                if len(temp)==0:
                    self._FLISR_variables['past_requested_path_eval'] = None
                else:
                    # evaluate paths
                    paths_eval=[]
                    for i in temp.keys():
                        if '_' in i:
                            """Do nothing"""
                        else:
                            print(temp[i])
                            for j in temp[i]:
                                paths_eval.append(PSA.identify_new_CB(system_status=self.system_status,possible_connections=self.area_agent_configuration['connections']['possible_connections'],path=j))
                    paths_eval_df=pd.DataFrame(paths_eval)
                    # Test if it should be sent to other areas
                    send_message=False
                    if (self._FLISR_variables['past_requested_path_eval'] is None):
                        log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Reconnection paths eval df:\n {paths_eval_df}")
                        send_message=True
                    elif (not test_df_are_equal(self._FLISR_variables['past_requested_path_eval'].drop(columns=['cbs_to_close']),paths_eval_df.drop(columns=['cbs_to_close']))):
                        log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Reconnection paths eval df not equal:\n {paths_eval_df}")
                        send_message=True
                    if send_message:
                        self._FLISR_variables['past_requested_path_eval'] = paths_eval_df.copy()
                        self.tell_all_other_area_agents(self._FLISR_variables['past_requested_path_eval'].to_json(),'eval_reconnection_path',"range_message_create_paths_send") # request to eval reconnection path 
                        self._FLISR_variables['messages_returns_from_request_v_eval']=True
                        
                        self.evaluate_reconnection_path(paths_eval_df=paths_eval_df,local=True)
    
    def is_matching_connection(self,row:pd.Series, connection_list:list[tuple]) -> bool:
        """Checks if a connection specified by the given row exists in the connection list.

        Args:
            row (pd.Series): A Pandas Series representing a row of data with 'area1' and 'area2' fields.
            connection_list (list[tuple]): A list of tuples where each tuple represents a connection 
                                          with two elements corresponding to areas.

        Returns:
            bool: True if the connection specified by the row ('area1', 'area2') exists in the connection list, 
                  False otherwise.
        """        
        return (row['area1'],row['area2']) in connection_list

    def evaluate_reconnection_path(self,paths_eval_df:pd.DataFrame, local:bool) -> pd.DataFrame:
        """Evaluates reconnection paths based on the given DataFrame and local flag.

        Args:
            paths_eval_df (pd.DataFrame): A DataFrame containing paths to be evaluated. 
                                        The DataFrame should include relevant columns for path evaluation.
            local (bool): A boolean flag indicating whether the evaluation is for self (True) 
                        or for a request from another source (False).

        Returns:
            pd.DataFrame: A DataFrame with the results of the path evaluation.
        """
        base_possible_connections=self.area_agent_configuration['connections']['possible_connections'].copy()
        base_area_connections_base=self.area_agent_configuration['connections']['area_connections_base'].copy()
        base_substation_area_connection=self.area_agent_configuration['connections']['substation_area_connection'].copy()
        base_sensor=self.area_agent_configuration['sensors'].copy()
        base_sensors_area_max_load=self.area_agent_configuration['max_area_load'].copy()
        base_ybus_reduced=self.area_agent_configuration["ybus_reduced"]
        # eval paths
        eval_paths_se=[]
        for index, row in paths_eval_df.iterrows():
            possible_connections=base_possible_connections.copy()
            area_connections_base=base_area_connections_base.copy()
            substation_area_connection=base_substation_area_connection.copy()
            sensor=base_sensor.copy()
            sensors_area_max_load=base_sensors_area_max_load.copy()
            areas=np.array(row['areas_involved'])
            areas=areas[areas>=0]
            
            cb_new=str(row['cb_path'])

            log.debug(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. cb_new: {cb_new}")

            indices_of_ones = [i for i, char in enumerate(cb_new) if char == '1']
            possible_connections=possible_connections.loc[possible_connections.index.isin(indices_of_ones)]
            possible_connections=possible_connections.loc[(possible_connections['area1'].isin(areas)) & (possible_connections['area2'].isin(areas))]
            # log.info(f"Agent {self.name}. Simulation time: {self.simulation_time}. Possible connections df:\n {possible_connections}")
            tuple_list = [tuple(x) for x in possible_connections[['area1','area2']].itertuples(index=False)]

            # Use .apply to filter rows based on the custom function with connection_list as a parameter
            log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Tuple list:\n {tuple_list}")
            area_connections = area_connections_base[area_connections_base.apply(self.is_matching_connection, axis=1, connection_list=tuple_list)]
            # log.info(f"Agent {self.name}. Simulation time: {self.simulation_time}. Possible area connections:\n {area_conections}")
            # log.info(f"ybus_reduced {self.ybus_reduced}. areas: {areas}.")
            areas_name=["area_"+str(a) for a in areas]
            temp_list=[]
            for i in areas_name:
                temp_list.append(base_ybus_reduced[i])
            Y_BUS=pd.concat(temp_list)
            Y_BUS.reset_index(drop=True,inplace=True)

            sensor=sensor.loc[(sensor['node1'].isin(Y_BUS.columns)) & (sensor['node2'].isin(Y_BUS.columns))]
            sensor.reset_index(drop=True,inplace=True)

            sensors_area_max_load=sensors_area_max_load.loc[(sensors_area_max_load['node1'].isin(Y_BUS.columns)) & (sensors_area_max_load['node2'].isin(Y_BUS.columns))]
            sensors_area_max_load.reset_index(drop=True,inplace=True)
            output=PSA.get_results_model_reduction(Y_BUS=Y_BUS,area_connections=area_connections,
                                            sensor=sensor,substation_area_connection=substation_area_connection,
                                            sensors_area_max_load=sensors_area_max_load)
            # log.info(f"Agent {self.name}. Simulation time: {self.simulation_time}. Reconnection paths voltage eval:\n {output}")
            eval_paths_se.append(output)
        eval_paths_se=pd.DataFrame(eval_paths_se)
        log.info(f"FA ID:{self.agent_id}. Agent {self.name}. Simulation time: {self._feeder_simulation_time}. Reconnection paths voltage eval list:\n {eval_paths_se}")
        # Concatenate df1 and df2 horizontally (along columns)
        result=pd.concat([paths_eval_df, eval_paths_se], axis=1)
        if local:
            self._FLISR_variables['evaluated_reconnection_path'] = result.copy()
        else:
            return result


    def tell_all_other_area_agents(self,m_content:str,m_type:str,key_delay:str) -> None:
        """Sends asynchronous messages to all other area agents.

        This method constructs and sends asynchronous messages with the specified content, message type, and delay 
        key to all other area agents. The delay key points to the time require to make the information for the message.
        The delay given the transportation delay is performed by the feeder agent. The messages are sent based on the 
        provided parameters.

        Args:
            m_content (str): The content of the message to be sent.
            m_type (str): The type of the message, which determines the nature of the message and how it should be handled.
            key_delay (str): A delay key that may influence the timing or order of message delivery.

        Returns:
            None: This method does not return any values. It sends asynchronous messages to other area agents.
        """
        message = make_empty_message_dict()
        message['command'] = 'forward'
        message['from'] = self.agent_id
        message['type'] = m_type #'eval_reconnection_path'
        message['content'] = m_content
        #### specific time conditions --> simulation added delay for computing the data for message
        process_delay=self.area_agent_configuration['FLISR'][key_delay]
        message_delay=timedelta(seconds=random.uniform(process_delay[0],process_delay[1]))
        if key_delay == "range_message_change_of_state":
            self._FLISR_variables['time_to_await_other_area_status_for_fault_identification'] += message_delay
        message['simulation_time'] = HP.comp_time(self._feeder_simulation_time+message_delay)
        #### send message to all
        for i in self.agent_dic.keys():
            if i != self.agent_id:
                message['to']=i
                self.publish_upstream(message)