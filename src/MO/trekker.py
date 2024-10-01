import pandas as pd
import uuid
from datetime import datetime, timedelta

import logging
log = logging.getLogger(__name__)

class DataTrekker:
    """A class used to process and store sensor data from simulations.

    Differently than the class DataProcessor. This class store data is utilized for decision making. Furthermore, this class 
    does not write any information to file. The simulation of the distributed power grid sensors is converted to the sensors
    according to the input sensor information with specific variances (i.e., precision). The data can be called to be utilized in
    evaluating the switch delimited area during the simulation.

    Attributes:
        meta_data (pd.DataFrame): The metadata for the simulation.
        sensors (pd.DataFrame): The sensor data for the simulation.
        max_rows (int): The maximum number of rows to store before removing old data.
        columns (list): The columns of the data being processed.
        sensor_data (pd.DataFrame): A DataFrame to temporarily store sensor data.
    """
    def __init__(self,
                 meta_data:pd.DataFrame,
                 sensors:pd.DataFrame,
                 max_rows:int=200000) -> None:
        """Initializes the DataTrekker with the given parameters.

        Args:
            meta_data (pd.DataFrame): The metadata for the simulation.
            sensors (pd.DataFrame): The sensor data for the simulation.
            max_rows (int, optional): The maximum number of rows to store before removing old data. Defaults to 200000.
        """
        self.process_metadata(meta_data.copy())
        self.process_sensors(sensors.copy())
        self.max_rows=max_rows
        self.columns=['GUID', 'timestamp', 'measurement']
        self.sensor_data = []
    
    def process_metadata(self,meta_data: pd.DataFrame) -> None:
        """Processes and updates the metadata.

        Args:
            meta_data (pd.DataFrame): The metadata to be processed.
        """
        update_dict = {'A': '1', 'B': '2','C':'3'}
        meta_data['phase']=meta_data['phase'].map(update_dict)
        meta_data['t_node_name'] = meta_data['t_node_name'].fillna(meta_data['f_node_name'])
        meta_data['f_node_name']=meta_data['f_node_name']+'.'+meta_data['phase']
        meta_data['t_node_name']=meta_data['t_node_name']+'.'+meta_data['phase']
        self.meta_data=meta_data
    
    def process_sensors(self,sensors: pd.DataFrame) -> None:
        """Processes and updates the sensor data.

        Args:
            sensors (pd.DataFrame): The sensor data to be processed.
        """
        sensors['GUID'] = [str(uuid.uuid4()) for _ in range(len(sensors))]
        sensors['node1']=sensors['node1'].str.lower()
        sensors['node2']=sensors['node2'].str.lower()
        self.sensors=sensors

    def add_receive_data(self,new_data:pd.DataFrame) -> None:
        """Adds new sensor data to the existing data storage.

        Args:
            new_data (pd.DataFrame): The new data to be added.
        """
        if len(self.sensor_data) > 0:
            self.sensor_data = pd.concat([self.sensor_data, new_data], ignore_index=True)
        else:
            self.sensor_data = new_data.copy(deep=True)

    def receive_data(self, unique_id: str, timestamp: pd.Timestamp, value: complex) -> None:
        """Receives and processes incoming sensor data.

        Args:
            unique_id (str): The unique identifier for the data entry.
            timestamp (pd.Timestamp): The timestamp of the data entry.
            value (complex): The value of the data entry.
        """
        sensors=self.sensors
        meta_data=self.meta_data
        meta={**meta_data.loc[meta_data['measurement_mRID']==(unique_id),['f_node_name','t_node_name','measurement_type']].to_dict(orient='records')[0]}
        try:
            if meta['measurement_type'] == 'VA':
                unique_id=sensors.loc[(sensors['sensor_type']=='Pi')&(sensors['node1']==meta['f_node_name']),'GUID'].item()
                self.add_receive_data(pd.DataFrame([[unique_id, timestamp, value.real]], columns=self.columns))
                unique_id=sensors.loc[(sensors['sensor_type']=='Qi')&(sensors['node1']==meta['f_node_name']),'GUID'].item()
                self.add_receive_data(pd.DataFrame([[unique_id, timestamp, value.imag]], columns=self.columns))
            elif meta['measurement_type'] == 'PNV':
                unique_id=sensors.loc[(sensors['sensor_type']=='Vi')&(sensors['node1']==meta['f_node_name']),'GUID'].item()
                self.add_receive_data(pd.DataFrame([[unique_id, timestamp, abs(value)]], columns=self.columns))
        except Exception as e:
            log.debug(f"meta: {meta},unique_id: {unique_id},timestamp: {timestamp},value: {value}, exception: {e}.")

    def get_most_recent_data(self) -> pd.DataFrame:
        """Retrieves the most recent sensor data.

        Returns:
            pd.DataFrame: A DataFrame containing the most recent data.
        """
        sensor_data = self.sensor_data
        sensors=self.sensors.copy()
        sensors.drop(columns='measurement',inplace=True)
        latest_indices = sensor_data.groupby('GUID')['timestamp'].idxmax()
        latest_info_df = sensor_data.loc[latest_indices].copy()
        latest_info_df.drop(columns='timestamp',inplace=True)
        
        data=sensors.merge(latest_info_df,on='GUID',how='inner')
        
        self.organize_data()
        self.remove_old_data()

        return data.drop(columns='GUID')
    
    def organize_data(self) -> None:
        """
        Organizes the sensor data by sorting and removing duplicates.
        """
        self.sensor_data = self.sensor_data.sort_values(by='timestamp', ascending=False)
        self.sensor_data=self.sensor_data.drop_duplicates()
        self.sensor_data.reset_index(drop=True,inplace=True)
    
    def remove_old_data(self) -> None:
        """
        Removes old sensor data exceeding the maximum row limit.
        """
        self.sensor_data = self.sensor_data.iloc[-self.max_rows:]



if __name__ == "__main__":
    import os
    # Example usage
    base_path = "~/grid/dltdflisr"
    base_path = os.path.expanduser(base_path)
    sensors=f"{base_path}/config/sensors.h5"
    connections=f"{base_path}/config/area_connections.h5"
    max_area_load=f"{base_path}/config/max_node_load.h5"

    meta=f"{base_path}/outputs/record_00.h5"
    measurement=f"{base_path}/outputs/switch_4.json"

    sensors=pd.read_hdf(sensors)
    meta_data=pd.read_hdf(meta,key='/meta_data')

    # make object 
    processor=DataTrekker(meta_data,sensors)
    # Receiving some data
    processor.receive_data('_924dfad1-f264-4835-8c10-58782ab8e24e', datetime(2023, 6, 24, 12, 0), 10)
    processor.receive_data('_924dfad1-f264-4835-8c10-58782ab8e24e', datetime(2023, 6, 24, 12, 5), 15)
    processor.receive_data('_cadea118-f954-4402-ab81-e6db36e836fe', datetime(2023, 6, 24, 12, 10), 8)
    processor.receive_data('_cadea118-f954-4402-ab81-e6db36e836fe', datetime(2023, 6, 24, 12, 0), 20)
    processor.receive_data('_cadea118-f954-4402-ab81-e6db36e836fe', datetime(2023, 6, 24, 12, 5), 25)

    # get latest data
    print(processor.get_most_recent_data())

