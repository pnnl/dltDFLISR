import pandas as pd
from datetime import datetime, timedelta

import logging
log = logging.getLogger(__name__)

class DataProcessor:
    """A class used to process and store data from simulations.

    This class initialize the single output file from the simulation. This class stores and process the sensor measurement data only.
    The measurement data store by this class object is not utilized in the analysis.

    Attributes:
        file_name (str): The name of the file to store the processed data.
        w_unprocessed_data (bool): A flag indicating if raw data should be stored without processing.
        max_rows (int): The maximum number of rows to store before processing.
        delta_time (timedelta): The time interval for resampling data.
        name_key (int): A counter for naming data chunks written to the file.
        complevel (int): The compression level for storing data in HDF5 format.
        columns (list): The columns of the data being processed.
        raw_data_storage (pd.DataFrame): A DataFrame to temporarily store raw data.
        processed_data_storage (pd.DataFrame): A DataFrame to store processed data.
    """
    def __init__(self,file_name:str,
                 meta_data:pd.DataFrame,
                 w_unprocessed_data:bool,
                 delta_time:int,
                 max_rows:int=200000,
                 complevel:int=1) -> None:
        """Initializes the DataProcessor with the given parameters.

        Args:
            file_name (str): The name of the file to store the processed data.
            meta_data (pd.DataFrame): The metadata for the simulation.
            w_unprocessed_data (bool): A flag indicating if raw data should be stored without processing.
            delta_time (int): The time interval for resampling data (in minutes).
            max_rows (int, optional): The maximum number of rows to store before processing. Defaults to 200000.
            complevel (int, optional): The compression level for storing data in HDF5 format. Defaults to 1.
        """
        self.file_name=file_name
        self.w_unprocessed_data=w_unprocessed_data
        self.max_rows=max_rows
        self.delta_time=timedelta(minutes=delta_time)
        self.name_key=0
        self.complevel=complevel
        self.columns=['unique_id', 'timestamp', 'value']
        self.raw_data_storage = []
        self.processed_data_storage = []
        self.start_file(meta_data)

    def receive_data(self, unique_id, timestamp, value) -> None:
        """Receives and stores incoming data. Processes the data if the storage exceeds the max_rows limit.

        Args:
            unique_id (int): The unique identifier for the data entry.
            timestamp (pd.Timestamp): The timestamp of the data entry.
            value (float): The value of the data entry.
        """
        new_data = pd.DataFrame([[unique_id, timestamp, value]], columns=self.columns)
        if len(self.raw_data_storage) > 0:
            self.raw_data_storage = pd.concat([self.raw_data_storage, new_data], ignore_index=True)
        else:
            self.raw_data_storage = new_data.copy(deep=True)
        if len(self.raw_data_storage) >= self.max_rows:
            self.process_data()

    def process_data(self) -> None:       
        """
        Processes the stored raw data based on the w_unprocessed_data flag and resampling requirements.
        If the number of processed rows exceeds max_rows, the data is written to the file.
        """
        if self.w_unprocessed_data:
            if len(self.processed_data_storage) > 0:
                self.processed_data_storage = pd.concat([self.processed_data_storage, self.raw_data_storage.copy(deep=True)], ignore_index=True)
                self.processed_data_storage = self.processed_data_storage.drop_duplicates()
            else:
                self.processed_data_storage = self.raw_data_storage.drop_duplicates().copy(deep=True)
        else:
            # Set the index to be a MultiIndex of 'unique_id' and 'timestamp'
            self.raw_data_storage.set_index(['timestamp'], inplace=True)
            # Resample and aggregate data
            resampled_data = self.raw_data_storage.groupby('unique_id').resample(self.delta_time).max()
            # Reset the index to have 'timestamp' as a column and 'unique_id' as the index
            resampled_data.reset_index(level=1,drop=False,inplace=True)
            resampled_data.reset_index(drop=True,inplace=True)
            # Append the resampled data to the processed data storage
            if len(self.processed_data_storage) > 0:
                self.processed_data_storage = pd.concat([self.processed_data_storage, resampled_data], ignore_index=True)
            else:
                self.processed_data_storage = resampled_data.copy(deep=True)
                # self.processed_data_storage.reset_index(drop=True,inplace=True)
        # Clear the raw data storage to free up memory
        self.raw_data_storage = []
        # Test is process data has reached the maximum row size
        if len(self.processed_data_storage) >= self.max_rows:
            self.to_file()
    
    def start_file(self,meta_data:pd.DataFrame) -> None:
        """Initializes the HDF5 file with metadata.

        Args:
            meta_data (pd.DataFrame): The metadata to be stored in the file.
        """
        meta_data.to_hdf(self.file_name, key=f'/meta_data', mode='w', index=False, complevel=self.complevel)

    def to_file(self) -> None:
        """
        Writes the processed data to the HDF5 file and increments the name_key. Clears the processed data storage.
        """
        self.processed_data_storage.to_hdf(self.file_name, key=f'/a{self.name_key}', mode='a', index=False, complevel=self.complevel)
        self.name_key += 1
        # Clear the process data storage to free up memory
        self.processed_data_storage = []

    def close_data_processor(self) -> None:
        """
        Processes any remaining data and writes it to the file before closing.
        """
        self.process_data()
        self.to_file()

    def get_processed_data(self) -> pd.DataFrame:
        """Returns the currently stored processed data.

        Returns:
            pd.DataFrame: A DataFrame containing the processed data.
        """
        return self.processed_data_storage

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor(file_name='test.h5',meta_data=pd.DataFrame({'oi':[1,2,2],'ola':[5,5,5]}),w_unprocessed_data=True)

    # Receiving some data
    processor.receive_data('sensor_1', datetime(2023, 6, 24, 12, 0), 10)
    processor.receive_data('sensor_1', datetime(2023, 6, 24, 12, 5), 15)
    processor.receive_data('sensor_1', datetime(2023, 6, 24, 12, 10), 8)
    processor.receive_data('sensor_2', datetime(2023, 6, 24, 12, 0), 20)
    processor.receive_data('sensor_2', datetime(2023, 6, 24, 12, 5), 25)

    # Process data with a delta time of 10 minutes
    processor.process_data()

    # Print the processed data
    processor.processed_data_storage.reset_index(drop=True,inplace=True)
    print(processor.get_processed_data())
    len(processor.processed_data_storage)
    len(processor.processed_data_storage.index.unique())
    processor.to_file()

    # large test
    temp=timedelta(minutes=5)
    for i in range(500000):
        processor.receive_data('sensor_1', datetime(2023, 6, 24, 12, 0)+temp*i, 10)
    