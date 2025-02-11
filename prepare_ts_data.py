#
# (0) imports
#
import os
import pandas as pd
import numpy as np
import json

try:
    import plotly
except ImportError:
    os.system("pip install plotly")
    import plotly

import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

from plot_utils import apply_plot_styling  # Importing from new mod

#
# (1) imports
#
class TimeSeriesDataHandler:

    # Constants
    DEFAULT_START_DATE = '2010-01-01'
    DEFAULT_END_DATE = '2019-12-31'
    DEFAULT_TEST_PERCENTAGE = 0.5
    DEFAULT_DATASET_INFO = "Not provided for this dataset"
    DEFAULT_TIME_SERIES_LABEL = "Time Series"
    PROBLEM_TYPE = "forecasting"

    # Initialization
    def __init__(self,
                 dataset_folder,
                 start_date=None,
                 end_date=None,
                 test_percentage=None,
                 dataset_info="No summary info provided for this dataset",
                 dataset_title="No title provided for this dataset",
                 forecast_horizon_in_days=30,
                 backsight_in_days=None):

        self.dataset_folder = os.path.join(dataset_folder)  # Pointing to the 'transformed' subfolder
        self.metadata = self._load_metadata()  # Load other metadata keys as instance variables

        self.metadata["title"] = dataset_title

        self.data = None
        self.start_date = start_date if start_date is not None else self.DEFAULT_START_DATE
        self.end_date = end_date if end_date is not None else self.DEFAULT_END_DATE
        self.test_percentage = test_percentage if test_percentage is not None else self.DEFAULT_TEST_PERCENTAGE

        self.dataset_title = dataset_title
        self.dataset_info = dataset_info if dataset_info is not None else self.DEFAULT_DATASET_INFO
        self.time_series_label = self.DEFAULT_TIME_SERIES_LABEL
        self.available_event_codes = self._list_available_event_codes()


        self.forecast_horizon_in_days = forecast_horizon_in_days
        self.forecast_horizon_in_intrinsec_scale = None
        self.backsight_in_days = backsight_in_days
        self.backsight_in_intrinsec_scale = None

        self.baseline_model = None
        self.train_sample = None
        self.predictive_model = None

    #
    def _load_metadata(self):
        metadata_path = os.path.join(self.dataset_folder, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    #
    def _list_available_event_codes(self):
        metadata = self.metadata
        codes_to_labels = metadata.get('codes_to_labels', {})

        if not codes_to_labels:
            raise ValueError("No 'codes_to_labels' found in metadata.")

        # Invert the dictionary to map labels to codes
        return {label: code for code, label in codes_to_labels.items()}


    #
    # loading time series data from disk
    #

    #
    def load_data_from_event_code(self, event_code):
        matching_files = [file for file in os.listdir(self.dataset_folder) if event_code in file and file.endswith('.csv')]
        if not matching_files:
            raise ValueError(f"No CSV files found matching event code: {event_code}")

        try:
            dfs = []
            for file in matching_files:
                file_path = os.path.join(self.dataset_folder, file)
                df = pd.read_csv(file_path)
                date_column = 'date' if 'date' in df.columns else 'datetime' if 'datetime' in df.columns else None
                if date_column is None:
                    raise ValueError(f"CSV file '{file}' does not contain 'date' or 'datetime' columns.")
                df[date_column] = pd.to_datetime(df[date_column])
                dfs.append(df.set_index(date_column))

            self.data = pd.concat(dfs, ignore_index=False).sort_index()
            self.data.dropna(inplace=True)

            self.time_series_label = next((key for key, value in self.available_event_codes.items() if value == event_code), self.DEFAULT_TIME_SERIES_LABEL)

            self.date_col = self._detect_date_col(allowed_date_col_names=None)
            self.frequency = self._detect_frequency()  # Detect frequency after setting date_col

            # Use initialized forecast horizon
            self.forecast_horizon_in_intrinsec_scale = self.convert_days_to_frequency_units(self.forecast_horizon_in_days)

            # Compute or use initialized backsight
            if self.backsight_in_days is None:
                self.backsight_in_days = self._compute_backsight_in_days()  # Call the heuristic method
            self.backsight_in_intrinsec_scale = self.convert_days_to_frequency_units(self.backsight_in_days)

        except Exception as e:
            raise ValueError(f"Error during loading process: {str(e)}")
    #
    def _detect_date_col(self, allowed_date_col_names):
        if allowed_date_col_names is None:
            allowed_date_col_names = ['date', 'datetime']  # Default values
        for file in os.listdir(self.dataset_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(self.dataset_folder, file)
                try:
                    df = pd.read_csv(file_path, nrows=1)  # Read only the first row for efficiency
                    for col in allowed_date_col_names:
                        if col in df.columns:
                            return col
                except Exception as e:
                    raise ValueError(f"Error reading '{file}' with exception: {e}")
        raise ValueError("No appropriate date column found")

    #
    def get_forecast_parameters(self):
        # Check if the necessary attributes are set
        if self.frequency is None or self.forecast_horizon_in_days is None or \
        self.forecast_horizon_in_intrinsec_scale is None or \
        self.backsight_in_days is None or self.backsight_in_intrinsec_scale is None:
            return {"error": "Forecast parameters are not set. Please load the data first."}

        # Construct the dictionary with parameters
        forecast_params = {
            "frequency": self.frequency,
            "forecast_horizon_in_days": self.forecast_horizon_in_days,
            "forecast_horizon_in_intrinsec_scale": self.forecast_horizon_in_intrinsec_scale,
            "backsight_in_days": self.backsight_in_days,
            "backsight_in_intrinsec_scale": self.backsight_in_intrinsec_scale
        }
        #
        return forecast_params

    def _detect_frequency(self):
        if self.data is None or self.date_col is None:
            raise ValueError("Data has not been loaded or date column has not been set")
        if not self.data.index.is_monotonic_increasing:
            raise ValueError("Index is not set to a datetime or is not sorted")
        return self.data.index.to_series().diff().value_counts().idxmax()
    #
    def convert_days_to_frequency_units(self, days):
        """
        Convert days to the equivalent number of frequency units based on self.frequency.
        """
        if self.frequency is None:
            raise ValueError("Frequency has not been detected. Load data and set frequency before converting days.")
        # Assuming self.frequency is a Pandas Timedelta object or similar
        return int(round((pd.to_timedelta(days, unit='D') / self.frequency)))
    #
    def _compute_backsight_in_days(self):
        if self.frequency is None:
            raise ValueError("Frequency has not been detected.")

        # Convert frequency to a total number of seconds for comparison
        freq_in_seconds = self.frequency.total_seconds()

        # Intra-hour data (less than 3600 seconds)
        if freq_in_seconds < 3600:
            return 7  # 7 days
        # Up to daily data (3600 seconds to 86400 seconds)
        elif freq_in_seconds <= 86400:
            return 10  # 15 days
        # Up to monthly data (more than a day but less than or equal to 30 days)
        elif freq_in_seconds <= 2592000:
            return 60  # 60 days
        # Monthly data or slower (more than 30 days)
        else:
            return 365  # 365 days


    def define_testing_policy(self, start_date=None, end_date=None, test_percentage=None):
        if self.data is None:
            raise ValueError("Data not loaded. Load data before defining a testing policy.")

        # Use method parameters if provided, otherwise use instance attributes
        start_date = pd.Timestamp(start_date if start_date is not None else self.start_date)
        end_date = pd.Timestamp(end_date if end_date is not None else self.end_date)
        test_percentage = test_percentage if test_percentage is not None else self.test_percentage

        df = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        num_test_samples = int(len(df) * test_percentage)

        df['sample'] = 'train'
        df.loc[df.tail(num_test_samples).index, 'sample'] = 'test'
        self.data = df
        return self.data['sample'].value_counts()

    def make_samples(self, sample_type='train', output_sequence=False, value_col='value'):
        if self.data is None:
            raise ValueError("Data has not been loaded.")
        if self.date_col is None:
            raise ValueError("Date column has not been set.")
        if self.frequency is None:
            raise ValueError("Frequency has not been detected.")

        if sample_type == 'train':
            data_sample = self.data[self.data['sample'] != 'test']
        elif sample_type == 'test':
            data_sample = self.data[self.data['sample'] == 'test']
        else:
            raise ValueError("sample_type must be either 'train' or 'test'.")

        back_sight = self.convert_days_to_frequency_units(self.backsight_in_days)
        fore_sight = self.convert_days_to_frequency_units(self.forecast_horizon_in_days)

        X, y = [], []

        if sample_type == 'train':
            for i in range(back_sight, len(data_sample) - fore_sight):
                X.append(data_sample.iloc[i - back_sight:i][value_col].values)
                if output_sequence:
                    y.append(data_sample.iloc[i:i + fore_sight][value_col].values)
                else:
                    y.append(data_sample.iloc[i + fore_sight - 1][value_col])
        elif sample_type == 'test':
            #
            start_index = len(self.data) - len(data_sample) - fore_sight
            end_index = len(self.data) - fore_sight
            #
            for i in range(start_index, end_index):
                X.append(self.data.iloc[i - back_sight:i][value_col ].values)
                if output_sequence:
                    y.append(self.data.iloc[i:i + fore_sight][value_col ].values)
                else:
                    y.append(self.data.iloc[i + fore_sight][value_col])  # No .values needed here

        X, y = np.array(X), np.array(y)
        return X, y
    #
    def get_train_data(self, **kwargs):
        return self.make_samples(sample_type='train', **kwargs)
    #
    def get_test_data(self, **kwargs):
        return self.make_samples(sample_type='test', **kwargs)

    #
    def plot(self, preds=None):
        """
        This methods is responsible for the plots of historical and forecasted values shown to end-users, wether in a server or UX access
        """
        #
        # (0.1) data need to be loaded to perform a plot
        #
        if self.data is None or self.data.empty:
            raise ValueError("No data to plot. Load data first.")
        #
        # (0.2) data needs to be indexed, and assigned to an experimental case
        #
        if 'sample' not in self.data.columns or 'value' not in self.data.columns:
            raise ValueError("Data does not contain required columns 'sample' and 'value'.")

        train_data = self.data[self.data['sample'] == 'train']
        test_data = self.data[self.data['sample'] == 'test']


        # Setting the dynamic title
        if preds is None:
            title = f"Historical values and sample split for time series '{self.time_series_label}'"
        else:
            title = f"Historical values, sample split, and optimal forecasts for time series '{self.time_series_label}'"
        #
        # Forecast line
        forecast_label = "Naive Forecast" if preds is None else "Forecast"


        # Use base_forecast with lagged_rolling_window method if no predictions are provided
        if preds is None:
            preds = self.base_forecast(method='lagged_rolling_window', back_sight_days=28*5, lag_days=30)

        # The rest of your existing plot method code goes here
        fig = go.Figure()


        # Default style
        for dataset, color, name in [(train_data, 'blue', 'Train'), (test_data, 'red', 'Test')]:
            # Line trace
            fig.add_trace(go.Scatter(
                x=dataset.index,
                y=dataset['value'],
                mode='lines',
                name=name,
                line=dict(shape='linear', color=color, width=1.4),
                opacity=0.7  # Set opacity for line
            ))

            # Marker trace
            fig.add_trace(go.Scatter(
                x=dataset.index,
                y=dataset['value'],
                mode='markers',
                name=name,
                marker=dict(color=color, size=7 * 1.5, opacity=0.15),  # 1.5 times bigger and 50% opacity for markers
                showlegend=False  # Prevent duplicate legend entries
            ))


        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=preds,
            mode='lines',
            name=forecast_label,
            line=dict(color='green', width=3.0, dash='dash'),
            #opacity=1  # Set opacity for line
        ))

        fig.update_layout(title=title)
        # Apply consistent styling (assuming this is a method in your class)
        fig = apply_plot_styling(fig)
        fig.show()
        return fig
    #
    def base_forecast(self,
                      method='average_train',
                      back_sight_days=60,
                      lag_days=30
                      ):
        """
        Generates forecasts using different methods.
        'average_train' replicates the average of training data.
        'lagged_rolling_window' uses a rolling window on the lagged series.

        Parameters:
        - method: The forecasting method to use ('average_train' or 'lagged_rolling_window').
        - back_sight_days: Number of days for the rolling window.
        - lag_days: Number of days to lag the series.
        """
        if self.data is None or 'sample' not in self.data.columns or 'value' not in self.data.columns:
            raise ValueError("Data not loaded or required columns are missing.")

        if method == 'average_train':
            train_data = self.data[self.data['sample'] == 'train']
            average_train = np.mean(train_data['value'])
            test_data = self.data[self.data['sample'] == 'test']
            return np.full(len(test_data), average_train)

        elif method == 'lagged_rolling_window':
            # Ensure frequency is set
            if self.frequency is None:
                raise ValueError("Frequency has not been detected. Load data and set frequency before forecasting.")

            # Convert days to frequency units and ensure they are integers
            back_sight = self.convert_days_to_frequency_units(back_sight_days)
            lag = self.convert_days_to_frequency_units(lag_days)

            if back_sight is None or lag is None:
                raise ValueError("Error in converting days to frequency units.")

            # Ensure the values are integers
            back_sight = int(back_sight)
            lag = int(lag)

            # Apply rolling window on the lagged series
            rolled_series = self.data['value'].shift(lag).rolling(window=back_sight).mean()

            # Extract the forecast for the test period
            test_data = self.data[self.data['sample'] == 'test']
            return rolled_series.loc[test_data.index].fillna(method='bfill').values

        else:
            raise ValueError(f"Unknown forecasting method: {method}")

    #
    # 0.
    #
    @classmethod
    def validate_etl(cls, dataset_folder):

        transformed_path = os.path.join(dataset_folder, 'transformed')
        metadata_path = os.path.join(transformed_path, 'metadata.json')

        if not os.path.isfile(metadata_path):
            print(f"Validation failed: 'metadata.json' not found in {dataset_folder}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
                    print("Validation failed: 'metadata.json' format is incorrect")
                    return False
        except json.JSONDecodeError:
            print("Validation failed: 'metadata.json' is not valid JSON")
            return False

        csv_files = [f for f in os.listdir( transformed_path) if f.endswith('.csv')]
        if not csv_files:
            print("Validation failed: No CSV files found in the dataset folder.")
            return False

        for file_name in csv_files:
            file_path = os.path.join( transformed_path, file_name)
            try:
                df = pd.read_csv(file_path)
                date_column = 'date' if 'date' in df.columns else 'datetime' if 'datetime' in df.columns else None
                if date_column is None:
                    print(f"Validation failed: CSV file '{file_name}' does not contain 'date' or 'datetime' columns.")
                    return False
            except Exception as e:
                print(f"Validation failed: Error reading '{file_name}' with exception: {e}")
                return False

        print("ETL Validation Passed.")
        return True
