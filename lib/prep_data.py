import os
import pandas as pd
import numpy as np

def prepare_data(data_dir, features, resolution, batch_size, seq_len, **kwargs):
    """loads, selects, resamples and scales timeseries.

    :param data_dir: directory in which database is contained
    :type data_dir: path string

    :param features: features required for training
    :type features: list

    :param resolution: desired time distance between consecutive points
    :type resolution: string (corresponding key values of pd.resample method)

    :param batch_size: intended batch size for training
    :type batch_size: int

    :param seq_len: intenden number of time steps per sample
    :type seq_len: int

    :return: np_data: scaled dateset with desired features
    :rtype: np.array, shape == (# batches, batch_size, # steps, features)

    .. notes:: The data loaded using this method should be timeseries data
            with no holes or gaps; In other words the time difference
            between consecutive steps of the time-series should be
            constant. If there is more then one file in data_dir, it
            is assumed that they are indexed such that os.listdir
            iterates through them in time_sequential order. It is
            further assumed that the time difference between the first
            step of one file and the first step of the preceding file
            is equal to the time difference between consecutive steps
            within a singular file.
    """
    dim = (batch_size, seq_len, len(features))
    data = pd.DataFrame()

    #TODO: Change code to ensure that the data is always concatenated as desired
    for file in os.listdir(data_dir):

        file_path = os.path.join(data_dir, file)
        file_data = pd.read_csv(file_path,
                                index_col=0,
                                header=0,
                                parse_dates=True,
                                infer_datetime_format=True)

        data = pd.concat([data, file_data])

    if not features == [name for name in features if name in data.columns]:
        raise ValueError('The passed features do not all exist in the passed data')
    else:

        # Resample timestamped data
        data = data[features]
        data = data.resample(resolution, closed='right', label='right').mean()
        data = data.iloc[1:]

        # Scale data
        # ToDo: Implement the manual normalization of each of the columns through the config file.
        for column in data.columns:
            data[column], min, max = MinMaxScaler(data[column].values)

        # Reshape data to appropriate dimension (# batches, batch_size, # steps, # features)
        batch_samples = dim[0] * dim[1]
        batch_number = int(len(data) / batch_samples)
        np_data = data.iloc[:batch_number * batch_samples].values.reshape(batch_number, dim[0], dim[1], dim[2])

    return np_data

def MinMaxScaler(data):
    """Min-Max Normalizer.

    :param data: raw data
    :type data: np.array

    :return norm_data: normalized data
    :rtype norm_data: np.array

    :return min_val: minimum value of data
    :rtype min_val: float

    :return max_val: maximum value of data
    :rtype max_val: float

    .. notes:: It is only intended that vector-like data
    be passed to this function (e.g. (24, 1)).
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    norm_data = (data - min_val) / (max_val - min_val + 1e-7)

    return norm_data, min_val, max_val

if __name__ == "__main__":

    os.chdir("C:\\Users\\jmw\\Workspace\\th-e-gan\\databi")

    np_data = prepare_data('system', features=['pv_power'], dim=(128, 24), resolution='H')

    print(np_data.shape)