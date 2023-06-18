import pandas as pd
import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='experiments_with_ltcs/data/stations_data')
    parser.add_argument('-o', '--output_dir', type=str, default='experiments_with_ltcs/data/stations_data_preprocessed')
    args = parser.parse_args()

    input_files = os.listdir(args.input_dir)

    #create output dir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for input_file in input_files:
        if not input_file.endswith('.csv'):
            continue
        
        input_path = os.path.join(args.input_dir, input_file)
        output_file = input_file.replace('.csv', '_preprocessed.csv')
        output_path = os.path.join(args.output_dir, output_file)
        
        data_ecl = pd.read_csv(input_path, parse_dates=True, index_col=0)

        #rename bikes column to value
        data_ecl = data_ecl.rename(columns={data_ecl.columns[0]: "value"})

        print(data_ecl.shape)
        print(data_ecl.head())
        # data_ecl = data_ecl.resample('1h', closed='right').mean()

        print(data_ecl.shape)
        print(data_ecl.head())

        #print nan
        print(data_ecl.isnull().sum())
        missing_values = data_ecl.isnull().sum()
        total_values = np.product(data_ecl.shape)
        percentage_missing = (missing_values.sum() / total_values) * 100
        print("Percentage of missing values in the DataFrame: ", percentage_missing, "%")

        # data_ecl.index = data_ecl.index.rename('date')
        # data_ecl = data_ecl.rename(columns={data_ecl.columns[0]: "value"})

        # handle missing values
        data_ecl.dropna(inplace=True)

        # create a boolean series where the value is True if the data point is missing (i.e., a dropout)
        dropouts = data_ecl['value'].isnull()

        # find the start and end times of the dropouts
        dropout_start_times = data_ecl.index[dropouts.diff() == True]
        dropout_end_times = data_ecl.index[dropouts.diff() == -1]

        # handle case when time series ends with a dropout
        if dropouts.iloc[-1]:
            dropout_end_times = dropout_end_times.append(pd.Index([data_ecl.index[-1]]))

        # calculate the number of dropouts
        num_dropouts = len(dropout_start_times)

        # print the start and end dates of each dropout period
        for start in dropout_start_times:
            end = dropout_end_times[start <= dropout_end_times].min()
            print(f"Start Date: {start}, End Date: {end.date()}")

        # print the number of dropouts
        print(f"Number of Dropouts: {num_dropouts}")

        # remove nan rows
        data_ecl.dropna(inplace=True)

        # select columns to keep
        columns_to_keep = ['value', 'temp_c', 'is_day', 'wind_kph', 'cloud', 'precip_mm', 'humidity', 'will_it_rain',
                           'bikes_plus_120mins', 'bikes_plus_60mins']
        data_ecl = data_ecl[columns_to_keep]

        data_ecl.to_csv(output_path)
