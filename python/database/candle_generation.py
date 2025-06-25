import datetime
import os

import mlfinlab
import pandas as pd
import numpy as np

from utils.pandas_utils.dataframe_utils import reduce_memory_usage


def generate_candle_time(
        df, resolution='MIN', num_units=1, batch_size: int = int(os.getenv("CANDLES_BATCH_SIZE", 10000))
) -> pd.DataFrame:
    '''

    :param df:
    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :return: dataframe of the candles where datetime is the closing time
    '''
    df = reduce_memory_usage(df)
    # reduce memory size before processing
    output_df = mlfinlab.data_structures.get_time_bars(
        file_path_or_df=df.reset_index(),
        resolution=resolution,
        num_units=num_units,
        batch_size=batch_size,
        verbose=False,
    )
    output_df = reduce_memory_usage(output_df)
    # output_df['date'] = pd.to_datetime(output_df['date_time']* 1000000000)
    # fillnas like in candle publisher
    last_close = output_df['close'].ffill().shift(1)
    for column in ['open', 'high', 'low', 'close']:
        output_df[column] = output_df[column].fillna(last_close)

    output_df['volume'] = output_df['volume'].fillna(0.0)
    output_df['cum_dollar_value'] = output_df['cum_dollar_value'].fillna(0.0)
    output_df['cum_ticks'] = output_df['cum_ticks'].fillna(0.0)

    # set datetime as timestamp[ns]
    output_df['datetime'] = pd.to_datetime(output_df['date_time'] * 1000000000, unit='ns')
    # if resolution == 'D':
    #     output_df['datetime'] = output_df['datetime'] - datetime.timedelta(days=num_units)


    output_df.set_index('datetime', inplace=True)
    output_df.sort_index(inplace=True)
    return output_df


def generate_candle_tick(df, number_of_ticks=5) -> pd.DataFrame:
    df = reduce_memory_usage(df)
    output_df = mlfinlab.data_structures.get_tick_bars(
        file_path_or_df=df.reset_index(), threshold=number_of_ticks
    )
    output_df.set_index('date_time', inplace=True)
    return output_df


def generate_candle_volume(df, volume=5000):
    df = reduce_memory_usage(df)  # reduce memory size before processing
    output_df = mlfinlab.data_structures.get_volume_bars(
        file_path_or_df=df.reset_index(), threshold=volume
    )
    output_df.set_index('date_time', inplace=True)
    return output_df


def generate_candle_dollar_value(df, dollar_value=5000):
    df = reduce_memory_usage(df)  # reduce memory size before processing
    output_df = mlfinlab.data_structures.get_dollar_bars(
        file_path_or_df=df.reset_index(), threshold=dollar_value
    )
    output_df.set_index('date_time', inplace=True)
    return output_df


if __name__ == '__main__':
    # test generate_candle_time
    from database.tick_db import TickDB

    tick = TickDB()
    instrument_pk = 'btceur_kraken'
    start_date = datetime.datetime(day=10, month=5, year=2024)
    end_date = datetime.datetime(day=10, month=5, year=2024)
    depth_df = tick.get_depth(instrument_pk=instrument_pk, start_date=start_date, end_date=end_date, first_hour=7,
                              last_hour=15)

    candles_df = generate_candle_time(depth_df, resolution='MIN', num_units=5)
    print(candles_df.head())
