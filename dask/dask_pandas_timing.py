import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import gc
import time

pd.options.mode.copy_on_write = True

dtypes = {
    'object_id': np.int32,
    'mjd': np.float64,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.bool_
}

passband = {
    0: 'ultraviolet',
    1: 'green',
    2: 'red',
    3: 'infrared',
    4: 'near-z',
    5: 'near-Y'
}

ALL_ROWS = 453_653_105

def sample_dataframe(file, sample_size):
    # Load file
    df_full = dd.read_csv(file, dtype=dtypes)
    
    # Sample DataFrame
    if (sample_size > 10_000_000):
        return df_full.sample(frac=(sample_size / ALL_ROWS)).persist()
    else:
        return df_full.sample(frac=(sample_size / ALL_ROWS)).compute()
    
    
def timed_func(df, func, nrows):
    start = time.perf_counter_ns()
    
    if func == 'add':
        result = df['flux'] + df['flux_err']
    elif func == 'mul':
        result = df['flux'] * df['flux_err']
    elif func == 'cumsum':
        result = df['flux'].cumsum()
    elif func == 'abs':
        result = df['flux'].abs()
    elif func == 'mean':
        result = df['flux'].mean()
    elif func == 'std':
        result = df['flux'].std()
    elif func == 'count':
        result = df['flux'].count()
    elif func == 'unique':
        result = df['flux'].unique()
    
    if nrows > 10_000_000:
        _ = result.compute()
    
    end = time.perf_counter_ns()
    print(f'Elapsed time: {end - start:.2f} ns')
    

if __name__ == "__main__":
    # Start Dask cluster/client
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    
    print("Dask Dashboard URL:", client.dashboard_link)
    
    # for nrows in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    for nrows in [100_000_000]:
        # Load sample
        df = sample_dataframe('data/test_set.csv', nrows)
    
        result = (df['flux'] + df['flux_err']).max().compute()
        print(result)
    
        # for func in ['add', 'mul', 'cumsum', 'abs', 'mean', 'std', 'count', 'unique']:
        #     for i in range(5):
        #         print(f'{nrows} rows, {func}, {i + 1}/5')
        #         timed_func(df, func, nrows)    
                
        #     print('\nEND FUNC\n')
        
        # print('\nEND ROW\n')