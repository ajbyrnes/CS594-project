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

DATA = 'data/test_set_batch1.csv'
ALL_ROWS = 10_855_959

def sample_dataframe(file, sample_size):
    # Load file
    df_full = pd.read_csv(file, dtype=dtypes)
    
    # Return sample
    return df_full.sample(n=sample_size)
    
    
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
    
    end = time.perf_counter_ns()
    print(f'Elapsed time: {end - start:.2f} ns')
    
    
if __name__ == '__main__':
    print('Using data:', DATA)
    
    for nrows in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
        # Load sample
        print(f'Using rows: {nrows}')
        df = sample_dataframe(DATA, nrows)
    
        for func in ['add', 'mul', 'cumsum', 'abs', 'mean', 'std', 'count', 'unique']:
            for i in range(5):
                print(f'{nrows} rows, {func}, {i + 1}/5')
                timed_func(df, func, nrows)    
                
            print('\nEND FUNC\n')
        
        print('\nEND ROW\n')