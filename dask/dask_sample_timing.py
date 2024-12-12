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
        _ = df_full.sample(frac=(sample_size / ALL_ROWS)).persist()
    else:
        _ = df_full.sample(frac=(sample_size / ALL_ROWS)).compute()
    

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    
    print("Dask Dashboard URL:", client.dashboard_link)

    for nrows in [1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        print(f'Sampling rows: {nrows}')
        times = []
        
        for i in range(5):
            print(f'{nrows} rows, {i+1}/5')
            start = time.perf_counter()
            sample_dataframe('data/test_set.csv', 1000)
            end = time.perf_counter()
            print(f'Elapsed time: {end - start:.2f} seconds')
            times.append(end - start)
            

        best_time = np.min(times)
        print(f'Best time: {best_time:.2f} seconds')
        print("\nEND ROW\n")
        
    
    input("Tasks finished, press any key to shutdown the cluster...")
    client.close()
    cluster.close()