import numpy as np
import pandas as pd

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

def sample_dataframe(df, sample_size):
    # Return sample
    return df.sample(n=sample_size)
    
    
def annotate(df, col, num_bins):
    # Annotate data by splitting into equal sized bins
    # Each bin gets an integer code
    df[f'{col}_annot'] = pd.qcut(df[col], q=num_bins)
    return df


if __name__ == '__main__':
    print('Using data:', DATA)
    
    # Load full data
    df_full = pd.read_csv(DATA, dtype=dtypes)
    
    for nrows in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
        # Load sample
        print(f'Using rows: {nrows}')
        df = sample_dataframe(df_full, nrows)
        
        # Print unannotated DataFrame size
        print(f'Unannotated DataFrame size: {df.memory_usage().sum()} bytes')
    
        for nbins in [10, 100, 1000, 10_000, 100_000, 1_000_000]:
            if nbins >= nrows:
                continue
    
            # Annotate
            for i in range(5):
                print(f'{nrows} rows, {nbins} bins, {i + 1}/5')
                start = time.perf_counter_ns()
                
                annotate(df, 'flux', nbins)
                
                end = time.perf_counter_ns()
                print(f'Time to annotate: {end - start:.2f} ns')
            
            # Print size of annotated DataFrame
            print(f'Annotated DataFrame size: {df.memory_usage().sum()} bytes')
                
            print('\nEND BIN\n')
                
        print('\nEND ROW\n')