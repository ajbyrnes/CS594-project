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


def filter1(df, sketch):
    return df[df.index.isin(sketch, level=0)]

def filter2(df, sketch):
    return df.loc[df.index.isin(sketch, level=0)]

def filter3(df, sketch):
    return df[np.isin(df.index.get_level_values(0), sketch)]

def filter4(df, sketch):
    return pd.concat([df.xs(i) for i in sketch])

if __name__ == '__main__':
    print('Using data:', DATA)
    
    # Load full data
    df_full = pd.read_csv(DATA, dtype=dtypes)
    
    for nrows in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
        # Load sample
        print(f'Using rows: {nrows}')
        df = sample_dataframe(df_full, nrows)
    
        for nbins in [10, 100, 1000, 10_000, 100_000, 1_000_000]:
            if nbins >= nrows:
                continue
    
            # Annotate
            annotate(df, 'flux', nbins)
            
            # MultiIndex with annotations
            mi = pd.MultiIndex.from_arrays([df['flux_annot'].cat.codes, df['object_id']], names=['flux_cat', 'object_id'])
            df_mi = df.set_index(mi).sort_index()
            
            for filter, filter_func in [('filter1', filter1), ('filter2', filter2), ('filter3', filter3), ('filter4', filter4)]:
                for sketch_size in [1, 2, 4, 8]:
                    # Generate random sketch
                    sketch = np.random.permutation(df['flux_annot'].cat.codes.unique())[:sketch_size]
                
                    for i in range(5):
                        print(f'{nrows} rows, {nbins} bins, {filter} filter, {sketch_size} sketch, {i + 1}/5')
                        start = time.perf_counter_ns()
                        
                        filter_func(df_mi, sketch)
                        
                        end = time.perf_counter_ns()
                        print(f'Time to filter: {end - start:.2f} ns')
                    
                    print('\nEND SKETCH\n')
                    
                print('\nEND FILTER\n')
                
            print('\nEND BIN\n')
                
        print('\nEND ROW\n')