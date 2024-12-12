import numpy as np
import pandas as pd

import timeit

pd.options.mode.copy_on_write = True

BATCH1_ROWS = 10_855_959

dtypes = {
    'object_id': np.int32,
    'mjd': np.float64,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.bool_
}


def annotate(df, col, num_bins):
    # Annotate data
    df[f'{col}_annot'] = pd.cut(df[col], bins=num_bins)
    df[f'{col}_annot_int'] = df[f'{col}_annot'].astype('category').cat.codes
    return df

# Filter data with sketch and annotated column
def filter1a(df, sketch):
    cond = df['mjd_annot'].cat.codes == sketch[0]
    for i in sketch[1:]:
        cond |= df['mjd_annot'].cat.codes == i
    return df[cond]

# Filter data with sketch and integer column
def filter1b(df, sketch):
    cond = df['mjd_annot_int'] == sketch[0]
    for i in sketch[1:]:
        cond |= df['mjd_annot_int'] == i
        
    return df[cond]

def filter2a(df, sketch):
    cond = np.equal(df['mjd_annot'].cat.codes.values, sketch[0])
    for i in sketch[1:]:
        cond |= np.equal(df['mjd_annot'].cat.codes.values, i)
        
    return df[cond]

def filter2b(df, sketch):
    cond = np.equal(df['mjd_annot_int'].values, sketch[0])
    for i in sketch[1:]:
        cond |= np.equal(df['mjd_annot_int'].values, i)
        
    return df[cond]

def filter3a(df, sketch):
    cond = df['mjd_annot'].cat.codes == sketch[0]
    for i in sketch[1:]:
        cond |= df['mjd_annot'].cat.codes == i
        
    return df.loc[cond]
    
def filter3b(df, sketch):
    cond = df['mjd_annot_int'] == sketch[0]
    for i in sketch[1:]:
        cond |= df['mjd_annot_int'] == i
        
    return df.loc[cond]
    
def filter4a(df, sketch):
    cond = ' or '.join([f'mjd_annot.cat.codes == {i}' for i in sketch])
    return df.query(cond, engine='numexpr')

def filter4b(df, sketch):
    cond = ' or '.join([f'mjd_annot_int == {i}' for i in sketch])
    return df.query(cond, engine='numexpr')


for num_rows in [10, 100, 1000, 10_000, 100_000, 1_000_000, BATCH1_ROWS]:
    print("NUM ROWS:", num_rows)
    
    # Load data
    df = pd.read_csv('data/test_set_batch1.csv', nrows=num_rows, dtype=dtypes)

    # Annotate data
    df = annotate(df, 'mjd', 10)
    
    # Create dummy sketches
    sketches = [np.random.permutation(df['mjd_annot'].cat.codes.unique())[:i] for i in [1, 2, 4, 8]]

    for sketch in sketches:
        print("SKETCH:", sketch)

        # Filter 1: df.isin(sketch)
        results_sec = timeit.repeat('filter1a(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter1a size: {filter1a(df, sketch).shape}')
        print(f'filter1a time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')

        results_sec = timeit.repeat('filter1b(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter1b size: {filter1b(df, sketch).shape}')
        print(f'filter1b time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')

        # Filter 2: np.isin(df.values, sketch)
        results_sec = timeit.repeat('filter2a(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter2a size: {filter2a(df, sketch).shape}')
        print(f'filter2a time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')

        results_sec = timeit.repeat('filter2b(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter2b size: {filter2b(df, sketch).shape}')
        print(f'filter2b time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')

        # Filter 3: df.loc[df.isin(sketch)]
        results_sec = timeit.repeat('filter3a(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter3a size: {filter3a(df, sketch).shape}')
        print(f'filter3a time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
        
        results_sec = timeit.repeat('filter3b(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter3b size: {filter3b(df, sketch).shape}')
        print(f'filter3b time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
        
        # Filter 4: df.query('isin(sketch)')
        results_sec = timeit.repeat('filter4a(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter4a size: {filter4a(df, sketch).shape}')
        print(f'filter4a time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
            
        results_sec = timeit.repeat('filter4b(df, sketch)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter4b size: {filter4b(df, sketch).shape}')
        print(f'filter4b time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
        
        print("\n**********\n")
