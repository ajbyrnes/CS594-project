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

def filter1(df, sketches, cols):
    cond = df[cols[0]].cat.codes.isin(sketches[0])
    for col, sketch in zip(cols[1:], sketches[1:]):
        cond &= df[col].cat.codes.isin(sketch)
        
    return df[cond]

def filter2(df, sketches, cols):
    cond = np.isin(df[cols[0]].cat.codes.values, sketches[0])
    for col, sketch in zip(cols[1:], sketches[1:]):
        cond &= np.isin(df[col].cat.codes.values, sketch)
        
    return df[cond]

def filter3(df, sketches, cols):
    cond = df[cols[0]].cat.codes.isin(sketches[0])
    for col, sketch in zip(cols[1:], sketches[1:]):
        cond &= df[col].cat.codes.isin(sketch)
        
    return df.loc[cond]
    
def filter4(df, sketches, cols):
    cond = ' and '.join(f'{col}.cat.codes in @sketches[{i}]' for i, col in enumerate(cols))
    return df.query(cond)

for num_rows in [10, 100, 1000, 10_000, 100_000, 1_000_000, BATCH1_ROWS]:
    print("NUM ROWS:", num_rows)
    
    # Load data
    df = pd.read_csv('data/test_set_batch1.csv', nrows=num_rows, dtype=dtypes)
    
    # Annotate data
    df = annotate(df, 'mjd', 10)
    df = annotate(df, 'flux', 10)
        
    # Create dummy sketches
    sketches1 = [np.random.permutation(df['mjd_annot'].cat.codes.unique())[:i] for i in [1, 2, 4, 8]]
    sketches2 = [np.random.permutation(df['flux_annot'].cat.codes.unique())[:i] for i in [1, 2, 4, 8]]
    
    cols = ['mjd_annot', 'flux_annot']    
    for sk1, sk2 in zip(sketches1, sketches2):
        print("SKETCH:", sk1, '+', sk2)
        
        # Filter 1: df.isin(sketch)
        results_sec = timeit.repeat('filter1(df, [sk1, sk2], cols)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter1 size: {filter1(df, [sk1, sk2], cols).shape}')
        print(f'filter1 time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')

        # Filter 2: np.isin(df.values, sketch)
        results_sec = timeit.repeat('filter2(df, [sk1, sk2], cols)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter2 size: {filter2(df, [sk1, sk2], cols).shape}')
        print(f'filter2 time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
        
        # Filter 3: df.loc[df.isin(sketch)]
        results_sec = timeit.repeat('filter3(df, [sk1, sk2], cols)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter3 size: {filter3(df, [sk1, sk2], cols).shape}')
        print(f'filter3 time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
        
        # Filter 4: df.query('isin(sketch)')
        results_sec = timeit.repeat('filter4(df, [sk1, sk2], cols)', globals=globals(), number=1, repeat=100)
        results_ms = np.array(results_sec) * 1000
        print(f'filter4 size: {filter4(df, [sk1, sk2], cols).shape}')
        print(f'filter4 time: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')
            
        print("\n**********\n")
