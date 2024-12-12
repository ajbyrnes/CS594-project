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


def test_query(df):
    return (df['flux'] + df['flux_err']).max()


def select_filter(df, query, sketch=None, multi=False):
    if multi:
        filtered = filter_df_multi(df, sketch)
        return query(filtered)
    elif sketch is not None:
        if (df.shape[0] >= 1_000_000):
            filtered = filter_df_large(df, sketch)
        else:
            filtered = filter_df_small(df, sketch)
            
        return query(filtered)
    else:
        return query(df)
        
    
def filter_df_small(df, sketch):
    cond = np.equal(df['flux_annot'].cat.codes.values, sketch[0])
    for i in sketch[1:]:
        cond |= np.equal(df['flux_annot'].cat.codes.values, i)
        
    return df[cond]

def filter_df_large(df, sketch):
    cond = df['flux_annot'].cat.codes == sketch[0]
    for i in sketch[1:]:
        cond |= df['flux_annot'].cat.codes == i
        
    return df.loc[cond]

def filter_df_multi(df, sketch):
    return pd.concat([df.xs(i) for i in sketch])


for num_rows in [10, 100, 1000, 10_000, 100_000, 1_000_000, BATCH1_ROWS]:
    print("NUM ROWS:", num_rows)
    
    # Load data
    df = pd.read_csv('data/test_set_batch1.csv', nrows=num_rows, dtype=dtypes)

    # Annotate data
    df = annotate(df, 'flux', 10)
    
    # Perform query without filter
    result = select_filter(df, test_query)
    print(f'query result: {result}')
    
    results_sec = timeit.repeat('select_filter(df, test_query)', globals=globals(), number=1, repeat=100)
    results_ms = np.array(results_sec).min() * 1000
    print(f'unfiltered query time: {results_ms:.8f} ms')

    # Perform query with filter
    sketch = [9]
    result = select_filter(df, test_query, sketch, multi=False)
    print(f'query result: {result}')
    
    results_sec = timeit.repeat('select_filter(df, test_query, sketch, multi=False)', globals=globals(), number=1, repeat=100)
    results_ms = np.array(results_sec).min() * 1000
    print(f'filtered query time: {results_ms:.8f} ms')
    
    # Perform query with MultiIndex filter
    mi = pd.MultiIndex.from_arrays([df['flux_annot'].cat.codes, df['object_id']], names=['flux_annot', 'object_id'])
    df_mi = df.set_index(mi)
    
    result = select_filter(df_mi, test_query, sketch, multi=True)
    print(f'query result: {result}')
    
    results_sec = timeit.repeat('select_filter(df_mi, test_query, sketch, multi=True)', globals=globals(), number=1, repeat=100)
    results_ms = np.array(results_sec).min() * 1000
    print(f'multiindex filtered query time: {results_ms:.8f} ms')
    
    print('\n**********\n')
