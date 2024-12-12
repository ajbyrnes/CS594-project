
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

def get_info(df):
    # Get info
    print("----------DataFrame----------")
    print(df.info())
    print()
    print('----------Annotation Column----------')
    print(df['mjd_annot'].info())
    print()


def annotate(df, num_bins):
    # Annotate data
    df['mjd_annot'] = pd.cut(df['mjd'], bins=num_bins)
    return df


# Timing annotation for DataFrame size
for num_rows in [10, 100, 1000, 10_000, 100_000, 1_000_000, BATCH1_ROWS]:
    # Load data
    df = pd.read_csv('data/test_set_batch1.csv', nrows=num_rows, dtype=dtypes)
    
    # Annotate data
    results_sec = timeit.repeat('annotate(df, 10)', globals=globals(), number=1, repeat=100)
    results_ms = np.array(results_sec) * 1000
    print(f'Time to annotate {num_rows} rows: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')


print('**********')    

    
# Load full data
df = pd.read_csv('data/test_set_batch1.csv', dtype=dtypes)
print("----------DataFrame----------")
print(df.info())
print()
for col in df.columns:
    print(f'----------{col}----------')
    print(df[col].info())
    print()


print('**********')    


# Timing annotation for number of bins
for num_bins in [2, 4, 8, 16, 32, 64]:
    # Annotate data
    results_sec = timeit.repeat('annotate(df, num_bins)', globals=globals(), number=1, repeat=100)
    results_ms = np.array(results_sec) * 1000
    print(f'Time to annotate data with {num_bins} bins: {results_ms.mean():.8f} ms +/- {results_ms.std():.8f} ms')


print('**********')    


# Checking memory usage for annotated column for number of bins
for num_bins in [2, 4, 8, 16, 32, 64]:
    # Annotate data
    annotate(df, num_bins)
    get_info(df)
    codes_only = pd.Series(df['mjd_annot'].cat.codes)
    print('Categorical overhead: ', df['mjd_annot'].memory_usage() - codes_only.memory_usage(), 'bytes')