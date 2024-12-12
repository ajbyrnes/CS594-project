import pandas as pd

passbands = [
    'ultraviolet',
    'green',
    'red',
    'infrared',
    'near-z',
    'near-y'
]

df = pd.read_csv('data/test_set_batch1.csv')
df['passband_str'] = df['passband'].astype('category').cat.rename_categories(passbands)
df.index.name = 'index'

df.to_csv('data/test_set_batch1.csv')