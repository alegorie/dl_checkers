import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/network-storage/bgdata/test_clean_checkers.csv')

print(df.columns)
array = df['checkers'].as_matrix()

result = []

for i in array:
    i = i.split(':')
    i = [abs(int(j)) for j in i]
    i = i[1:-1]
    result.append(i)

answer = np.array(result[:30000])

np.save('test_clean_checkers', answer)
