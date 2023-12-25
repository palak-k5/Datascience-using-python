

import pandas as pd
import numpy as np
bank_train=pd.read_csv(r'C:\DSPR_Data_Sets\Website Data Sets\bank_marketing_training')
print(bank_train.loc[0])
print(bank_train.loc[[0, 2, 3]])
print(bank_train[0:10])
print(bank_train['age'])
print(bank_train[['age', 'job']])