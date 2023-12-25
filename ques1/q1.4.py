

#This is a comment 
import pandas as pd
import numpy as np
bank_train=pd.read_csv(r'C:\DSPR_Data_Sets\Website Data Sets\bank_marketing_training')

#print(pd.crosstab(bank_train['previous_outcome'], bank_train['response']))
crosstab_01 = pd.crosstab(bank_train['previous_outcome'], bank_train['response'])
print(crosstab_01)