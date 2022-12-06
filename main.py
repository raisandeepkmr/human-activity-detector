import pickle
import numpy as np
import pandas as pd
lr = pickle.load(open("Log_reg2.pkl",'rb'))
pca = pickle.load(open("pca.pkl",'rb'))
dataframe = pd.read_csv('test.csv')
dataframe.drop(['Activity','subject'],axis=1,inplace=True)

value = dataframe.iloc[[100]].values
print(value)
print(value.shape)
my_pca = pca.transform(value)
result = lr.predict(my_pca)

if result == 0:
    print('LAYING ')
elif result == 1:
    print('SITTING')
elif result == 2:
    print('STANDING ')
elif result == 3:
    print('WALKING')
elif result == 4:
    print('WALKING_DOWNSTAIRS')
elif result == 5:
    print('WALKING_UPSTAIRS')
else:
    print('Invalid input')
