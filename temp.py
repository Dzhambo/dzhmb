import numpy as np
import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
k=data['Survived'].value_counts()
print (k[1]/(k.sum()))*100

