### Ex-06-Feature-Transformation

### AIM

To read the given data and perform Feature Transformation process and save the data to a file.

### EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

### ALGORITHM

### STEP 1:

Read the given Data

### STEP 2:

Clean the Data Set using Data Cleaning Process

### STEP 3:

Apply Feature Transformation techniques to all the features of the data set

### STEP 4:

Print the transformed features

PPROGRAM:

NAME:A.ARUVI

REG NO:212222230014

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

### OUTPUT:

![image](https://user-images.githubusercontent.com/120443233/232683593-15851ac7-41a4-4561-8b62-87baa2802cc3.png)

![image](https://user-images.githubusercontent.com/120443233/232683623-c8eef1ad-e4db-496b-8e67-097cc26f3b77.png)

![image](https://user-images.githubusercontent.com/120443233/232683716-0b7f7e3f-2a09-4514-8fa9-f1f4b18f883a.png)

![image](https://user-images.githubusercontent.com/120443233/232684058-c3ce2796-4fc8-48de-b41b-3628706d16b0.png)

![image](https://user-images.githubusercontent.com/120443233/232684117-109e7697-ea60-4ef4-b454-bfa1da171c04.png)

![image](https://user-images.githubusercontent.com/120443233/232684174-3fade8f7-43d5-4dfa-903e-56eb274455e1.png)

![image](https://user-images.githubusercontent.com/120443233/232684222-b90fb423-ac1e-439e-9ed2-a3a3bca46999.png)

![image](https://user-images.githubusercontent.com/120443233/232684249-595ad637-5717-4675-858a-d84ff479b857.png)

![image](https://user-images.githubusercontent.com/120443233/232684282-dba73918-581a-4b32-884c-573acad2aca3.png)

![image](https://user-images.githubusercontent.com/120443233/232684513-7b324bfd-cc2e-4ae3-992b-fcb397452002.png)

### RESULT:

Thus feature transformation is done for the given dataset.

