
# Customer Conversion Prediction

**Problem Statement :**

You are working for a new-age insurance company and employ multiple outreach plans to sell term insurance to your customers. Telephonic marketing campaigns still remain one of the most effective ways to reach out to people however they incur a lot of cost. Hence, it is important to identify the customers that are most likely to convert beforehand so that they can be specifically targeted via call. We are given the historical marketing data of the insurance company and are required to build a ML model that will predict if a client will subscribe to the insurance.

**Features :**

● age (numeric) ● job : type of job ● marital : marital status ● educational_qual : education status ● call_type : contact communication type ● day: last contact day of the month (numeric) ● mon: last contact month of year ● dur: last contact duration, in seconds (numeric) ● num_calls: number of contacts performed during this campaign and for this client ● prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

**Output variable (desired target):**

● y - has the client subscribed to the insurance?

**Minimum Requirements**

It is not sufficient to just fit a model - the model must be analysed to find the important factors that contribute towards the price. AUROC must be used as a metric to evaluate the performance of the models


## Author

- [@jacksabari](https://github.com/JackSabari)


## Tech Stack

![Logo](https://www.python.org/static/img/python-logo@2x.png)

![Logo](https://www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/scikit-learn-logo.png)

![Logo](https://seaborn.pydata.org/_static/logo-wide-lightbg.svg)

![Logo](https://matplotlib.org/_static/images/logo_dark.svg)

![Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png)

![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_VfYfuw4JGQC0QLtbrhWyAQgW9qD9fXanG34lWGAyI1y34PxtAPagPNkCTAoX7_x7sFw&usqp=CAU)

![Logo](https://www.fullstackpython.com/img/logos/scipy.png)


## Installation


```Python
  pip install category_encoders
```
    
## Import Libraries
```python
#Data preprocessing and read csv files, dataframes, etc
import pandas as pd
import numpy as np

#Encoding the data 
import category_encoders as ce

#Visualize the data
import seaborn as sns
import matplotlib.pyplot as plt

#Find the outliers in data
from scipy import stats

#Machine learning models and techniques
from sklearn.utils import resample,all_estimators
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#To ignore warnings
import warnings
warnings.simplefilter('ignore')

```


## 🔗 Raw Dataset
[![Data](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0odO2hjYnaY_wtkaLzzF23UM24MrwtKK1GEaQo6HCmw&s)](https://drive.google.com/file/d/1-O4yGvX4Iq0k6KZkJhMhNyqJ4Ic15QQQ/view?usp=share_link)




## Screenshots

         To Find Null Values

![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/NJull.png)



