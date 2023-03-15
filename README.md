
# Customer Conversion Prediction

**Problem Statement :**

You are working for a new-age insurance company and employ multiple outreach plans to sell term insurance to your customers. Telephonic marketing campaigns still remain one of the most effective ways to reach out to people, however they incur a lot of cost. Hence, it is important to identify the customers that are most likely to convert beforehand so that they can be specifically targeted via call. We are given the historical marketing data of the insurance company and are required to build a ML model that will predict if a client will subscribe to the insurance.

**Features :**

● age (numeric) ● job : type of job ● marital : marital status ● educational_qual : education status ● call_type : contact communication type ● day: last contact day of the month (numeric) ● mon: last contact month of year ● dur: last contact duration, in seconds (numeric) ● num_calls: number of contacts performed during this campaign and for this client ● prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

**Output variable (desired target):**

● y - has the client subscribed to the insurance?

**Minimum Requirements :**

It is not sufficient to just fit a model - the model must be analyzed to find the important factors that contribute towards the price. AUROC must be used as a metric to evaluate the performance of the models


## Author

- [@Sabarinathan](https://github.com/JackSabari)

  **Batch - D42**

## Tech Stack

<img align="left" alt="Coding" width=250 src="https://www.python.org/static/img/python-logo@2x.png">
<img align="left" alt="Coding" width=250 src="https://www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/scikit-learn-logo.png">
<img align="left" alt="Coding" width=250 src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg">
<img align="left" alt="Coding" width=355 src="https://matplotlib.org/_static/images/logo_dark.svg">
<img align="left" alt="Coding" width=250 src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png">
<img align="left" alt="Coding" width=250 src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_VfYfuw4JGQC0QLtbrhWyAQgW9qD9fXanG34lWGAyI1y34PxtAPagPNkCTAoX7_x7sFw&usqp=CAU">
<img align="left" alt="Coding" width=250 src="https://mljar.com/images/machine-learning/xgboost_v2.png">


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


## 🔗 Raw Dataset - From Guvi
[![Data](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0odO2hjYnaY_wtkaLzzF23UM24MrwtKK1GEaQo6HCmw&s)](https://drive.google.com/file/d/1-O4yGvX4Iq0k6KZkJhMhNyqJ4Ic15QQQ/view?usp=share_link)




## Process

        I checked null values are there or not and there is no null values

![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/NJull.png)
   

       we have to check the dataset is balanced or not, so I checked, dataset is not balanced    
       
       Before Resampling
       
![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/Before%20resampling.png)

       So, I did resampling to convert imbalance dataset to balance dataset.
       
       After Resampling
       
![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/After_resampling.png)       

       We should encode the text data into numeric, so I used one hot encoder, ordinal encoder and map function.
       
       After encoding, we should remove outliers, so I made unique values for each features except the encoding features then,
       I recognize only dur column has the outliers, so I graph it.
       
![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/Before_outliers.png)  

       After rectify the outliers in dur features
       
![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/After_outliers.png)

## Machine Learning

1. This dataset is classifier model dataset
2. Implement the Random Forest Classifier
3. Metrics scores are good with random forest, but better to check the feature importance once 

![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/Feature%20Importance.png)

      job_unknown feature is the least important feature, so I'm going to drop it and again build the model.
      
4. Implement the Random Forest Classifier and Logistic Regression      
5. still Metrics scores are good with random forest classifier only, but we should check with ROC Curve

![Screenshot](https://github.com/JackSabari/Guvi_Final_Projects/blob/main/Processed/Screenshots/Roc_curve.png)

     It is evident from the plot that the AUC for the Random Forest ROC curve is higher than that for the Logistic Regression ROC curve. Therefore, we can say that        Random Forest did a better job of classifying the positive class in the dataset.
     
6. Implement the Gradient Boosting Classifier     
7. XGB has good score after parameter tuning, but didn't break the Random Forest score.

## Conclusion

The most contributing and important feature is "dur". Roc curve, accuracy score, confusion_matrix, classification report said that random forest is better than Gradient Boosting and logistic regression.

Accuracy score of Random Forest Classifier : 97%

Accuracy score of Gradient Boosting : 94%

Accuracy score of Logistics Regression : 80%

The Random Forest model is fitted to this dataset.


## Support

  **E-Mail   : jacksabari999@gmail.com**
  
  **LinkedIn : https://www.linkedin.com/in/sabarinathan-j-218a11205/**
