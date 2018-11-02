
# Project: Iris Flower Classification

## Supervised Learning, Classification

![](logo.jpg)

----
## Table of Contents

- [Getting Started](#Getting-Started)
- [Load the Data](#Load-The-Data)
 - [Version Check-In](#Version-Check-In)
 - [Import Library](#Import-Library)
 - [Data Is Here](#Data-is-Here)
- [Data Exploration](#Data-Exploration)
 - [Peak at Data](#Peak-at-Data)
 - [Statistical Summary](#Statistical-Summary)
- [Data Visualization](#Data-Visualization)
 - [Univariate Plots](#Univariate-Plots)
 - [Multivariate Plots](#Multivariate-Plots)
- [Evaluate Algorithms](#Evaluate-Algorithms)
 - [Create a Validation Dataset](#Create-a-Validation-Dataset)
 - [Developing Model](#Developing-Model)
- [Make Prediction](#Make-Prediction)
 
-----

## Getting Started

## Load The Data

### Version Check-In


```python
# Library Version check-in
import sys, numpy, scipy, pandas as pd, matplotlib, sklearn, seaborn as sns
```


```python
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(sns.__version__))
```

    Python: 3.7.0 (default, Jun 28 2018, 08:04:48) [MSC v.1912 64 bit (AMD64)]
    scipy: 1.1.0
    numpy: 1.15.2
    pandas: 0.23.4
    sklearn: 0.20.0
    matplotlib: 3.0.0
    Seaborn: 0.9.0
    

### No Warnings


```python
# No warning of any kind please!
import warnings
# will ignore any warnings
warnings.filterwarnings("ignore")
```

### Import Library


```python
# Loading required Libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
```

### Data is Here


```python
# Load the dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# column names for the dataset
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# feeding the data with pandas, giving column names to dataset. 
dataset = pd.read_csv(url, names= names)
```

## Data Exploration

### Peak at Data


```python
# Peak at the data
dataset.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dimensions of the dataset
r, c = dataset.shape
print('This dataset has ',r,' rows and ' ,c,' columns.')
```

    This dataset has  150  rows and  5  columns.
    


```python
# Grouping by Class
dataset.groupby('class').size()
```




    class
    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64



### Statistical Summary


```python
# Statistical Summary
dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Visualization

### Univariate Plots
Univariate plots to better understand each attribute.


```python
# plotting each variable
# box and whiskers plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(15,13))
plt.show()
```


![png](output_24_0.png)



```python
dataset.plot(kind='hist', subplots=True, layout = (2,2), sharex=False, sharey=False, figsize=(15,13))
plt.show()
```


![png](output_25_0.png)


### Multivariate Plots
Multivariate plots to better understand the relationships between attributes.


```python
scatter_matrix(dataset, figsize=(15,10))
plt.show()
```


![png](output_27_0.png)


## Evaluate Algorithms
_Describe the tools and techniques you will use necessary for a model to make a prediction_

### Create a Validation Dataset


```python
# 80-20 train-test-split
from sklearn.model_selection import train_test_split
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
test = 0.2
seed = 53
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, random_state = seed)
```


```python
scoring = 'accuracy'
```

We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

### Developing Model
We'll evaluate these 6 algorithms:

- Logistic Regression (LR)
- Linear Discriminate Analysis (LDA)
- K-Nearest Neighbours (KNN)
- Classification and Regression Trees (CRT)
- Guassian Naive Bayes (GNN)
- Support Vector Machine (SVM)

This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CRT, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
```


```python
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CRT', DecisionTreeClassifier()))
models.append(('GNN', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
```


```python
# Now evaluating each model
from sklearn.model_selection import KFold, cross_val_score

results = []
names = []

# looping models in the list
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print(name, ': ', cv_results.mean(), cv_results.std())
```

    LR :  0.95 0.055277079839256664
    LDA :  0.9833333333333332 0.03333333333333335
    KNN :  0.9583333333333333 0.04166666666666669
    CRT :  0.9666666666666666 0.055277079839256664
    GNN :  0.95 0.055277079839256664
    SVM :  0.975 0.03818813079129868
    RF :  0.975 0.03818813079129868
    

### Dimentionality Reduction

#### ETC


```python
# importing model for feature importance
from sklearn.ensemble import ExtraTreesClassifier

# passing the model
model = ExtraTreesClassifier(random_state = 53)

X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, -1:]

# training the model
model.fit(X, y)

# extracting feature importance from model and making a dataframe of it in descending order
ETC_feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['ETC']).sort_values('ETC', ascending=False)

# removing traces of this model
model = None

# results
ETC_feature_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ETC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>petal_length</th>
      <td>0.461489</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>0.329650</td>
    </tr>
    <tr>
      <th>sepal_length</th>
      <td>0.147077</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>0.061783</td>
    </tr>
  </tbody>
</table>
</div>



#### RFC


```python
# passing the model
model = RandomForestClassifier(random_state = 53)

# training the model
model.fit(X, y)

# extracting feature importance from model and making a dataframe of it in descending order
RFC_feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['RFC']).sort_values('RFC', ascending=False)

# removing traces of this model
model = None

# show top 10 features
RFC_feature_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RFC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>petal_width</th>
      <td>0.500073</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>0.414914</td>
    </tr>
    <tr>
      <th>sepal_length</th>
      <td>0.076769</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>0.008244</td>
    </tr>
  </tbody>
</table>
</div>



#### ADBC


```python
# importing model for feature importance
from sklearn.ensemble import AdaBoostClassifier

# passing the model
model = AdaBoostClassifier(random_state = 53)

model.fit(X, y)

# extracting feature importance from model and making a dataframe of it in descending order
ADB_feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['ADB']).sort_values('ADB', ascending=False)

# removing traces of this model
model = None

ADB_feature_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ADB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>petal_length</th>
      <td>0.54</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>0.46</td>
    </tr>
    <tr>
      <th>sepal_length</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



#### GBC


```python
# importing model for feature importance
from sklearn.ensemble import GradientBoostingClassifier

# passing the model
model = GradientBoostingClassifier(random_state = 53)

# training the model
model.fit(X, y)

# extracting feature importance from model and making a dataframe of it in descending order
GBC_feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['GBC']).sort_values('GBC', ascending=False)

# removing traces of this model
model = None

# show top 10 features
GBC_feature_importances.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GBC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>petal_width</th>
      <td>0.799085</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>0.183235</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>0.013661</td>
    </tr>
    <tr>
      <th>sepal_length</th>
      <td>0.004019</td>
    </tr>
  </tbody>
</table>
</div>



#### Select K Best Classifier


```python
from sklearn.feature_selection import SelectKBest

kbest = SelectKBest(k = 3).fit(X,y)
mask = kbest.get_support()
new_features = X.columns[mask]

new_features
```




    Index(['petal_length', 'petal_width'], dtype='object')



## Make Prediction


```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lda =  DecisionTreeClassifier()

lda.fit(X_train, y_train)

predict = lda.predict(X_test)

print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

lda = None
```

    0.9333333333333333
    [[10  0  0]
     [ 0  9  2]
     [ 0  0  9]]
                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       1.00      0.82      0.90        11
     Iris-virginica       0.82      1.00      0.90         9
    
          micro avg       0.93      0.93      0.93        30
          macro avg       0.94      0.94      0.93        30
       weighted avg       0.95      0.93      0.93        30
    
    


```python

```
