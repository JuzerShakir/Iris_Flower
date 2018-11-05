# Project: Iris Flower
## Supervised Learning, Classification

<p align = 'center'><img src = 'logo.jpg', height=350, width =420></p>

----

<h3><ins>Table Of Contents:</ins></h3>

- [Description](#description)<br>
    - [About the project](#about-the-project)<br>
    - [What needs to be done](#what-needs-to-be-done)<br>
    - [Sources](#sources)
- [Data](#data)<br>
    - [Files](#files)<br>
    - [Dataset file](#dataset-file)<br>
- [Loading Project](#loading-project)<br>
    - [Requirements](#requirements)<br>
    - [Execution](#execution)<br>

----

<h3><ins>Description</ins></h3>

#### About the project:
This is the most famous dataset in ML and best for beginners who wants to get there hands dirty with _ML/Data Science_. Having less features and observations of the Iris flowers, no missing values or outliers to deal with, this makes implementing ML models easier and simple.

#### What needs to be done:
Since the project is clean and small, we will use this to our advantage and get practice on how to perform data visualization with matplotlib and seaborn (Data Visualization Libraries), implement most used feature selection methods in _ML/Data Science project_, and apply all classification models on this dataset. This will give us practice and hands on experience on how and when to implement and which works best given the dataset.

#### Sources:
- **Creator:** *R.A. Fisher*
- **Donor:** *Michael Marshall*

----

<h3><ins>Data</ins></h3>

#### Files:

This project contains 1 file and 2 folders:

- `report.ipynb`: This is the main file where I have performed my work on the project.
- `export/` : Folder containing HTML and PDF version file of notebook.
- `plots/` : Contains images of all the plots that are displayed in `report.ipynb` file.


#### Dataset file:

|||
| ------ | ------ |
| **Associated Task** | Classification |
| **Data Set Characteristics** | Multivariate |
| **Attribute Characteristics** | Real |
| **Number of Instances** | 150 |
| **Number of Attributes** | 4 |
| **Missing Values?** | **No** |
| **Area** | **Life** |

The data set contains _3 classes_ of _50 instances each_, total _150 instances_, where each class refers to a type of Iris plant. One class is linearly separable from the other 2 and the latter are **not linearly separable** from each other. 

**Predicting attribute:** Class of Iris plant. 

**Attribute Information:** We have _4 features_ in this dataset and a target variable `class`.

- sepal length in _cm_.
- sepal width in _cm_.
- petal length in _cm_.
- petal width in _cm_.
- Class:
    - _Iris Setosa_
    - _Iris Versicolour_
    - _Iris Virginica_
        
----

<h3><ins>Loading Project</ins></h3>

#### Requirements:

This project was solved with the following versions of libraries installed:

|             **Libraries\Language**            |               **Use**         | **Version** |
| :---------------------------------------------: | ----------------------------- | :-------: |
| [Python](https://www.python.org/downloads/) | _Language Used for the project_ | **3.7.0** |
| [NumPy](http://www.numpy.org/) | _For Scientific Computing_ | **1.15.2** |
| [Pandas](http://pandas.pydata.org) | _For Data Analysis_ | **0.23.4** |
| [matplotlib](http://matplotlib.org/) | _For Visualization_ | **3.0.0** |
| [seaborn](https://seaborn.pydata.org/installing.html) | _For Visualization_ | **0.9.0** |
| [scikit-learn](http://scikit-learn.org/stable/) | _ML Library for training & testing data_ | **0.20.0** |


If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included in it.

You will also need to have software [Jupyter Notebook](http://jupyter.org/install) installed to run and execute `report.ipynb` file. You can also use [Jupyterlab](https://github.com/jupyterlab/) too to run and execute, _Jupyterlab_ is better version of _Jupyter Notebook_. Instructions to download Jupyterlab can be found [here](https://github.com/jupyterlab/jupyterlab#installation).

#### Execution:

In a terminal or command window, navigate to the top-level project directory `Iris_Flower` (that contains this README) and run one of the following commands:

```bash
ipython notebook report.ipynb
```  
or
```bash
jupyter notebook report.ipynb
```
or if you have 'Jupyter Lab' installed
```bash
jupyter lab
```

This will open the Jupyter/iPython Notebook or Jupyterlab software and project file in your browser.

----