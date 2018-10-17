# Project: Iris Flower Classification
## Supervised Learning, Classification

<p align = 'center'><img src = 'logo.jpg', height=350, width =390></p>

----

### Table Of Contents:
- [Description](#description)<br>
    - [About the project](#about-the-project)<br>
    - [What needs to be done](#what-needs-to-be-done)<br>
    - [Why this project](#why-this-project)<br>
- [Data](#data)<br>
    - [Files](#files)<br>
    - [Dataset file](#dataset-file)<br>
- [Loading Project](#loading-project)<br>
    - [Requirements](#requirements)<br>
    - [Execution](#execution)<br>

----

### Description

#### About the project
"Describe the project in detail. What is your task and what is that you want to predict."


#### What needs to be done
"Describe how are you going to solve this project. How are you going to approach it?"


#### Why this project
"Provide Motivation for this project."


----

### Data

#### Files

This project contains 1 file and 2 folders:

- `report.ipynb`: This is the main file where I have performed my work on the project.
- `export/` : Folder containing HTML and PDF version file of notebook.
- `plots/` : Contains images of all the plots that are displayed in `report.ipynb` file.


#### Dataset file

The data set contains 3 classes of 50 instances each, total 150 instances, where each class refers to a type of iris plant. One class is linearly separable from the other 2 and the latter are **not linearly separable** from each other. 

**Predicting attribute:** Class of iris plant. 

**Attribute Information:** We have 4 features in this dataset and a target variable `class`.

    - sepal length in cm.
    - sepal width in cm.
    - petal length in cm.
    - petal width in cm.
    - Class:
        - Iris Setosa
        - Iris Versicolour
        - Iris Virginica
        
----

### Loading Project

#### Requirements

This project requires **Python 3.7** and the following Python libraries installed:

- [Python 3.7](https://www.python.org/downloads/)(_Language Used for the project_)
- [NumPy](http://www.numpy.org/)(_For Scientific Computing_)
- [Pandas](http://pandas.pydata.org)(_For Data Analysis_)
- [matplotlib](http://matplotlib.org/)(_For Visualization_)
- [seaborn](https://seaborn.pydata.org/installing.html)(_For Visualization_)
- [scikit-learn](http://scikit-learn.org/stable/)(_ML Library for training & testing data_)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included.

#### Execution

In a terminal or command window, navigate to the top-level project directory `Iris_Flower_Classification` (that contains this README) and run one of the following commands:

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

This will open the Jupyter/iPython Notebook software and project file in your browser.

----