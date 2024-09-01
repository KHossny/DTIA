# Decision Tree Insight Analysis Tool

## Installation

### Using ZIP File
Download and unzip the package. <br />
Open anaconda prompt PowerShell. <br />
```powershell
cd "your\path\to\dtia"
python -m pip install .
```

### Using git
Open anaconda prompt PowerShell. <br />
```powershell
git clone https://github.com/KHossny/DTIA.git your\path\to\dtia
cd "your\path\to\dtia"
python -m pip install .
```

### Using PIP
Download and install git tools for Windows using the link (https://git-scm.com/download/win)
```
pip install git+https://github.com/KHossny/DTIA
```

## Examples
### iris Example
```
python iris_example.py
```

### Quick Generic Example
Make a new directory.
```powershell
cd "your\path\to\the\file\to\be\run\in\the\new\directory"
```
Make a new Python file with the following content. <br />
load the input features in a variable called 'x' <br />
load the labels in a variable called 'y' <br />
<br />
|Variable Name                      |Description                                                                                                   |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------|
|test_percent:                      |Percentage of the data taken as a test for the generated decision tree models.                                |
|min_s_leaf_inc:                    |Increment in the minimum number of samples per leaf specified in the generated decision tree models.          |
|min_s_leaf:                        |Maximum number of minimum samples per leaf in the generated decision tree models.                             |
|max_depth_inc:                     |Increment in the maximum depth of the generated decision tree models.                                         |
|max_depth:                         |Maximum depth of the generated decision tree models.                                                          |
|number_of_folds:                   |Number of folds over which each of the generated decision tree models will be trained and tested.             |
|metrics_diff:                      |The difference between the training and test precision for each developed decision tree model.                |
|avg_tst_metrics:                   |The average test metrics for each developed decision tree model.                                              |
|Model_Metrics_Out_Path:            |Path where the selected model metrics and performance file will be saved.                                     |
|Model_Details_Out_Path:            |Path where the files containing details of each selected model will be saved.                                 |
|Imp_Nodes_Path_file:               |Path where the file containing important nodes in all of the selected models is saved.                        |
|N_ID_Feature_Threshold_Path_file:  |Path where the file including node ID, feature number, and feature threshold for all selected models is saved.|
<br />
The following code imports the 'iris' dataset and develops decision tree models using different hyperparameters. The minimum number of samples per leaf ranges from one to 20 with the step of one. The maximum depth of the tree ranges between two and ten with a step of one. Each model was trained and tested over ten folds. The selection criteria for the models to be analyzed were to have an average difference between training and test classification metrics of 0.01, and the average test metrics should be higher than 0.9. Finally, it used the same path where you are in the Anaconda prompt PowerShell to create the results folder. The results folder is named '.#dtia#'. In '.#dtia#' there is a folder named by the time stamp. This folder includes two folders. The 'csvs' and 'joblibs' folders contain the csv files describing the details of each model that passed the selection criteria and the joblib files of all the generated models, respectively.

```python
from sklearn.datasets import load_iris
from dtia import DecisionTreeInsightAnalyser

x, y = load_iris(return_X_y=True)

dtia_clf = DecisionTreeInsightAnalyser(
    test_percent=0.2,
    min_s_leaf_inc=1,
    min_s_leaf=20,
    max_depth_inc=1,
    max_depth=10,
    number_of_folds=10,
    metrics_diff=0.01,
    avg_tst_metrics=0.90,
    use_time_stamped_folders=True,
    Model_Metrics_Out_Path=None,
    Model_Details_Out_Path=None,
    Imp_Nodes_Path_file=f"./imp_nodes.csv",
    N_ID_Feature_Threshold_Path_file=f"./n_id_feat_thresh.csv")

dtia_clf.fit(x, y)
```

A more generic example is demonstrated below using pandas in reading the csv so you will have to import pandas first. The operating code will be as follows. 

```python
from dtia import DecisionTreeInsightAnalyser
import pandas as pd
Data_1 = pd.read_csv('csv\\path\\file.csv', usecols = ['col_name_1', 'col_name_2'])
Data_2 = pd.read_csv('csv\\path\\file.csv', usecols = ['Labels_col_Name'])
x = Data_1.values
y = Data_2.values

dtia_clf = DecisionTreeInsightAnalyser(
    test_percent=0.2,
    min_s_leaf_inc=1,
    min_s_leaf=20,
    max_depth_inc=1,
    max_depth=10,
    number_of_folds=10,
    metrics_diff=0.01,
    avg_tst_metrics=0.90,
    use_time_stamped_folders=True,
    Model_Metrics_Out_Path='path\\to\\the\\place\\where\\you\\want\\the\\file\\to\\be\\saved\\',
    Model_Details_Out_Path='path\\to\\the\\place\\where\\you\\want\\the\\file\\to\\be\\saved\\',
    Imp_Nodes_Path_file="path\\to\\the\\place\\where\\you\\want\\the\\file\\to\\be\\saved\\imp_nodes.csv",
    N_ID_Feature_Threshold_Path_file="path\\to\\the\\place\\where\\you\\want\\the\\file\\to\\be\\saved\\n_id_feat_thresh.csv")

dtia_clf.fit(x, y)
```

Specific iris example with default location for output saving.
```python
import logging
logging.basicConfig(level=logging.INFO)

from sklearn.datasets import load_iris
from dtia import DecisionTreeInsightAnalyser


X, y = load_iris(return_X_y=True)
dtia_clf = DecisionTreeInsightAnalyser(                 
    Model_Metrics_Out_Path="output/joblibs/",
    Model_Details_Out_Path="output/csvs/",
    Imp_Nodes_Path_file=f"./imp_nodes.csv",
    N_ID_Feature_Threshold_Path_file=f"./n_id_feat_thresh.csv")

dtia_clf.fit(X, y)
```
