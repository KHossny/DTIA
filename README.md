# Decision Tree Insight Analysis Tool

## Installation

### From Source
Download and unzip the package. 
Open anaconda prompt PowerShell.
```
cd "your\\path\\to\\dtia"
python -m pip install .
```
#### iris Example
```
python iris_example.py
```

### Using PIP
Download and install git tools for Windows using the link (https://git-scm.com/download/win)
```
pip install git+https://github.com/KHossny/DTIA
```



#### Quick Generic Example
Make a new directory.
```
cd "your\\path\\to\\the\\file\\to\\be\\run\\in\\the\\new\\directory"
```
Make a new python file with the following content.
load the input features in a variable called x
load the labels in a variable called y
test_percent: percentage of the data taken as test for the generated decision tree models.
min_s_leaf_inc: increment in minimum number of samples per leaf specified in the generated decision tree models.
min_s_leaf: maximum number of minimum samples per leaf in the generated decision tree models.
max_depth_inc: increment in the maximum depth of the generated decision tree models.
max_depth: maximum depth of the generated decision tree models.
number_of_folde: number of folds over which each of the generated decision tree models will be trained and tested.
metrics_diff: the average difference between the training and test metrics for each developed decision tree model.
avg_tst_metrics: the average test metrics for each developed decision tree model. 
Model_Metrics_Out_Path: path where the selected model metrics and performance file will be saved.
Model_Details_Out_Path: path where the files containing details of each selected model will be saved.
Imp_Nodes_Path_file: path where the file containing important nodes in all of the selected models is saved.
N_ID_Feature_Threshold_Path_file: path where the file including node ID, feature number, and feature threshold for all selected models is saved. 

In summary, the following code imports the 'iris' dataset and develops decision tree models 
```
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
