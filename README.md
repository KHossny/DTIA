# Decision Tree Insight Analysis Tool

## Installation

### From Source
Download and unzip the package. 
Open anaconda prompt PowerShell.
```
cd "your\\path\\to\\dtia"
python -m pip install .
```

### Using PIP
```
pip3 install git+https://github.com/KHossny/DTIA
```

## Quick Example
```
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
