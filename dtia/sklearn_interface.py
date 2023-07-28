import logging
from dtia import DT_Model_Dev, Extract_N_ID_Feature_Threshold
from pathlib import Path


class DecisionTreeInsightAnalyser:
    def __init__(self,
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
                 N_ID_Feature_Threshold_Path_file=f"./n_id_feat_thresh.csv"):
        """DecisionTreeInsightAnalyser

        :param test_percent: 
        :type test_percent: 
        :param min_s_leaf_inc: 
        :type min_s_leaf_inc: 
        :param min_s_leaf: 
        :type min_s_leaf: 
        :param max_depth_inc: 
        :type max_depth_inc: 
        :param max_depth: 
        :type max_depth: 
        :param number_of_folds: 
        :type number_of_folds: 
        :param metrics_diff: 
        :type metrics_diff: 
        :param avg_tst_metrics: 
        :type avg_tst_metrics: 
        :param use_time_stamped_folders: 
        :type use_time_stamped_folders: 
        :param Model_Metrics_Out_Path: 
        :type Model_Metrics_Out_Path: 
        :param Model_Details_Out_Path: 
        :type Model_Details_Out_Path: 
        :param Imp_Nodes_Path_file: 
        :type Imp_Nodes_Path_file: 
        :param N_ID_Feature_Threshold_Path_file: 
        :type N_ID_Feature_Threshold_Path_file: 
        :returns: 

        """

        self.test_percent                       = test_percent
        self.min_s_leaf_inc                     = min_s_leaf_inc
        self.min_s_leaf                         = min_s_leaf
        self.max_depth_inc                      = max_depth_inc
        self.max_depth                          = max_depth
        self.number_of_folds                    = number_of_folds
        self.metrics_diff                       = metrics_diff
        self.avg_tst_metrics                    = avg_tst_metrics
        self.Imp_Nodes_Path_file                = Imp_Nodes_Path_file
        self.N_ID_Feature_Threshold_Path_file   = N_ID_Feature_Threshold_Path_file

        import time
        ts = time.time() if use_time_stamped_folders else ""

        joblibs_path = Model_Metrics_Out_Path
        csvs_path = Model_Details_Out_Path
        self.Model_Metrics_Out_Path = joblibs_path or f".#dtia#/{ts}/joblibs/"
        self.Model_Details_Out_Path = csvs_path or f".#dtia#/{ts}/csvs/"

        from pathlib import Path
        joblibsPath = Path(self.Model_Metrics_Out_Path)
        csvsPath = Path(self.Model_Details_Out_Path)

        if not joblibsPath.exists():
            logging.info(f"Making a new directory {joblibsPath}")
            joblibsPath.mkdir(parents=True)
            pass

        if not csvsPath.exists():
            logging.info(f"Making a new directory {csvsPath}")
            csvsPath.mkdir(parents=True)
            pass
        pass

    def fit(self, X, y):
        """Trains decision tree classifiers on in put data `X` and labels `y`

        :param X: Independent variables
        :type X: numpy array-like
        :param y: Dependent variables (class labels)
        :type y: numpy array-like or iterable
        :returns: File paths for trained models (.joblib files) and
        file paths for generated CSV files.

        """
        models, files = DT_Model_Dev(X, y,
                                     test_percent=self.test_percent,
                                     min_s_leaf_inc=self.min_s_leaf_inc,
                                     min_s_leaf=self.min_s_leaf,
                                     max_depth_inc=self.max_depth_inc,
                                     max_depth=self.max_depth,
                                     number_of_folds=self.number_of_folds,
                                     metrics_diff=self.metrics_diff,
                                     avg_tst_metrics=self.avg_tst_metrics,
                                     Model_Metrics_Out_Path=self.Model_Metrics_Out_Path,
                                     Model_Details_Out_Path=self.Model_Details_Out_Path)

        Extract_N_ID_Feature_Threshold(Model_Details_Paths=files,
                                       Imp_Nodes_Path_file=self.Imp_Nodes_Path_file,
                                       N_ID_Feature_Threshold_Path_file=self.N_ID_Feature_Threshold_Path_file)
        return models, files
    pass
