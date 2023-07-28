# python "C:\\Users\\hossn\\Desktop\Test_Datasets\Iris\Iris_Analysis.py"

#####################################################DT_Models_Development_and_Selection_Function####################################

import re
import pandas as pd
import numpy as np
import tqdm

import logging


def DT_Model_Dev(x, y, test_percent, 
                 min_s_leaf_inc, min_s_leaf, 
                 max_depth_inc, max_depth, 
                 number_of_folds, 
                 metrics_diff, avg_tst_metrics, 
                 Model_Metrics_Out_Path, Model_Details_Out_Path):
    
    import pandas as pd
    import numpy as np
    import pdb
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    import graphviz
    import re
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from joblib import dump, load
    
    
    

    
    
    
    def preparefeatures(x, y, random_state=0):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
        return x_train, y_train, x_val, y_val


    List_of_Model_Details_Out_Path = []



    Model_Graphviz = []
    Left_Nodes = []
    Right_Nodes = []

    Fold_Number = []
    Depth = []
    Min_Samples_Leaf = []
    
    Acc_Tr_Precision  = []
    Acc_Tr_Accuracy   = []
    Acc_Tr_F1_Score   = []
    Acc_Tr_Recall     = []
    
    Acc_Tst_Precision = []
    Acc_Tst_Accuracy  = []
    Acc_Tst_F1_Score  = []
    Acc_Tst_Recall    = []
    
    logging.info(f"Training DT models...")
    for i in tqdm.trange(1, min_s_leaf + 1, min_s_leaf_inc):    # loop over min num samples per leaf per tree
        for j in range (2, max_depth + 1, max_depth_inc):  # loop over max depth per tree
            for f in range(number_of_folds + 1):           # loop over num of folds
                
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = test_percent)
                
                number_of_models = min_s_leaf * max_depth * number_of_folds
                # print(i, j, f)
                logging.debug(f"{i, j, f}")
            
                # Model Training and Prediction
            
                clf = DecisionTreeClassifier(min_samples_leaf = i, max_depth = j)
                clf.fit(x_train, y_train)
                dump(clf, f'{Model_Metrics_Out_Path}DT_{i}_{j}_{f}.joblib')

                model_details = repr(tree.export_graphviz(clf))
                Model_Graphviz.append(model_details)
                
                y_hat_train = clf.predict(x_train)
                y_hat_val = clf.predict(x_val)
                
                
                # Evaluating Training and Test Performances for each model
                
                Tr_Precision = precision_score(y_train, y_hat_train, average = 'weighted')
                Tr_Accuracy = accuracy_score(y_train, y_hat_train)
                Tr_F1_Score = f1_score(y_train, y_hat_train, average = 'weighted')
                Tr_Recall = recall_score(y_train, y_hat_train, average = 'weighted')
                
                Tst_Precision = precision_score(y_val, y_hat_val, average = 'weighted')
                Tst_Accuracy = accuracy_score(y_val, y_hat_val)
                Tst_F1_Score = f1_score(y_val, y_hat_val, average = 'weighted')
                Tst_Recall = recall_score(y_val, y_hat_val, average = 'weighted')
                
                # print(Tr_Accuracy, Tst_Accuracy)
                logging.debug(f"{Tr_Accuracy=}, {Tst_Accuracy=}")
                
                
                # Storing Developed Models Training and Test Metrics
                
                Fold_Number.append(f)
                Min_Samples_Leaf.append(i)
                Depth.append(j)
                
                Acc_Tr_Precision.append(Tr_Precision)
                Acc_Tr_Accuracy.append(Tr_Accuracy)
                Acc_Tr_F1_Score.append(Tr_F1_Score)
                Acc_Tr_Recall.append(Tr_Recall)
                
                Acc_Tst_Precision.append(Tst_Precision)
                Acc_Tst_Accuracy.append(Tst_Accuracy)
                Acc_Tst_F1_Score.append(Tst_F1_Score)
                Acc_Tst_Recall.append(Tst_Recall)
                pass
            pass
        pass

    # Selecting Model Parameters
    # Selection Crieteria 1
    Sel_Fold_Number = []
    Sel_Model_Graphviz = []
    Sel_Min_Samples_Leaf = []
    Sel_Depth = []
    
    Sel_Tr_Precision = []
    Sel_Tr_Accuracy = []
    Sel_Tr_F1_Score = []
    Sel_Tr_Recall = []
    
    Sel_Tst_Precision = []
    Sel_Tst_Accuracy = []
    Sel_Tst_F1_Score = []
    Sel_Tst_Recall = []


    # Average Difference between Training and Test Metrics
    
    for m in range(len(Min_Samples_Leaf)):
        #pdb.set_trace()
        average_difference = (np.mean([abs((np.array(Acc_Tr_Precision[m]) - np.array(Acc_Tst_Precision[m]))), 
                                      abs((np.array(Acc_Tr_Accuracy[m])   - np.array(Acc_Tst_Accuracy[m]))), 
                                      abs((np.array(Acc_Tr_F1_Score[m])   - np.array(Acc_Tst_F1_Score[m]))), 
                                      abs((np.array(Acc_Tr_Recall[m])     - np.array(Acc_Tst_Recall[m])))]))

    
        if average_difference <= metrics_diff:
            Sel_Fold_Number.append(Fold_Number[m])
            Sel_Model_Graphviz.append(Model_Graphviz[m])
            Sel_Min_Samples_Leaf.append(Min_Samples_Leaf[m])
            Sel_Depth.append(Depth[m])
            
            Sel_Tr_Precision.append(Acc_Tr_Precision[m])
            Sel_Tr_Accuracy.append(Acc_Tr_Accuracy[m])
            Sel_Tr_F1_Score.append(Acc_Tr_F1_Score[m])
            Sel_Tr_Recall.append(Acc_Tr_Recall[m])
            
            Sel_Tst_Precision.append(Acc_Tst_Precision[m])
            Sel_Tst_Accuracy.append(Acc_Tst_Accuracy[m])
            Sel_Tst_F1_Score.append(Acc_Tst_F1_Score[m])
            Sel_Tst_Recall.append(Acc_Tst_Recall[m])
            pass
        pass

    # Selection Crieteria 2
    final_Sel_Fold_Number = []
    final_Sel_Model_Graphviz = []
    final_Sel_Min_Samples_Leaf = []
    final_Sel_Depth = []
        
    final_Sel_Tr_Precision = []
    final_Sel_Tr_Accuracy = []
    final_Sel_Tr_F1_Score = []
    final_Sel_Tr_Recall = []
          
    final_Sel_Tst_Precision = []
    final_Sel_Tst_Accuracy = []
    final_Sel_Tst_F1_Score = []
    final_Sel_Tst_Recall = []
   
    if len(Sel_Min_Samples_Leaf) > 0:
       for sm in range(len(Sel_Min_Samples_Leaf)):
           average_tst_metrics = (np.mean([np.array(Sel_Tst_Precision[sm]), 
                                          np.array(Sel_Tst_Accuracy[sm]), 
                                          np.array(Sel_Tst_F1_Score[sm]), 
                                          np.array(Sel_Tst_Recall[sm])]))

           if average_tst_metrics >= avg_tst_metrics:
              final_Sel_Fold_Number.append(Sel_Fold_Number[sm])
              final_Sel_Model_Graphviz.append(Sel_Model_Graphviz[sm])
              final_Sel_Min_Samples_Leaf.append(Sel_Min_Samples_Leaf[sm])
              final_Sel_Depth.append(Sel_Depth[sm])
 
              final_Sel_Tr_Precision.append(Sel_Tr_Precision[sm])
              final_Sel_Tr_Accuracy.append(Sel_Tr_Accuracy[sm])
              final_Sel_Tr_F1_Score.append(Sel_Tr_F1_Score[sm])
              final_Sel_Tr_Recall.append(Sel_Tr_Recall[sm])
                 
              final_Sel_Tst_Precision.append(Sel_Tst_Precision[sm])
              final_Sel_Tst_Accuracy.append(Sel_Tst_Accuracy[sm])
              final_Sel_Tst_F1_Score.append(Sel_Tst_F1_Score[sm])
              final_Sel_Tst_Recall.append(Sel_Tst_Recall[sm])
              pass
           pass
       pass 

 # Models generated and filtered

 # Extracting data from GraphViz
    if len(final_Sel_Model_Graphviz) > 0:
        for model in range(len(final_Sel_Min_Samples_Leaf)):
            Model_Details = DT_DOT_to_DF(final_Sel_Model_Graphviz[model], Model_Details_Out_Path)
            Model_Details.to_csv('' + Model_Details_Out_Path
                                 + 'min_samples_leaf_' + str(final_Sel_Min_Samples_Leaf[model])
                                 + '_depth_' + str(final_Sel_Depth[model])
                                 + '_fold_' + str(final_Sel_Fold_Number[model])
                                 + '.csv')
            Model_Path =  str('' + Model_Details_Out_Path
                              + 'min_samples_leaf_' + str(final_Sel_Min_Samples_Leaf[model])
                              + '_depth_' + str(final_Sel_Depth[model])
                              + '_fold_' + str(final_Sel_Fold_Number[model])
                              + '.csv')
            List_of_Model_Details_Out_Path.append(Model_Path)

            pass
        pass


    Selected_Models_Parameters_and_Performance = pd.DataFrame({'final_Sel_Model_Graphviz'      :pd.Series (final_Sel_Model_Graphviz),
                                                               'fold_number'                   :pd.Series (final_Sel_Fold_Number),
                                                               'Sel_Model_Min Samples per Leaf':pd.Series (final_Sel_Min_Samples_Leaf),
                                                               'Sel_Model_Depth'               :pd.Series (final_Sel_Depth),
                                                               'Sel_Model_Tr_Precision'        :pd.Series (final_Sel_Tr_Precision), 
                                                               'Sel_Model_Tr_Accuracy'         :pd.Series (final_Sel_Tr_Accuracy), 
                                                               'Sel_Model_Tr_F1_Score'         :pd.Series (final_Sel_Tr_F1_Score), 
                                                               'Sel_Model_Tr_Recall'           :pd.Series (final_Sel_Tr_Recall), 
                                                               'Sel_Model_Tst_Precision'       :pd.Series (final_Sel_Tst_Precision), 
                                                               'Sel_Model_Tst_Accuracy'        :pd.Series (final_Sel_Tst_Accuracy), 
                                                               'Sel_Model_Tst_F1_Score'        :pd.Series (final_Sel_Tst_F1_Score),
                                                               'Sel_Model_Tst_Recall'          :pd.Series (final_Sel_Tst_Recall)})
    Selected_Models_Parameters_and_Performance.to_csv('' + Model_Metrics_Out_Path + 'Selected_Models_Parameters_Performance_Metrics.csv')
    return Selected_Models_Parameters_and_Performance, List_of_Model_Details_Out_Path


























#####################################################DT_Models_Parameters_Extraction_Function####################################


def DT_DOT_to_DF(Model_Graph_Viz, Model_Details_Out_Path):
    
    S = Model_Graph_Viz
    #with open (file_name, 'r') as graph_viz_file:
    #     S = graph_viz_file.read()
    
    ###############################################------NODE PATHS------###############################################
    node_to_node = re.findall("(\\d+ -> \\d+)+", S)   #Looking for the expression example '1 -> 2', where 1, and 2 are represented as '\\d'
    
    list_node_to_node = []
    
    for node in node_to_node:
        list_node_to_node.append(node.split(" -> "))
    
    list_node_to_node = np.array(list_node_to_node).astype(np.int).tolist()
    
    def find_path(i, list_node_to_node, out):
        if i == 0:
            return out
        for node in list_node_to_node:
            if node[-1] == i:
                break
        parent, child = node
        out.append(parent)
        find_path(parent, list_node_to_node, out)
        return out
    
    #pdb.set_trace()
    for node in list_node_to_node:    
        node_path = [node[-1]] + find_path(node[-1], list_node_to_node, out = [])
        # print(node_path)
        logging.debug(f"{node_path}")
    #print(nodes_path.node)
    ###############################################------NODE PATHS------###############################################
    
    

    
    
    #######################################################------Node Id , Feature and Threshold------#########################################################
    node_ID_feature_and_threshold = re.findall(' \\;\\\\n(\\d+)+ \\[label=\\"(?:X\\[(\\d+)\\])? <= (\\d+\\.\\d+)+', S , flags = re.DOTALL)
    node_ID = re.findall(" \\;\\\\n(\\d+)+ \\[", S , flags = re.DOTALL)
    
    #node_ID_and_threshold = re.findall(' \\;\\\\n(\\d+)+ \\[label=\\"(?:X\\[\\d+\\])? <= (\\d+\\.\\d+)+', S , flags = re.DOTALL)
    #feature_number = re.findall("X\\[(\\d+)+\\]", S , flags = re.DOTALL)
    #node_threshold = re.findall("<= (\\d+\\.\\d+)+", S)   #Looking for the number that is after '<= ' , where 1, and 2 are represented as '\\d'
    #######################################################------Node Id , Feature and Threshold------#########################################################
    
    
    



    ###############################################------Number of Samples and Number of Samples per Class------###############################################
    no_samples_per_node = re.findall("\\\\nsamples = (\\d+)+", S , flags = re.DOTALL)
    no_samples_per_Class_per_node = re.findall("\\\\nvalue = (\\[(?:\\d+(?:, |\\\\\\\\n)?)+\\])+", S , flags = re.DOTALL)
    ###############################################------Number of Samples and Number of Samples per Class------###############################################
        
 
    
    assert len(node_ID) == len(no_samples_per_node) == len(no_samples_per_Class_per_node)

    
    Extracted = {
                 eval(n_id): dict(n_ID = eval(n_id),
                                  n_Samples_N = eval(nN), 
                                  n_Samples_C = eval(nC.replace('\\\\n', ',')), 
                                  feature = None, 
                                  threshold = None, 
                                  node_path = [eval(n_id)] + find_path(eval(n_id), list_node_to_node, out = [])) 
                 for n_id, nN, nC in zip(node_ID, no_samples_per_node, no_samples_per_Class_per_node)
                 }
    


    for n_id, feature, threshold in node_ID_feature_and_threshold:
        Extracted[eval(n_id)]['feature'] = eval(feature) 
        Extracted[eval(n_id)]['threshold'] = eval(threshold)
    
    
    
    for n_id in Extracted.keys():
        temp_dict_for_Classes = {f'n_Samples_C{i}': n for i, n in enumerate(Extracted[n_id]['n_Samples_C'])}
        Extracted[n_id].update(temp_dict_for_Classes)    
    
    
    
    for n_id in Extracted.keys():
        temp_dict_for_Paths = {f'node_path{i}': n for i, n in enumerate(reversed(Extracted[n_id]['node_path']))}
        Extracted[n_id].update(temp_dict_for_Paths) 
    




    Extracted_Data_Frame = pd.DataFrame(Extracted).T
    

    return Extracted_Data_Frame






























#####################################################DT_Models_Nodes_Features_Threshold_Extraction_Function#################################### 

def Extract_N_ID_Feature_Threshold(Model_Details_Paths,
                                   Imp_Nodes_Path_file,
                                   N_ID_Feature_Threshold_Path_file):

    # Model_Details_Paths = List_of_Model_Details_Out_Path
    nans_in_node_path = []

    # Finding depth
    for i in range(0, 100):
        nans_in_node_path = []
        flag = False
        flag_1 = False 
        for path in Model_Details_Paths:
            try: 
                node_path = pd.read_csv(path, usecols = [f'node_path{i}'])
                node_path = node_path.values
                nans_in_node_path.append(np.count_nonzero(np.isnan(node_path)))

            except ValueError:
                flag_1 = True
                break
        for nan in nans_in_node_path:
            if nan != nans_in_node_path[0]: 
               flag = True 
               break
        if flag_1 or flag:
            break
        
        
    n = i - 1
    # print(f'node_path{i-1}')
    logging.debug(f'node_path{i-1}')
    
    
    node_IDs_Path = dict()
    node_IDs = []
    
    for i in range(0, n):
        
        node_IDs_Path[f'node_ID_path{i}'] = []
        
        for path in Model_Details_Paths:            
            node_path = pd.read_csv(path, usecols = [f'node_path{i}'])
            node_path = node_path.values
            node_IDs_Path[f'node_ID_path{i}'].append(np.unique(node_path).tolist())
            pass
        pass

    # dict to dataframe
    dict_to_data_frame_1 = pd.DataFrame.from_dict(node_IDs_Path)
    
    v = 0
    
    for i in range(0, n):
        column_names = []
        coverted_to_list = dict_to_data_frame_1[f'node_ID_path{i}'].tolist()
        place_begin = v-1
        
        if i == 0:
           v = 1
        else:    
           v = len(coverted_to_list[0]) -1 #len(coverted_to_list[0]) #(2**i) + 1    
           pass

        for j in range(0, v):
            # print(i, j)
            logging.debug(f"{i, j}")

            #pdb.set_trace()
            if i == 0:
               new_data_frame = pd.DataFrame(coverted_to_list,  columns = [f'node_ID_path{i}_{j}'])
            else:
               column_names.append(f'node_ID_path{i}_{j}')
               place = place_begin + j + 1
               new_data_frame.insert(place, column_names[j], [item[j] for item in coverted_to_list])

    new_data_frame.to_csv(Imp_Nodes_Path_file)
    
    
    
    
    p = 0

    logging.info(f"Extracting features up to level {n}...")
    for i in tqdm.trange(0, n):
        column_names_nodes = []
        column_names_features = []
        column_names_thresholds = []
    
        
        place_begin = p + 3     # id, feature, threshold
        
        if i == 0:
            important_nodes = pd.read_csv(Imp_Nodes_Path_file, usecols = [f'node_ID_path{0}_{0}'])
            important_nodes = important_nodes.values
    
            location_nodes = []
            location_features = []
            location_threshold = []
            
            for counter, path in enumerate(Model_Details_Paths):
                
                # print(i, j, counter)
                logging.debug(f"{i, j, counter}")
                nodes = pd.read_csv(path, usecols = ['n_ID'])
                nodes = nodes.values
                Feature = pd.read_csv(path, usecols = ['feature'])
                Feature = Feature.values 
                Threshold = pd.read_csv(path, usecols = ['threshold'])
                Threshold = Threshold.values
                
    
                
                
                Index = nodes.tolist().index(important_nodes[counter])
                location_nodes.append(nodes[Index])
                location_features.append(Feature[Index])
                location_threshold.append(Threshold[Index])
            
                pass
            
            location_nodes = np.array(location_nodes).flatten()
            location_features = np.array(location_features).flatten()
            location_threshold = np.array(location_threshold).flatten()
            Data = np.column_stack([location_nodes, location_features, location_threshold])
    
            important_data_frame = pd.DataFrame(Data, columns = ['node_ID_path_0', 'feature_path_0', 'threshold_path_0']) 
        else:
            p = 2**i
        
            for j in range(0, p):
                try:
                   important_nodes = pd.read_csv(Imp_Nodes_Path_file, usecols = [f'node_ID_path{i}_{j}'])
                   important_nodes = important_nodes.values
                except: ValueError
                    
                location_nodes = []
                location_features = []
                location_threshold = []
                
                column_names_nodes.append(f'node_ID_path_{i}_{j}')
                column_names_features.append(f'feature_path_{i}_{j}')
                column_names_thresholds.append(f'threshold_path_{i}_{j}')
                
                for counter, path in enumerate(Model_Details_Paths):
                    # print(i, j, counter)
                    logging.debug(f"{i, j, counter}")
                    nodes = pd.read_csv(path, usecols = ['n_ID'])
                    nodes = nodes.values
                    Feature = pd.read_csv(path, usecols = ['feature'])
                    Feature = Feature.values 
                    Threshold = pd.read_csv(path, usecols = ['threshold'])
                    Threshold = Threshold.values
                    
                    
                    
                    
                    Index = nodes.tolist().index(important_nodes[counter])
                    location_nodes.append(nodes[Index])
                    location_features.append(Feature[Index])
                    location_threshold.append(Threshold[Index])

                    pass

                location_nodes = np.array(location_nodes).flatten()
                location_features = np.array(location_features).flatten()
                location_threshold = np.array(location_threshold).flatten()
    
                
  
                place_1 = place_begin + j
                place_2 = place_begin + j + 1
                place_3 = place_begin + j + 2
                important_data_frame.insert(place_1, column_names_nodes[j], location_nodes) 
                important_data_frame.insert(place_2, column_names_features[j], location_features) 
                important_data_frame.insert(place_3, column_names_thresholds[j], location_threshold)

                pass
            pass
        pass

    important_data_frame.to_csv(N_ID_Feature_Threshold_Path_file)
    return important_data_frame

