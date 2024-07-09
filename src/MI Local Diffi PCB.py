#print("scikit-multiflow package installation")
#!pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow
#!pip install -U git+https://github.com/Elmecio/IForestASD_based_methods_in_scikit_Multiflow.git
#print("scikit-multiflow package installation")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-  

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time
import datetime
from scipy import stats
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from collections import deque
import pytest



#--------------------------Class NDKSWIN for the drift Detection-----------------------------------#

    ####Class to define the NDKSWIN Method for the drift Detection######
    class NDKSWIN(BaseDriftDetector):
        
        def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None,
                    n_dimensions:int=2, n_tested_samples=0.01,
                    fixed_checked_dimension=False, fixed_checked_sample=False):
            super().__init__()
            self.window_size = window_size
            self.n_tested_samples = n_tested_samples
            self.fixed_checked_dimension = fixed_checked_dimension
            self.fixed_checked_sample = fixed_checked_sample
            self.n_dimensions = n_dimensions
            self.stat_size = stat_size
            self.alpha = alpha
            self.change_detected = False
            self.p_value = 0
            self.n = 0
            if self.alpha < 0 or self.alpha > 1:
                raise ValueError("Alpha must be between 0 and 1")

            if self.window_size < 0:
                raise ValueError("window_size must be greater than 0")

            if self.window_size < self.stat_size:
                raise ValueError("stat_size must be smaller than window_size")

            self.window = data

            if self.n_dimensions <= 0 or (data is not None and self.n_dimensions > data.shape[1]):
                
                self.n_dimensions = data.shape[1]
        
            if self.n_tested_samples <= 0.0 or self.n_tested_samples > 1.0 :
                raise ValueError("n_tested_samples must be between > 0 and <= 1")
            else:
                self.n_samples_to_test = int(self.window_size*self.n_tested_samples)

        def add_element(self, input_value):

            self.change_detected = False

            if self.fixed_checked_dimension:
                sample_dimensions=list(range(self.n_dimensions))
            else:
                if self.n_dimensions > input_value.shape[1]:
                    print("n_dimensions must be between 1 and <= input_value.shape[1]. We will consider the first dimension only to compute the drift detection.")
                    sample_dimensions = [0]
                else:
                    sample_dimensions = random.sample(list(range(input_value.shape[1])), self.n_dimensions)



            if self.fixed_checked_sample:
                sample_test_data = input_value[list(range(self.n_samples_to_test))]
            else:
                if self.n_samples_to_test > input_value.shape[0]:
                    print("Not enough data in input_value to pick "+str(self.n_samples_to_test)+" We will use 100% of input_value.")
                    sample_test_data = input_value
                else:
                    sample_test_data = input_value[random.sample(list(range(input_value.shape[0])), self.n_samples_to_test)]


            for value in sample_test_data:
                
                if self.change_detected == False:
                    self.n += 1
                    currentLength = self.window.shape[0]
                    if currentLength >= self.window_size:
                        self.window = np.delete(self.window, 0,0)


                        for i in sample_dimensions:
                            
                            rnd_window = np.random.choice(np.array(pd.DataFrame(self.window)[i])[:-self.stat_size], self.stat_size)

                            
                            (st, self.p_value) = stats.ks_2samp(rnd_window,
                                                                np.array(pd.DataFrame(self.window)[i])[-self.stat_size:], mode="exact")

                            if self.p_value <= self.alpha and st > 0.1:
                                self.change_detected = True
                                self.window = self.window[-self.stat_size:]
                                
                                break
                            else:
                                self.change_detected = False
                                
                    else:  
                        
                        self.change_detected = False

                    self.window = np.concatenate([self.window, [value]])
                    
                else:
                    
                    break

        def detected_change(self):#Return a boolean value which indicates whether the drift is detected or not.
            return self.change_detected

        def reset(self): #Reset the change detector parameters to default values.
            self.p_value = 0
            self.window = np.array([])
            self.change_detected = False
        
#---------------------------------Class to build Isolation Tree-------------------------------------------#
class IsolationTree:
    def __init__(self, height_limit, current_height, parent=None):
        if height_limit < 0:
            raise ValueError("height_limit must be greater than or equal to 0")
    
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1
        self.max_x = None
        self.min_x = None
        self.q_left = 0
        self.q_right = 0
        self.data_summary = None
        self.parent = parent
        self.pc_value = 0

    def fit_improved(self, X: np.ndarray, improved=False):
        """
        Add Extra while loop
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = len(X)

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        self.min_x = X[:, split_by].min()
        self.max_x = X[:, split_by].max()

        if self.min_x == self.max_x:
            self.exnodes = 1
            self.size = len(X)
            self.data_summary = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }


            return self
        condition = True

        while condition:

            split_value = self.min_x + random.betavariate(0.5,0.5)*(self.max_x-self.min_x)
    

            a = X[X[:, split_by] < split_value]
            b = X[X[:, split_by] >= split_value]
            if len(X) < 10 or a.shape[0] < 0.25 * b.shape[0] or b.shape[0] < 0.25 * a.shape[0] or (
                    a.shape[0] > 0 and b.shape[0] > 0):
                condition = False


            self.size = len(X) # This is 'q'
            self.split_by = split_by
            self.split_value = split_value
            self.q_left = len(a)  # 'q0' for the left child node
            self.q_right = len(b) #'q0' for the right child node

            

            self.left = IsolationTree(self.height_limit, self.current_height + 1,parent=self).fit_improved(a, improved=False)
            self.right = IsolationTree(self.height_limit, self.current_height + 1,parent=self).fit_improved(b, improved=False)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        self.min_x = X_col.min()
        self.max_x = X_col.max()

        if self.min_x == self.max_x:
            self.exnodes = 1
            self.size = len(X)
            self.data_summary = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }


            return self

        else:

            split_value = self.min_x + random.betavariate(0.5, 0.5) * (self.max_x - self.min_x)
        
            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.data_summary = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }


            self.left = IsolationTree(self.height_limit, self.current_height + 1,parent=self).fit(X[w], improved=True)
            self.right = IsolationTree(self.height_limit, self.current_height + 1,parent=self).fit(X[~w], improved=True)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

######------------------------A structured dictionary containing details about the tree.-----------------#######

    def export_tree(self, depth=0):
        """
        Traverses the tree to capture details about each node in a structured format.
        Includes depth of each node and a reference to the parent node ID.

        Args:
            depth (int): The depth of the current node, default is 0 for the root.

        Returns:
            dict: A structured dictionary containing details about the tree.
        """
        # Base information for any node
        node_info = {
            "size": self.size,
            "exnodes": self.exnodes,
            "n_nodes": self.n_nodes,
            "depth": depth,
            "parent_id": id(self.parent) if self.parent else None,
            "data_summary": self.data_summary,
        }

        if self.exnodes == 1:
            return node_info
        else:
            node_info.update({
                "split_by": self.split_by,
                "split_value": self.split_value,
                "left": self.left.export_tree(depth + 1) if self.left else None,
                "right": self.right.export_tree(depth + 1) if self.right else None,
            })
        return node_info


####################Helper functions #################
def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

###################### path length for an observation in a given tree, with optional detailed tracking.#########################      
def path_length_tree(x, t, e, track_info=False):
    """Calculate the path length for an observation in a given tree, with optional detailed tracking.

    Parameters
    ----------
    x : ndarray
        The observation for which the path length is calculated.
    t : IsolationTree or similar
        The tree in which the path length is being calculated.
    e : int
        The current path length.
    track_info : bool, optional
        Whether to track detailed information of splits. Default is False.

    Returns
    -------
    float
        The updated path length.
    list, optional
        A list of detailed split info if tracking is enabled.
    """
    # Check if the current node is None
    if t is None:
        print("Warning: Attempted to access an undefined branch.")
        return e, [{'direction': 'undefined'}] if track_info else e

    # Base case: if the current node is an external node (leaf)
    if t.exnodes == 1:
        path_info = {
            'size': t.size, 
            'type': 'leaf', 
            'depth': e
        }
        return e + c(t.size), [path_info] if track_info else e + c(t.size)

    # Recursive case: node has further splits
    else:
        a = t.split_by
        if t.split_value is not None:
            if x[a] < t.split_value:
                direction = 'left'
                next_node = t.left
            else:
                direction = 'right'
                next_node = t.right

            # Calculate the sizes for parent and child nodes
            parent_size = t.size
            if next_node is not None:
                child_size = next_node.size
            else:
                child_size = 0  # In case the next node has not been initialized

            # Track split information if required
            if track_info:
                split_info = {
                    'feature': a,
                    'split_value': t.split_value,
                    'direction': direction,
                    'depth': e,
                    'parent_size': parent_size,
                    'child_size': child_size,
                    'max_x': t.max_x,  # Ensure these are calculated/defined elsewhere in your tree construction
                    'min_x': t.min_x
                }
            next_e, next_info = path_length_tree(x, next_node, e + 1, track_info)
            return next_e, [split_info] + next_info if track_info else next_e
        else:
            print("Error: split_value is None in a non-leaf node.")
            return e, [{'direction': 'error'}] if track_info else e
##################  IsolationTreeEnsemble  ##################

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10, window_size=100, threshold=0.5, alpha=0.005):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = int(np.ceil(np.log2(sample_size)))  # Ensure it's an integer
        self.trees = []
        self.window_size= deque(maxlen=window_size)  # Sliding window for drift detection
        self.threshold = threshold
        self.alpha = alpha
        self.drift_detector = NDKSWIN(alpha=alpha, window_size=window_size)  # Drift detector
        self.path_details = []  # Dictionary to store path details for each data point
        self.path_data_values = []  # to store data values associated with each path
        self.new_tree_indices = set()  # Set to track indices of new or updated trees 
    def fit(self, X: np.ndarray, improved=False):


        if isinstance(X, pd.DataFrame):
            X = X.values

        len_x = len(X)
        col_x = X.shape[1]
        self.trees = []
        self.path_details = [{} for _ in range(len_x)]  # Initialize a dictionary for each observation
        self.path_data_values = [{} for _ in range(len_x)]  # Initialize a dictionary for each observation's data

        #current_sample_size = min(self.sample_size, len_x)

        if improved:
            for i in range(self.n_trees):
                sample_idx = random.sample(list(range(len_x)), self.sample_size)
                temp_tree = IsolationTree(self.height_limit, 0).fit_improved(X[sample_idx, :], improved=True)
                self.trees.append(temp_tree)
        else:
            for i in range(self.n_trees):
                sample_idx = random.sample(list(range(len_x)), self.sample_size)
                temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :], improved=False)
                self.trees.append(temp_tree)

        return self


    def path_length(self, X: np.ndarray, detailed=False):


        if isinstance(X, pd.DataFrame):
            X = X.values

        path_lengths = []
        detailed_info = []  # Initialize detailed_info list to store details if needed

        for index, x in enumerate(X):
            tree_paths = []
            path_info = []

        for t in self.trees:
            pl, node_info = path_length_tree(x, t, 0, track_info=detailed)
            tree_paths.append(pl)
            if detailed:
                path_info.extend(node_info)  # Collect detailed info from each tree

        average_path_length = np.mean(tree_paths)
        path_lengths.append(average_path_length)
        detailed_info.append(path_info if detailed else [])

        if detailed:
            return np.array(path_lengths).reshape(-1, 1), detailed_info
        else:
            return np.array(path_lengths).reshape(-1, 1), []  # Always return two values


    def anomaly_score(self, X: np.ndarray, detailed=False):
        path_lengths, details = self.path_length(X, detailed=detailed)
    # Calculate normalization constant c for each sample size
        c_values = np.array([c(len_x) for len_x in path_lengths.flatten()])
        scores = 2.0 ** (-path_lengths / c_values)

        if detailed:
            return scores, details
        return scores
        
    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    
    def predict(self, X: np.ndarray, threshold: float):
        scores = self.anomaly_score(X)
        predictions = [1 if score >= threshold else 0 for score in scores]

        return np.array(predictions)
    ################Additional FUnction for MI-Local-Diffi Method.#######################################################
    ########################## ollects detailed information from all trees in the ensemble after they are fitted ###########################
    def get_all_trees_info(self):
        """
        Collects detailed information from all trees in the ensemble after they are fitted.
        Each tree's structure is captured as a dictionary detailing all nodes and splits.

        Returns:
        list of dict - Each dictionary contains detailed structure and split info of a tree.
        """
        all_trees_info = [tree.export_tree() for tree in self.trees]
        return all_trees_info

    def compile_paths(self, paths, path_dict):
        for path in paths:
            feature, value, direction = path
            combined_key = f'{feature},{value},{direction}'
            if combined_key in path_dict:
                path_dict[combined_key] += 1
            else:
                path_dict[combined_key] = 1
        return path_dict

    def store_path_data(self, paths, path_dict, data_values):
        for index, path in enumerate(paths):
            feature, value, direction = path
            combined_key = f'{feature},{value}'  # Note: direction is not used here
            if combined_key in path_dict:
                path_dict[combined_key].append(data_values[index])
            else:
                path_dict[combined_key] = [data_values[index]]
        return path_dict

    def find_outliers(self, X: np.ndarray, percentile=95):
        """
        Detect outliers in the dataset based on the anomaly scores.
        Outliers are defined as observations with anomaly scores above the specified percentile threshold.

        Parameters:
        X : np.ndarray - The dataset to analyze.
        percentile : int - The percentile used to cut off outliers.

        Returns:
        np.ndarray - Array of outlier indices.
        """
        scores = self.anomaly_score(X).flatten()  # Ensure scores are a flat array
        threshold = np.percentile(scores, percentile)  # Define threshold as the given percentile of scores
        outliers = np.where(scores >= threshold)[0]  # Find indices where scores are above the threshold

        return outliers



        #------------------Implementation of PCB-Iforest starts from here onwards --------------------------------#
    #------------------Function to replace the tree in self.tree within IsolationTreensemble------------------#
    def replace_trees(self):
        n_replaced_trees = 0  # Initialize a counter for replaced trees
        for i, tree in enumerate(self.trees):
            if tree.pc_value < 0:  # Assuming each tree has a performance counter `pc_value`
            # Rebuild the tree with current window of data
                self.trees[i] = self.build_tree(np.array(self.window_size))
                tree.pc_value = 0  # Reset the performance counter
                self.new_tree_indices.add(i)  # Track this tree as new
                n_replaced_trees += 1  # Increment the counter
        return n_replaced_trees  # Return the count of replaced trees
    
    #------------------Function to update the performance counter of each tree------------------#
    def update_performance_contributions(self, X: np.ndarray):
        scores, score_E = self.calculate_scores(X)
        for i, tree in enumerate(self.trees):
            score_C_i = scores[i]
        # Capture old PC value for logging
        old_pc_value = tree.pc_value
        if score_E > self.threshold and score_C_i > self.threshold:
            tree.pc_value += 1
        elif score_E <= self.threshold and score_C_i <= self.threshold:
            tree.pc_value += 1
        else:
            tree.pc_value -= 1
        # Print statement for debugging
        print(f"Tree {i} PC value updated from {old_pc_value} to {tree.pc_value}")
        
    #------------------Function to calculate the score of each tree and score of each Ensemble------------------#
    def calculate_scores(self, X: np.ndarray):
        scores = np.array([path_length_tree(X, tree, 0) for tree in self.trees])
        # Add a print statement to show the scores for each tree
        for i, score in enumerate(scores):
            print(f"Tree {i} scores: {score}")
        # Calculate the ensemble score across the correct axis (trees)
            score_E = np.mean(scores, axis=0)
        # Add a print statement to show the ensemble score for each data point
            print(f"Ensemble scores for each data point: {score_E}")
            return scores, score_E

    #------------------Function to build new tree------------------#
    def build_tree(self,window_size):
    # Ensure window_data is not empty
        if window_size.size == 0:
            raise ValueError("Cannot build a tree with no data.")

    # Ensure window_data is a 2D array
        if window_size.ndim == 1:
            window_size = window_size.reshape(1, -1)

    # Create a new IsolationTree instance
        new_tree = IsolationTree(self.height_limit, 0)

    # Fit the tree with the provided data
        new_tree.fit(window_size, improved=False)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_tree_to_json(new_tree, f'new_tree_{timestamp}.json')

    # Return the fitted tree
        return new_tree
    
    #---------------------Function to handle the drift and build new tree within self.trees in Isolationtreeensemble--------------#
    def handle_drift_and_update_ensemble(self, X: np.ndarray):
    # Add new data to the window and remove old data
        self.window_size.append(X)
        print("Updating model with new data point...") # Assuming self.window_size is a deque with a maxlen set to the window size
        if self.drift_detector.detected_change():
            print("Drift detected - updating model.")
        for i, tree in enumerate(self.trees):
            if tree.pc_value < 0:
                # Rebuild the tree with current window of data
                self.trees[i] = self.build_tree(np.array(self.window_size))
                self.trees[i].pc_value = 0
                n_replaced_trees += 1
        print(f"Replaced {n_replaced_trees} trees due to drift.")
        self.drift_detector.reset()
        #---------------------Functions to update the model after the drift detection--------------#    
    def update_model_with_drift_handling(self, X: np.ndarray):
        self.handle_drift_and_update_ensemble(X)

    def calculate_anomaly_scores_post_update(self, X: np.ndarray):
        _, score_E = self.calculate_scores(X)
        return score_E

    def update_model_and_calculate_scores(self, X: np.ndarray):
    # Step 1: Incorporate new data and handle drift
        self.update_model_with_drift_handling(X)

    # Step 2: Calculate anomaly scores for the (same or new) data after model update
        anomaly_scores = self.calculate_anomaly_scores_post_update(X)

        print("Anomaly scores after handling drift and updating the ensemble:", anomaly_scores)

        return anomaly_scores
    
#############################path_length_indicator########################################
def path_length_indicator(path_length, sample_size):
    """
    Calculates the path length indicator used in MI-Local-DIFFI.

    Parameters
    ----------
    path_length : float
        The path length of an observation.
    subspace : int
        Specified subspace for isolation forest.

    Returns
    -------
    float
        Path length indicator score as a float.
    """
    n = sample_size
    PL_lower = 1
    PL_upper = 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)  # Average path length
    p_l_i = max(0.1, min(1, 1 - ((path_length - PL_lower) / (PL_upper - PL_lower))))
    return p_l_i

#############################splitProportion########################################
def splitProportion(q, q0):
    """
    Calculates the split proportion indicator used in MI-Local-DIFFI
    Parameters
    ----------
    q : Number of observations in parent node (aij in your description).
    q0 : Number of observations in child node (qij in your description).
    Returns
    -------
    sp : Split proportion indicator score as float
    """
    if q <= 2:
        sp = 0
    else:
        sp = 1 - ((q0 - 1) / (q - 2))
    return sp


#############################splitProportion########################################
def split_interval_length_indicator(split_value, split_direction, max_x, min_x):
    split_value = float(split_value)
    interval = abs(max_x - min_x)
    a_interval = abs(split_value - min_x)
    b_interval = abs(max_x - split_value)

    # Calculate s based on the direction of the split
    if split_direction == "left":
        s = a_interval / interval
    else:
        s = b_interval / interval

    # Calculate the split interval weight ws
    ws = 1.5 - (1 / (s + 1))

    # Ensure that ws is within the range [0.5, 1]
    ws = max(0.5, min(ws, 1.0))

    return ws

#############################MILocalDIFFI########################################  
    
def MILocalDIFFI(df, outlier_index, ensemble, sample_size, all_trees_info):
    instance = df.iloc[outlier_index]
    features = list(df.columns.values)
    feature_importance = {f: [] for f in features}
    sigma = {feature: 0 for feature in features}
    occurrences = {feature: 0 for feature in features}
    paths_info = []

    for tree, tree_info in zip(ensemble.trees, all_trees_info):
        path_length, path_details = path_length_tree(instance.values, tree, 0, track_info=True)
        path_features = []
        w_PL = path_length_indicator(path_length, sample_size)
        w_SP = []
        w_SI_path = []
        dict_features_PL = {}
        dict_features_SP = {}
        dict_features_SI = {}

        for detail in path_details:
            feature = detail['feature']
            path_features.append(feature)
            # Check for path length contribution
            dict_features_PL.setdefault(feature, []).append(w_PL)
            # Calculate and append split proportion
            sp = splitProportion(detail['parent_size'], detail['child_size'])
            w_SP.append(sp)
            dict_features_SP.setdefault(feature, []).append(sp)
            # Calculate split interval length indicator
            ws = split_interval_length_indicator(detail['split_value'], detail['direction'], detail['max_x'], detail['min_x'])
            w_SI_path.append(ws)
            dict_features_SI.setdefault(feature, []).append(ws)

        paths_info.append({
            'path_length': path_length,
            'path_features': path_features,
            'features_split_lengths': dict_features_PL,
            'features_split_proportions': dict_features_SP,
            'features_split_intervals': dict_features_SI
        })

    # Aggregate scores across all features
    for path in paths_info:
        for feature in features:
            if feature in path['path_features']:
                pl_score = sum(path['features_split_lengths'].get(feature, [0]))
                sp_score = max(path['features_split_proportions'].get(feature, [0]))
                si_score = max(path['features_split_intervals'].get(feature, [0]))
                occurrences[feature] += 1
                sigma[feature] += pl_score * sp_score * si_score

    # Normalize the importance scores
    for feature in sigma:
        if occurrences[feature] > 0:
            sigma[feature] /= occurrences[feature]

    DIFFI_score = sigma
    return DIFFI_score



#######################Finally saving the newly formed trees in json format to transfer it to the server ############
def serialize_tree(tree) -> dict:
    """
    Recursively serialize an IsolationTree to a dictionary.
    """
    if tree is None:
        return None
    tree_dict = {
        'split_by': tree.split_by,
        'split_value': tree.split_value,
        'size': tree.size,
        'exnodes': tree.exnodes,
        'n_nodes': tree.n_nodes,
        'pc_value': tree.pc_value,
        'left': serialize_tree(tree.left),
        'right': serialize_tree(tree.right)
    }
    return tree_dict


def save_tree_to_json(tree, filename):
    """
    Serialize and save the tree to a JSON file.

    Parameters:
        tree (IsolationTree): The tree to serialize and save.
        filename (str): The filename where the tree will be saved.
    """
    tree_dict = serialize_tree(tree)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f, indent=4)
    print(f"Saved new tree to '{filename}'.")
    
    
    
###Pickle Method to save the newly formed trees#####################
import pickle

def save_new_trees(ensemble, base_filename):
    for idx in ensemble.new_tree_indices:
        filename = f"{base_filename}_tree_{idx}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(ensemble.trees[idx], file)
    # Clear the indices after saving
    ensemble.new_tree_indices.clear()


#Function to use the saved new  trees ####
def load_tree(filename):
    with open(filename, 'rb') as file:
        tree = pickle.load(file)
    return tree
