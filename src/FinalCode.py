###Some Part of the code has been reused from https://github.com/Divya-Bhargavi/isolation-forest/projects?query=is%3Aopen###############
###Some of the code ideas has been used from @misc{soderstrom2022interpretable,
"""title={Interpretable Outlier Detection in Financial Data: Implementation of Isolation Forest and Model-Specific Feature Importance},
  author={S{\"o}derstr{\"o}m, Vilhelm and Knudsen, Kasper},
  year={2022}
}"""

#!pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow
#!pip install -U git+https://github.com/Elmecio/IForestASD_based_methods_in_scikit_Multiflow.git
#print("scikit-multiflow package installation")
#!/usr/bin/env python3


import numpy as np
import pandas as pd
import random
import sys
import time
import datetime
import json
import pickle
from scipy import stats
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from collections import deque
from sklearn.metrics import confusion_matrix





###################################### Drift Detection Class ##########################

####Class to define the NDKSWIN Method for the drift Detection######

class NDKSWIN(BaseDriftDetector):
    
    def __init__(self, value_of_alpha=0.005, Length_of_window=100, Length_of_stat=30, data=None,
                no_of_dimensions:int=2, no_tested_samples=0.01,
                fixed_checked_dim=False, fixed_checked_sample=False):
        super().__init__()
        self.Length_of_window = Length_of_window 
        self.no_tested_samples = no_tested_samples
        self.fixed_checked_dim = fixed_checked_dim
        self.fixed_checked_sample = fixed_checked_sample
        self.no_of_dimensions = no_of_dimensions
        self.Length_of_stat = Length_of_stat
        self.value_of_alpha = value_of_alpha
        self.change_detected = False
        self.value_of_p = 0
        self.n = 0
        self.validate_parameters() 
        self.initialize_window(data) 

    def validate_parameters(self):
        # Validation of the parameters to make sure that they are within acceptable ranges
        # Checking the 'value_of_alpha'  between 0 and including 1.
        # Actually this parameter represent a probability or a significance level,and it must be between 0 and 1.
        if not (0 <= self.value_of_alpha <= 1):
            raise ValueError("value_of_alpha must lie between 0 and 1")
        # Make sure 'Length_of_window' is positive ,while 'Length_of_window' represents the number of data points being consider at one time,
        # It  must be a positive int to become a valid window of data.
        if self.Length_of_window <= 0:
            raise ValueError("Length_of_window must be more than 0")
        # Make Sure 'Length_of_stat' is lesser than 'Length_of_window','Length_of_stat' must  be smaller than 'Length_of_window' because it represents a subset
        # of the window being used for statistical calculations for drift detection.
        if self.Length_of_window < self.Length_of_stat:
            raise ValueError("Length_of_stat must be lesser than Length_of_window")
        # see that 'no_tested_samples' lies between 0 and including 1.
        # This value show a portion of the total samples that should be tested,so it must lie within these range to be logically true.
        if not (0 < self.no_tested_samples <= 1):
            raise ValueError("no_tested_samples must lie be between > 0 and <= 1")

    def initialize_window(self, data):
    # Initialization the data window for drift detection.
    
    # if there is no data provided at initialization, set the window to an empty or null numpy array.
    
        if data is None:
            self.window = np.array([])
        else:
        # If there is data provided, initialization the window with this data.
            self.window = data

        # check the 'no_of_dimensions' that if the number of dimensions provided exceeds the dimensions of the data.
        # This step checks that that the class does not attempt to access non-existent dimensions, which can cause errors.
        if self.no_of_dimensions > data.shape[1]:
            self.no_of_dimensions = data.shape[1]

    # Calculate number.of.samples to test in each step check for drift.
    # 'no_tested_samples' is to be a fraction; multiply this by 'Length_of_window' which gives the actual number of samples to test and convert it to integer.
    # --This value calcukate how many samples are randomly-- selected--- from the window during drift detection
        self.n_samples_to_test = int(self.Length_of_window * self.no_tested_samples)


    def insert_element(self, input_value):
    # method  to adds a new element or stream of elements to the window and checks for  drift.

    # restore_defaults the change detection  so that it shows the current method call's result.
        self.change_detected = False

    # To Determine which dimensions of the data to be checked based on whether the dimension checking is fixed.
        if self.fixed_checked_dim:
        # If dimension checking is fixed, then use all dimensions specified by 'no_of_dimensions'.
            sample_dimensions = list(range(self.no_of_dimensions))
        else:
        #If not fixed, random the selection of 'no_of_dimensions' from the input data.
            sample_dimensions = random.sample(list(range(input_value.shape[1])), self.no_of_dimensions)

    #Determine if sample checking is fixed to be used for drift detection.
        if self.fixed_checked_sample:
        # If sample checking is already fixed, use the first 'n_samples_to_test' samples from the input.
            sample_test_data = input_value[list(range(self.n_samples_to_test))]
        else:
        # If sample checking is not fixed, randomly select 'n_samples_to_test' samples from the input.
            if self.n_samples_to_test > input_value.shape[0]:
                print(f"Not enough data in input_value to pick {self.n_samples_to_test} samples. Using 100% of input_value.")
                sample_test_data = input_value
            else:
                sample_indices = random.sample(list(range(input_value.shape[0])), self.n_samples_to_test)
                sample_test_data = input_value[sample_indices]

    # Process each sampled value to update window and drift detection.
        for value in sample_test_data:
            if not self.change_detected:
                self.n += 1  # Increase the counter of total processed samples.
            # See if the window is full.
            if self.window.shape[0] >= self.Length_of_window:
                # If the window is full, delete or remove the oldest entry.
                self.window = np.delete(self.window, 0, 0)
                # do drift detection for each and every selected dimension.
                for i in sample_dimensions:
                    # Select 'Length_of_stat' samples from the historical window data randomly.
                    rnd_window = np.random.choice(pd.DataFrame(self.window)[i].iloc[:-self.Length_of_stat], self.Length_of_stat)
                    # Compare the random sample to the most recent statistics by a two-sample Kolmogorov-Smirnov test.
                    (st, self.value_of_p) = stats.ks_2samp(rnd_window, pd.DataFrame(self.window)[i].iloc[-self.Length_of_stat:], mode="exact")
                    # Check if the p-value is below the threshold and the statistic is significant.
                    if self.value_of_p <= self.value_of_alpha and st > 0.1:
                        self.change_detected = True
                        # If there is a drift, restore_defaults the window for the latest statistics in order to focus on new trends.
                        self.window = self.window[-self.Length_of_stat:]
                        break
            # Add the recent value to the window.
            self.window = np.concatenate([self.window, [value]])


    def observed_change(self):
        """
    Check to see if there has been a change (drift), in the data stream.

    Outcome;
    A value; if drift was identified in the most recent segment of data checked, False if not.
    This function is usually used once new data has been incorporated into the detector 
    to establish whether any statistical variances have triggered the detection of drift, within the state.
    """
        return self.change_detected

    def restore_defaults(self):
        """restore_defaults the drift detector back, to its settings.
        Use this function to restore_defaults the detectors settings essentially starting the detection process from scratch. 
        It comes in handy when there have been changes in the data stream or when starting an analysis of a new data stream. 
        This action will erase all gathered data set the p value back, to its default and turn off the drift detection indicator."""
        
        self.value_of_p = 0 #restore_defaults the value_of_p to it default value.
        self.window = np.array([]) # Empty the window of all data.
        self.change_detected = False  #Intialize the original State.
        
#---------------------------------Class to build Isolation Tree-------------------------------------------#
class IsolationTree:
    def __init__(self, ht_limit, current_ht, parent=None):
        #make sure the height limit is Non-Negative.
        if ht_limit < 0:
            raise ValueError("ht_limit must be more than or equal to 0")
        #Intialize the variables for the Tree
        self.split_feature = None    # current split  feature  Index 
        self.value_of_split = None  #  threshold value by which current feature is split.
        self.current_ht = current_ht  #Tree's current height at the time of the recursive building.
        self.ht_limit = ht_limit  # allowable maximum height/depth of the tree.
        self.child_right = None  #Right Child of the current node.
        self.child_left = None   #Left  Child of the current node.
        self._ex_nodes = 0    #No .of External Leaf Nodes.
        self.Total_nodes = 1  #No. of nodes total in the tree.         
        self.size = 0         #No. of samples at the current node. 
        self.x_max = None     #  Maximum values of the splitting feature at the current node.
        self.x_min = None     #  Minimum values of the splitting feature at the current node.
        self.q_left = 0       # Size of left splits...
        self.q_right = 0      # Size of right splits...
        self.summary_data = None  # Captures summary information statistics of the data at the node.
        self.parent = parent # Parent node reference
        self.pc_value = 0  #Intialize Performace counter value of Each tree.

    def optimize_fit(self, X: np.ndarray, optimize=False):
        

    # Check if the tree should become a leaf node due to constraints in datasize or height .
        to_be_leaf = (len(X) <= 1) or (self.current_ht >= self.ht_limit)

        if to_be_leaf:
    # Initialize as a leaf node since either data is insufficient or height limit reached.
            self._ex_nodes = 1
            self.size = len(X)
            return self


        split_feature = random.choice(np.arange(X.shape[1]))
        self.x_min, self.x_max = X_col.min(), X_col.max()
        X_col = X[:, split_feature]
        

        if self.x_min == self.x_max:
            # If there's no changeability in the feature, treat it as a leaf node.
            self._ex_nodes = 1
            self.size = len(X)
            self.summary_data = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }
            return self
        
        # Continuing by splitting until a proper split point is found.
        condition = True
        while condition:
            value_of_split = self.x_min + random.betavariate(0.5,0.5)*(self.x_max-self.x_min)
            a,b = X[X[:, split_feature] < value_of_split],X[X[:, split_feature] >= value_of_split]
            #See that split results in a proper significant partition
            if len(X) < 10 or a.shape[0] < 0.25 * b.shape[0] or b.shape[0] < 0.25 * a.shape[0] or (
                    a.shape[0] > 0 and b.shape[0] > 0):
                condition = False


            
            self.split_by = split_feature
            self.value_of_split = value_of_split
            self.q_left = len(a)  
            self.q_right = len(b) 

            
            #Continously fit the right and left child.
            self.child_left = IsolationTree(self.ht_limit, self.current_ht + 1,parent=self).optimize_fit(a, optimize=False)
            self.child_right = IsolationTree(self.ht_limit, self.current_ht + 1,parent=self).optimize_fit(b, optimize=False)
            self.Total_nodes = self.child_left.Total_nodes + self.child_right.Total_nodes + 1

        return self

    def fit(self, X:np.ndarray, optimize=False):


        if len(X) <= 1 or self.current_ht >= self.ht_limit:
            self._ex_nodes = 1
            self.size = X.shape[0]
            return self

        split_feature = random.choice(np.arange(X.shape[1]))
        self.x_min, self.x_max = X_col.min(), X_col.max()
        X_col = X[:, split_feature]
        

        if self.x_min == self.x_max:
            self._ex_nodes = 1
            self.size = len(X)
            self.summary_data = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }
            return self

        else:
            value_of_split = self.x_min + random.betavariate(0.5, 0.5) * (self.x_max - self.x_min)
            w = np.where(X_col < value_of_split, True, False)
            self.summary_data = {
            'split_feature_mean': X_col.mean(),
            'split_feature_std': X_col.std(),
            'num_samples': self.size
        }
            del X_col

            self.size = X.shape[0]
            self.split_feature = split_feature
            self.value_of_split = value_of_split

            self.child_left = IsolationTree(self.ht_limit, self.current_ht + 1,parent=self).fit(X[w], optimize=True)
            self.child_right = IsolationTree(self.ht_limit, self.current_ht + 1,parent=self).fit(X[~w], optimize=True)
            self.Total_nodes = self.child_left.Total_nodes + self.child_right.Total_nodes + 1

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
            "size": self.size, #No. of samples at the nodes
            "_ex_nodes": self._ex_nodes, #No. of external nodes
            "Total_nodes": self.Total_nodes, #Total no. of Nodes in the subtree with this node. 
            "depth": depth, #Depth pf the node of the tree.
            "parent_id": id(self.parent) if self.parent else None, # Memory ID of the  parent node.
            "summary_data": self.summary_data, #Summary of the data at this node.
        }
        #Condition to whether the node is leaf node or not.
        if self._ex_nodes == 1:
            return node_info
        else:
            #if the node is not a leaf node, then collect all the information relevant for the Feaature Information.
            node_info.update({  
                "split_feature": self.split_feature,    
                "value_of_split": self.value_of_split,
                "left": self.child_left.export_tree(depth + 1) if self.child_left else None,
                "right": self.child_right.export_tree(depth + 1) if self.child_right else None,
            })
        return node_info


####################Helper functions #################
def c(n):


    if n > 2:
        # Use the harmonic number approximation and adjust for finite sample sizes.
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        # Special case for n=2, directly return 1 as per isolation forest properties.
        return 1
    if n == 1:
        # For a single sample, the path length adjustment is 0.
        return 0

###################### path length for an observation in a given tree, with optional detailed tracking.#########################      
def _tree_path_length(x, t, e, track_info=False):

    # Check if the current node is None
    # Check if the current node is None, which indicates an error in tree navigation.
    if t is None:
        print("Warning: Attempted to access an undefined branch.")
        return e, [{'direction': 'undefined'}] if track_info else e

    # Base case: if the current node is an external node (leaf)
    # Handle the leaf node scenario where no further splits are possible.   
    if t._ex_nodes == 1:
        path_info = {
            'size': t.size, 
            'type': 'leaf', 
            'depth': e
        }
        return e + c(t.size), [path_info] if track_info else e + c(t.size)

    # Recursive case: node has further splits
    # Process the internal node with further splits.
    else:
        feature_index = t.split_feature
        if t.value_of_split is not None:
            # Determine the direction of traversal based on the split condition.
            if x[feature_index] < t.value_of_split:
                direction = 'left'
                next_node = t.child_left
            else:
                direction = 'right'
                next_node = t.child_right

            # Calculate the sizes for parent and child nodes
            parent_size = t.size
            if next_node is not None:
                child_size = next_node.size
            else:
                child_size = 0  # In case the next node has not been initialized

            # Track split information if required
            # Prepare split information if detailed tracking is enabled.
            if track_info:
                split_info = {
                    'feature': feature_index,
                    'value_of_split': t.value_of_split,
                    'direction': direction,
                    'depth': e,
                    'parent_size': parent_size,
                    'child_size': child_size,
                    'x_max': t.x_max,  # Ensure these are calculated/defined elsewhere in your tree construction
                    'x_min': t.x_min
                }
            # calculate length of the path in the selected subtree.
            next_e, next_info = _tree_path_length(x, next_node, e + 1, track_info)
            return next_e, [split_info] + next_info if track_info else next_e
        else:
            # Error if there is no good or valid split value in a non-leaf node.
            print("Error: value_of_split is None in a non-leaf node.")
            return e, [{'direction': 'error'}] if track_info else e
        
        
        
##################  PCB_Isolation_Forest  ##################

class PCB_Isolation_Forest:
    def __init__(self, size_of_sample, no_of_trees=10, Length_of_window=100, threshold=0.5, value_of_alpha=0.005):
        """Initialize an ensemble of isolation trees for anomaly detection.
        
        Parameters:
        size_of_sample : int
            The number of samples to train each isolation tree.
        no_of_trees : int
            The number of trees to be included in the forest.
        Length_of_window : int
            The size of the window for drift detection.
        threshold : float
            The threshold for decision making in drift detection.
        value_of_alpha : float
            Significance level for statistical tests in drift detection.

        Attributes:
        trees : list
            List to store each tree in the ensemble.
        ht_limit : int
            The maximum height of trees in the forest.
        Length_of_window : deque
            Sliding window for drift detection using a deque for efficient front removal.
        drift_detector : NDKSWIN
            Drift detector instance for monitoring data streams.
        path_details : list
            List to store path details for each data point for analysis.
        path_data_values : list
            List to store data values associated with each path for analysis.
        new_tree_indices : set
            Set to track indices of new or updated trees in response to detected drift.
        """

        
        self.no_of_trees = no_of_trees #no. of trees to be in one forest.
        self.size_of_sample = size_of_sample # no. of samples to be taken from the dataset to train each isolation tree.
        self.trees = [] # list to store every tree in the ensemble.
        self.ht_limit = int(np.ceil(np.log2(size_of_sample)))  # Ensure it's an integer
        self.Length_of_window= deque(maxlen=Length_of_window)  # Sliding window for drift detection
        self.threshold = threshold
        self.value_of_alpha = value_of_alpha
        self.drift_detector = NDKSWIN(value_of_alpha=value_of_alpha, Length_of_window=Length_of_window)  # Drift detector
        self.path_details = []  # Dictionary to store path details for each data point
        self.path_data_values = []  # to store data values associated with each path
        self.new_tree_indices = set()  # Set to track indices of new or updated trees 
        
        
    def fit(self, X: np.ndarray, optimize=False):
        """Fit the group of the trees to the data, either using an optimize method or the normal one.

        Parameters:
        X : np.ndarray or pd.DataFrame
            The data on which the group of the trees is trained.
        optimize : bool
            option to choose between the optimize or normal fit method.

        Returns:
        self : object
            The instance of the Isolation forest.
        """
        # Convert DataFrame to numpy array if necessary


        X = X.values if isinstance(X, pd.DataFrame) else X

        len_x = len(X)
        col_x = X.shape[1]
        self.path_details = [{} for _ in range(len_x)]  # Initialize a dictionary for each observation
        self.path_data_values = [{} for _ in range(len_x)]  # Initialize a dictionary for each observation's data
        self.trees = []

        #current_size_of_sample = min(self.size_of_sample, len_x)
        ## Training trees on randomly selected samples from the dataset.
        for i in range(self.no_of_trees):
            sample_idx = random.sample(range(len_x), min(self.size_of_sample, len_x))
            tree = IsolationTree(self.ht_limit, 0)
            temp_tree = tree.optimize_fit(X[sample_idx, :], optimize=True) if optimize else tree.fit(X[sample_idx, :], optimize=False)
            self.trees.append(temp_tree)

        return self


    def path_length(self, X: np.ndarray, detailed=False):

    # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        path_lengths = []   # List to store the  path length average of every sample  
        detailed_info = []  # Initialize detailed_info list to store details if needed

        # Iterate over each sample in X
        for index, x in enumerate(X):
            tree_paths = []  # List to store path lengths from each tree for the current sample
            path_info = []   # List to store detailed path information for the current sample if detailed

        # Calculate path length for the sample in each tree
            for t in self.trees:
                pl, node_info = _tree_path_length(x, t, 0, track_info=detailed)
                tree_paths.append(pl)
                if detailed:
                    path_info.extend(node_info)  # Collect detailed info from each tree
                    
        # Compute the average path length for the current data.
        average_path_length = np.mean(tree_paths)
        path_lengths.append(average_path_length)
        detailed_info.append(path_info if detailed else [])
        
        # Return results
        path_lengths = np.array(path_lengths).reshape(-1, 1)  # Reshape as column vector for consistency.
        if detailed:
            return np.array(path_lengths).reshape(-1, 1), detailed_info
        else:
            return np.array(path_lengths).reshape(-1, 1), []  # Always return two values to maintain consistency in output structure


    

    def outlier_score(self, X: np.ndarray, detailed=False):
        """
    Calculate outlier scores for each data value in the dataset X based on their path lengths in the isolation forest.

    Parameters:
    X : np.ndarray or pd.DataFrame
        dataset for which outlier scores are to be calculated.
    detailed : bool, optional
        option to determine if detailed path information should be returned.

    Returns:
    np.ndarray
        An array of outlier scores for each sample.
    list, optional
        Detailed information about each sample's path, returned only if detailed is True.
    """
    # Calculate the path lengths and optionally the detailed path info
        path_lengths, details = self.path_length(X, detailed=detailed)

    # Calculate the 'c' for the expected path length of an unsuccessful search in a BST
    # Don't path_lengths.flatten()  used to find 'c' values. directly use the size of the sample .
        c_values = np.array([c(self.size_of_sample) for _ in range(len(path_lengths))])

    # Calculate anomaly scores using the formula: 2^(-path_length / c)
    # make sure path_lengths and c_values are use correctly if needed.
        scores = 2.0 ** (-path_lengths / c_values.reshape(-1, 1))

        if detailed:
            return scores, details
        return scores
    


    def predict_from_outlier_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """


    Parameters:
    scores : np.ndarray
        An array of anomaly scores calculated for each sample.
    threshold : float
        The cutoff value above which a sample is considered anomalous.

    Returns:
    np.ndarray
        An array of binary predictions: 1 indicates an anomaly, 0 indicates normal.
    """

    # Using NumPy to vectorize , which is fast and  memory efficient.
        predictions = (scores >= threshold).astype(int)

        return predictions


    
    def predict(self, X: np.ndarray, threshold: float):
        """
    estimate whether the data points in X are outliers or not based on a determined threshold.
    
    Inputs:
    X : np.ndarray or pd.DataFrame
        The dataset to Estimate on.
    threshold : float
        The cutoff outlier score above which data points are evaluated outliers.
    
    Returns:
    np.ndarray
        An array of binary Estimation where 1 signifies an outlier and 0 indicates normal.
    """
    # Calculate anomaly scores for all data points in X
        scores = self.outlier_score(X)

    # Convert scores to binary predictions using vectorized operations
        predictions = (scores >= threshold).astype(int)

        return predictions
    ################-------Additional Function for MI-Local-Diffi Method----#######################################################
    ########################## Collect detailed information from all trees in the group of trees after they are fitted ###########################
    def get_all_trees_info(self):
        
        """ Gathers and delivers detailed information from every trees in the forest.
    After fitting the trees, we utilize the function to analyze their structure and decisions.
    
    Returns: -------
    The list of dicts contains the detailed structure and split information for each tree in the forest. 
    This has nodes, splits, and other important data.
    Start the export_tree method on each tree in the forest using a list comprehension. 
    This method should return a complete dictionary about the tree structure.
    """
        all_trees_info = [tree.export_tree() for tree in self.trees]

        return all_trees_info


    def compile_paths(self, paths, path_dict):
        """
    Compiles path informa into a dictionary to count occurrences of every path .

    Parameters:
    paths : list of tuples
        A list where each tuple has information about a path about (feature, value, direction).
    path_dict : dict
        A dictionary where every key is a unique path configuration and the value is the count of configuration.

    Return:
    dict:
        The updated dictionary with path counts based on the given paths list.
    """
    # Iterate over every path in the list of paths
        for path in paths:
            feature, value, direction = path  # Unpack the tuple into variables for easiness
            combined_key = f'{feature},{value},{direction}'  # Create a unique key for the dictionary

        # Increase the count for this path in the dictionary
            if combined_key in path_dict:
                path_dict[combined_key] += 1
            else:
                path_dict[combined_key] = 1

        return path_dict


    def store_path_data(self, paths, path_dict, data_values):
        """
    Stores data values associated with each path configuration in a dictionary.

    Parameters:
    paths : list of tuples
        A list of paths where each tuple contains the path's feature, value, and direction.
    path_dict : dict
        A dictionary where keys are unique combinations of feature and value, and values are lists of data values.
    data_values : list
        List of data values corresponding to each path.

    Returns:
    dict:
        The updated dictionary with each path configuration mapped to relevant data values.
    """
    # Iterate over the paths and  data values
        for index, path in enumerate(paths):
            feature, value, direction = path  # Unpack the tuple to get feature, value, direction
            combined_key = f'{feature},{value}'  # Form a key using feature and value only.

        # add data value to the list associated with the key in the dictionary
            if combined_key in path_dict:
                path_dict[combined_key].append(data_values[index])
            else:
                path_dict[combined_key] = [data_values[index]]

        return path_dict

    def find_outliers(self, X: np.ndarray, percentile=95):

    # Calculate anomaly scores for the dataset
        scores = self.outlier_score(X).flatten()  # Flatten to ensure a 1D array for percentile calculation

    # Calculate the score threshold based on the specified percentile
        threshold = np.percentile(scores, percentile)  

    # Determine indices of the data points whose scores are above the threshold
        outliers = np.where(scores >= threshold)[0]  # Extract indices of scores surpassing the threshold

        return outliers




    #########------------------Implementation of PCB-Iforest starts from here onwards --------------------------------#
    #------------------Function to replace the tree in self.tree within IsolationTreensemble------------------#
    def replace_trees(self):
        """
    Replace trees in the group of the trees whose performance counter value is below zero using the data in the window.

    Return:
    integer
        The no. of trees replaced in the group of the trees.
    """
        n_replaced_trees = 0  # Initialize a counter for replaced trees
        for i, tree in enumerate(self.trees):
            if tree.pc_value < 0:  # Assuming each tree has a performance counter `pc_value`which is intial value to be Zero.
            # Rebuild the tree with current window of data
                self.trees[i] = self.build_tree(np.array(self.Length_of_window))
                tree.pc_value = 0  # restore_defaults the performance counter
                self.new_tree_indices.add(i)  # Track this tree as new
                n_replaced_trees += 1  # Increment the counter
        return n_replaced_trees  # Return the count of replaced trees
    
    #------------------Function to update the performance counter of each tree------------------#
    def update_performance_contributions(self, X: np.ndarray):
        """
    Update the performance counter for every tree in the forest based on
    the comparing of individual tree scores and forest scores to a defined threshold.

    Parameters:
    X : np.ndarray
        The dataset used to calculate scores that determine tree performance.

    Returns:
    None
        Updates a performance counter of the trees.
    """
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
        """
    Calculate  score for each tree in the forest for the dataset data value and the average ensemble score.

    Parameters:
    X : np.ndarray
        The dataset on which we have to calculate the scores.

    Returns:
    tuple
        A tuple containing an array of scores for every tree and the average score for all trees.
    """
        scores = np.array([self.outlier_score(X, tree) for tree in self.trees])
        # Add a print statement to show the scores for each tree
        for i, score in enumerate(scores):
            print(f"Tree {i} scores: {score}")
        # Calculate the ensemble score across the correct axis (trees)
            score_E = np.mean(scores, axis=0)
        # Add a print statement to show the ensemble score for each data point
            print(f"Ensemble scores for each data point: {score_E}")    
            return scores, score_E

    #------------------Function to build new tree------------------#
    def build_tree(self,Length_of_window):
        """
    Build a new IsolationTree using the given window data.

    

    Returns:
    IsolationTree
        A newly fitted isolation tree.

    Raises:
    ValueError
        If the input data is empty.
    """
    # Ensure window_data is not empty
        if Length_of_window.size == 0:
            raise ValueError("Cannot build a tree with no data.")

    # Ensure window_data is a 2D array
        if Length_of_window.ndim == 1:
            Length_of_window = Length_of_window.reshape(1, -1)

    # Create a new IsolationTree instance
        new_tree = IsolationTree(self.height_limit, 0)

    # Fit the tree with the provided data
        new_tree.fit(Length_of_window, improved=False)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_tree_to_json(new_tree, f'new_tree_{timestamp}.json')

    # Return the fitted tree
        return new_tree
    
    #---------------------Function to handle the drift and build new tree within self.trees in PCB_Isolation_Forest--------------#
    def handle_drift_and_update_ensemble(self, X: np.ndarray):
        """
    Updates data window, checks for drift, and replaces underperforming trees as needed.

    Parameters: X: np.ndarray.
        The new data point or points will be added to the model's window.

    Note: Assumes self.Length_of_window is a deque with a maxlen parameter that manages the window size automatically.


    """

    # Add new data to the window and remove old data
        self.Length_of_window.append(X)
        print("Updating model with new data point...") # Assuming self.Length_of_window is a deque with a maxlen set to the window size
        if self.drift_detector.observed_change ():
            print("Drift detected - updating model.")
        for i, tree in enumerate(self.trees):
            if tree.pc_value < 0:
                # Rebuild the tree with current window of data
                self.trees[i] = self.build_tree(np.array(self.Length_of_window))
                self.trees[i].pc_value = 0
                n_replaced_trees += 1
        print(f"Replaced {n_replaced_trees} trees due to drift.")
        self.drift_detector.restore_defaults()
        #---------------------Functions to update the model after the drift detection--------------#    
    def update_model_with_drift_handling(self, X: np.ndarray):
        self.handle_drift_and_update_ensemble(X)

    def calculate_outlier_scores_post_update(self, X: np.ndarray):
        """
    Calculate anomaly scores for the dataset after potentially updating the model for drift.

    Parameters:
    X : np.ndarray
        The dataset for which outlierScores are to be calculated.

    Returns:
    np.ndarray
        An array of anomaly scores for the dataset.
    """
        _, score_E = self.calculate_scores(X)
        return score_E

    def update_model_and_calculate_scores(self, X: np.ndarray):
        """
    Update the model by handling any detected drift with the incoming data, then calculate
    outlierScore for the dataset.

    Parameters:
    X : np.ndarray
        The dataset used for updating the model and calculating anomaly Scores.

    Returns:
    np.ndarray
        An array of outlier Scores for the dataset after handling drift and updating the model.
    """
    # Step 1: Incorporate new data and handle drift
        self.update_model_with_drift_handling(X)

    # Step 2: Calculate anomaly scores for the (same or new) data after model update
        outlier_scores = self.calculate_outlier_scores_post_update(X)

        print("outlierScores after handling drift and updating the ensemble:", outlier_scores)

        return outlier_scores
    
#############################path_length_indicator########################################
def path_length_indicator(path_length, size_of_sample):
    """
    Calculates the path length indicator used in MI-Local-DIFFI.

    Parameters
    ----------
    path_length : float
        The path length of an observation.
    size_of_sample : int
        Specified size_of_sample for isolation forest.

    Returns
    -------
    float
        Path length indicator score as a float.
    
    """
    ##eulers_constant = 0.5772156649  # Euler's constant
    n = size_of_sample
    PL_lower = 1  # Minimum path length of an isolation tree
    PL_upper = 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)  # Average path length
    p_l_i = max(0.1, min(1, 1 - ((path_length - PL_lower) / (PL_upper - PL_lower))))
    return p_l_i

#############################splitProportion########################################
def splitProportion(q, q0):
    """
    Calculates the split proportion indicator used in MI-Local-DIFFI
    Parameters
    
    q : No. of obs in parent node.
    q0 : No. of obs in child node.

        """
    if q <= 2:# No meaningful split can occur if there are 2 or fewer observations
        sp = 0
    else:
        # Calculate the proportion of observations effectively left in the parent node after the split
        sp = 1 - ((q0 - 1) / (q - 2))
    return sp


#############################splitProportion########################################
def split_interval_length_indicator(value_of_split, direction_of_split, x_max, x_min):
    """
    Calculates a split interval length indicator for how balanced a split is within a given interval.
    direction_of_split : str
        The direction of the split, either "left" or "right", indicating which side of the split is isolated.
    value_of_split : float or int
        The value where the split occurs.
    x_max : float or int
        max value in the range of data.
    x_min : float or int
        min value in the range of data.

    Output:
    -------
    float
        The split interval length values lies between 0.5 and 1.0.
    """
    value_of_split = float(value_of_split) # Ensure value_of_split is a float for precise calculations
    interval = abs(x_max - x_min) # Calculate the total interval length
    interval_of_a = abs(value_of_split - x_min) # Determine intervals to the left and right of the split
    interval_of_b = abs(x_max - value_of_split)

    # Calculate s based on the direction of the split
    if direction_of_split == "left":
        s = interval_of_a / interval
    else:
        s = interval_of_b / interval

    # Calculate the split interval weight ws
    ws = 1.5 - (1 / (s + 1))

    # Ensure that ws is within the range [0.5, 1]
    ws = max(0.5, min(ws, 1.0))

    return ws

#############################_MI_Local_DIFFi########################################  
    
def MI_Local_DIFFI(df, outlier_index, ensemble, size_of_sample, all_trees_info):
    """
    Calculate the _MI_Local_DIFFi scores for a  outlier based on
    group of  the trees and the path taken by them.

    Parameters:
    df : pd.DataFrame
        The dataset with data.
    outlier_index : integer value
        Index of the outlier in the dataframe.
    ensemble : Ensemble
        The group of trees used for isolation forest.
    size_of_sample : int
        The number of samples used to build each tree in the forest.
    all_trees_info : list
        Information about all the trees in the ensemble.

    Returns:
    dict
        DIFFI scores for each feature, indicating their importance in isolating the outlier.
    """
    instance = df.iloc[outlier_index]
    features = list(df.columns)
    feature_importance = {f: [] for f in features}
    sigma = {feature: 0 for feature in features}
    occurrences = {feature: 0 for feature in features}
    paths_info = []

    # Iterate over trees and their corresponding information
    for tree, tree_info in zip(ensemble.trees, all_trees_info):
        path_length, path_details = _tree_path_length(instance.values, tree, 0, track_info=True)
        path_features = []
        w_PL = path_length_indicator(path_length, size_of_sample)
        w_SP = []
        w_SI_path = []
        dict_features_PL = {}
        dict_features_SP = {}
        dict_features_SI = {}

        # Process each detail in the path information
        for detail in path_details:
            feature = detail['feature']
            path_features.append(feature)

            # Path length contributions
            dict_features_PL.setdefault(feature, []).append(w_PL)

            # Split proportion
            sp = splitProportion(detail['parent_size'], detail['child_size'])
            w_SP.append(sp)
            dict_features_SP.setdefault(feature, []).append(sp)

            # Split interval length indicator
            ws = split_interval_length_indicator(detail['value_of_split'], detail['direction'], detail['x_max'], detail['x_min'])
            w_SI_path.append(ws)
            dict_features_SI.setdefault(feature, []).append(ws)

        paths_info.append({
            'path_length': path_length,
            'path_features': path_features,
            'features_split_lengths': dict_features_PL,
            'features_split_proportions': dict_features_SP,
            'features_split_intervals': dict_features_SI
        })

    # Aggregate and normalize scores for each feature
    for path in paths_info:
        for feature in features:
            if feature in path['path_features']:
                pl_score = sum(path['features_split_lengths'].get(feature, [0]))
                sp_score = max(path['features_split_proportions'].get(feature, [0]))
                si_score = max(path['features_split_intervals'].get(feature, [0]))
                occurrences[feature] += 1
                sigma[feature] += pl_score * sp_score * si_score

    # Normalize the importance scores by the number of occurrences
    for feature in sigma:
        if occurrences[feature] > 0:
            sigma[feature] /= occurrences[feature]

        DIFFI_score = sigma
        return DIFFI_score


import json
#######################Finally saving the newly formed trees in json format to transfer it to the server ############
def serialize_tree(tree) -> dict:
    """
    Recursively serialize an IsolationTree to a dictionary format suitable for JSON serialization,
    storage, or detailed analysis.

    Parameters:
    tree : IsolationTree
        An instance of IsolationTree or a similar tree structure with defined attributes.

    Returns:
    dict
        A dictionary representing the tree with all its nodes and attributes or None if the tree is None.
    """
    # Handle the base case where the tree node does not exist
    if tree is None:
        return None

    # Serialize the current node and recursively serialize its children
    tree_dict = {
        'split_feature': tree.split_feature,   # Index of the feature used for splitting
        'value_of_split': tree.value_of_split,       # Value of the feature at the split point
        'size': tree.size,                     # Number of samples at the node
        '_ex_nodes': tree._ex_nodes,           # Indicates if the node is an external (leaf) node
        'Total_nodes': tree.Total_nodes,       # Total number of subnodes including this node
        'pc_value': tree.pc_value,             # Performance counter value for the tree
        'child_left': serialize_tree(tree.child_left),   # Recursively serialize the left child
        'child_right': serialize_tree(tree.child_right)  # Recursively serialize the right child
    }

    return tree_dict


import json

def save_tree_to_json(tree, filename):
    """
    Serialize and save the tree to a JSON file.

    Parameters:
    tree : IsolationTree
        The tree to serialize and save.
    filename : str
        The filename where the tree will be saved.

    This function serializes the tree using a custom serialization function and saves it
    in JSON format to the specified file. If the tree is None, the function does not
    perform serialization or save the file.
    """
    if tree is None:
        print("No tree to save.")
        return

    # Serialize the tree to a dictionary format
    tree_dict = serialize_tree(tree)

    # Try to save the serialized tree to a JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(tree_dict, f, indent=4)
        print(f"Saved new tree to '{filename}'.")
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

    
    
    
###Pickle Method to save the newly formed trees#####################
import pickle

def save_new_trees(ensemble, base_filename):
    """
    Save newly updated trees from an ensemble to individual files using pickle.

    Parameters:
    ensemble : object
        An ensemble object that contains trees and a set of indices for new or updated trees.
    base_filename : str
        The base path and filename prefix for saving the trees, where each tree's index is appended to create unique filenames.

    This function iterates over the indices of newly updated trees in the ensemble and serializes each one to a
    separate file. After saving, it clears the indices to restore_defaults the state.
    """
    for idx in ensemble.new_tree_indices:
        filename = f"{base_filename}_tree_{idx}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(ensemble.trees[idx], file)
        print(f"Saved tree {idx} to '{filename}'.")
    # Clear the indices after saving
    ensemble.new_tree_indices.clear()


def load_tree(filename):
    """
    Load a tree from a file serialized with pickle.

    Parameters:
    filename : str
        The filename from which to load the tree.

    Returns:
    object
        The tree object loaded from the file.
    """
    with open(filename, 'rb') as file:
        tree = pickle.load(file)
    return tree
