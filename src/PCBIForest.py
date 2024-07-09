#print("scikit-multiflow package installation")
#!pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow
#!pip install -U git+https://github.com/Elmecio/IForestASD_based_methods_in_scikit_Multiflow.git. 
#print("scikit-multiflow package installation")

#--------------------------Class NDKSWIN for the drift Detection-----------------------------------#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####Class to define the NDKSWIN Method for the drift Detection######

import numpy as np
from scipy import stats
import random
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import pandas as pd
from collections import deque


class NDKSWIN(BaseDriftDetector):
    r""" Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha: float (default=0.005)
        Probability for the test statistic of the Kolmogorov-Smirnov-Test
        The alpha parameter is very sensitive, therefore should be set
        below 0.01.

    window_size: float (default=100)
        Size of the sliding window

    stat_size: float (default=30)
        Size of the statistic window

    ---data: numpy.ndarray of shape (n_samples, 1) (default=None,optional)
        Already collected data to avoid cold start.---
    data: numpy.ndarray of shape (n_samples, n_attributes) (default=None,optional)
        Already collected data to avoid cold start.

    n_dimensions = the number of random dimensions to consider when computing
        stats and detecting the drift

    Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) [1]_ is a concept change detection method based
    on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with
    no assumption of underlying data distribution. KSWIN can monitor data or performance
    distributions. ----Note that the detector accepts one dimensional input as array.---
    Note that this version is free of dimensional number input as array

    KSWIN maintains a sliding window :math:`\Psi` of fixed size :math:`n` (window_size). The
    last :math:`r` (stat_size) samples of :math:`\Psi` are assumed to represent the last
    concept considered as :math:`R`. From the first :math:`n-r` samples of :math:`\Psi`,
    :math:`r` samples are uniformly drawn, representing an approximated last concept :math:`W`.

    The KS-test is performed on the windows :math:`R` and :math:`W` of the same size. KS
    -test compares the distance of the empirical cumulative data distribution :math:`dist(R,W)`.

    A concept drift is detected by KSWIN if:

    * :math:`dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}`

    -> The difference in empirical data distributions between the windows :math:`R` and :math:`W`
    is too large as that R and W come from the same distribution.

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
    Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> from skmultiflow.drift_detection import KSWIN
    >>> import numpy as np
    >>> # Initialize KSWIN and a data stream
    >>> kswin = KSWIN(alpha=0.01)
    >>> stream = SEAGenerator(classification_function = 2,
    >>>     random_state = 112, balance_classes = False,noise_percentage = 0.28)
    >>> # Store detections
    >>> detections = []
    >>> # Process stream via KSWIN and print detections
    >>> for i in range(1000):
    >>>         data = stream.next_sample(10)
    >>>         batch = data[0][0][0]
    >>>         kswin.add_element(batch)
    >>>         if kswin.detected_change():
    >>>             print("\rIteration {}".format(i))
    >>>             print("\r KSWINReject Null Hyptheses")
    >>>             detections.append(i)
    >>> print("Number of detections: "+str(len(detections)))
    """

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

        #if not isinstance(data, np.ndarray) or data is None:
            #self.window = np.array([])
        #else:
        #    self.window = data
        self.window = data

        if self.n_dimensions <= 0 or (data is not None and self.n_dimensions > data.shape[1]):
            print("Warning: n_dimensions must be between 1 and <= input_value.shape[1]. We will consider all dimensions to compute the drift detection.")
            #raise ValueError("n_dimensions must be between 1 and <= data.shape[1]")
            self.n_dimensions = data.shape[1]
        #else:
        #    self.n_dimensions = n_dimensions

        if self.n_tested_samples <= 0.0 or self.n_tested_samples > 1.0 :
            raise ValueError("n_tested_samples must be between > 0 and <= 1")
        else:
            self.n_samples_to_test = int(self.window_size*self.n_tested_samples)

    def add_element(self, input_value):

        """ Add element to sliding window

        Adds an element on top of the sliding window and removes
        the oldest one from the window. Afterwards, the KS-test
        is performed.

        Parameters
        ----------
        input_value: ndarray
            New data sample the sliding window should add.
        """

        #print("input_value = ")
        #print(input_value)
        self.change_detected = False

        if self.fixed_checked_dimension:
            sample_dimensions=list(range(self.n_dimensions))
        else:
            if self.n_dimensions > input_value.shape[1]:
                print("n_dimensions must be between 1 and <= input_value.shape[1]. We will consider the first dimension only to compute the drift detection.")
                sample_dimensions = [0]
            else:
                sample_dimensions = random.sample(list(range(input_value.shape[1])), self.n_dimensions)

        #print("sample_dimensions")
        #print(sample_dimensions)

        if self.fixed_checked_sample:
            sample_test_data = input_value[list(range(self.n_samples_to_test))]
        else:
            if self.n_samples_to_test > input_value.shape[0]:
                #print("self.n_samples_to_test = "+str(self.n_samples_to_test))
                #print("input_value.shape[0] = "+str(input_value.shape[0]))
                print("Not enough data in input_value to pick "+str(self.n_samples_to_test)+" We will use 100% of input_value.")
                sample_test_data = input_value
            else:
                sample_test_data = input_value[random.sample(list(range(input_value.shape[0])), self.n_samples_to_test)]

        #print("sample_test_data")
        #print(sample_test_data)
        for value in sample_test_data:
            #print("self.change_detected = "+str(self.change_detected))
            if self.change_detected == False:
                self.n += 1
                currentLength = self.window.shape[0]
                if currentLength >= self.window_size:
                    #print(type(self.window))
                    #print(type(input_value))
                    #print("self.window 1 = ")
                    #print(self.window)
                    #self.window = np.delete(self.window, 0)
                    self.window = np.delete(self.window, 0,0)
                    #print("self.window = np.delete(self.window, 0) = ")
                    #print(self.window)

                    for i in sample_dimensions:
                        #rnd_window = np.random.choice(self.window[:,i][:-self.stat_size], self.stat_size)
                        rnd_window = np.random.choice(np.array(pd.DataFrame(self.window)[i])[:-self.stat_size], self.stat_size)

                        #print("rnd_window = ")
                        #print(rnd_window)
                        #print("np.array(pd.DataFrame(self.window)[i])[:-self.stat_size] = ")
                        #print(np.array(pd.DataFrame(self.window)[i])[:-self.stat_size])
                        #print("np.array(pd.DataFrame(self.window)[i]) = ")
                        #print(np.array(pd.DataFrame(self.window)[i]))
                        #print("np.array(pd.DataFrame(self.window)[i])[-self.stat_size:] = ")
                        #print(np.array(pd.DataFrame(self.window)[i])[-self.stat_size:])

                        #(st, self.p_value) = stats.ks_2samp(rnd_window,
                        #                                   self.window[:,i][-self.stat_size:], mode="exact")
                        (st, self.p_value) = stats.ks_2samp(rnd_window,
                                                            np.array(pd.DataFrame(self.window)[i])[-self.stat_size:], mode="exact")
                        #print("self.p_value = ")
                        #print(self.p_value)
                        #print("st = ")
                        #print(st)

                        if self.p_value <= self.alpha and st > 0.1:
                            self.change_detected = True
                            self.window = self.window[-self.stat_size:]
                            #print("Change_detected in dimension "+str(i)+" on data "+str(value))
                            break
                        else:
                            self.change_detected = False
                            #print("self.change_detected = False")
                else:  # Not enough samples in sliding window for a valid test
                    #raise ValueError("Not enough samples in sliding window for a valid test")
                    #print("Not enough samples in sliding window for a valid test")
                    self.change_detected = False

                self.window = np.concatenate([self.window, [value]])
                #print(self.window)
            else:
                #print("break execution")
                break

    def detected_change(self):
        """ Get detected change

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        return self.change_detected

    def reset(self):
        """ reset

        Resets the change detector parameters.
        """
        self.p_value = 0
        self.window = np.array([])
        self.change_detected = False
        
        
#---------------------------------Class to build Isolation Tree-------------------------------------------#



class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None ###Split Attribute####
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1
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
        min_x = X[:, split_by].min()
        max_x = X[:, split_by].max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self
        condition = True

        while condition:

            split_value = min_x + random.betavariate(0.5,0.5)*(max_x-min_x)

            a = X[X[:, split_by] < split_value]
            b = X[X[:, split_by] >= split_value]
            if len(X) < 10 or a.shape[0] < 0.25 * b.shape[0] or b.shape[0] < 0.25 * a.shape[0] or (
                    a.shape[0] > 0 and b.shape[0] > 0):
                condition = False

            self.size = len(X)
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit_improved(a, improved=False)
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit_improved(b, improved=False)
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
        min_x = X_col.min()
        max_x = X_col.max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[w], improved=True)
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[~w], improved=True)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self
    
    #### Helper Function ######
def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_tree(x, t,e):
    print(f"Inside path_length_tree, x shape: {x.shape}, attempting to access index: {t.split_by}")
    e = e
    if t.exnodes == 1:
        e = e+ c(t.size)
        return e
    else:
        a = t.split_by
    if x[a] < t.split_value :
        return path_length_tree(x, t.left, e+1)

    if x[a] >= t.split_value :
        return path_length_tree(x, t.right, e+1)


    
#-----------------------Class to build the isolation forest with integration of Performance Counter Based Method--------------------#

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10, window_size=100, threshold=0.5, alpha=0.005):

        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []
        self.window_size= deque(maxlen=window_size)  # Sliding window for drift detection
        self.threshold = threshold
        self.alpha = alpha
        self.drift_detector = NDKSWIN(alpha=alpha, window_size=window_size)  # Drift detector


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values


        len_x = len(X)
        col_x = X.shape[1]
        self.trees = []

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

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))
        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions
    #------------------Implementation of PCB-Iforest starts from here onwards --------------------------------#
    #------------------Function to replace the tree in self.tree within IsolationTreensemble------------------#
    def replace_trees(self):
        n_replaced_trees = 0  # Initialize a counter for replaced trees
        for i, tree in enumerate(self.trees):
            if tree.pc_value < 0:  # Assuming each tree has a performance counter `pc_value`
            # Rebuild the tree with current window of data
                self.trees[i] = self.build_tree(np.array(self.window_size))
                tree.pc_value = 0  # Reset the performance counter
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
        
