""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""

import numpy as np


class DTLearner(object):
    def __init__(self, leaf_size = 1, verbose=False):
        """
        Constructor
        
        Parameters
        ----------
        leaf_size : int, optional
            size of the leaf node. The default is 1.
        verbose : bool, optional
            If true, print out debug messages. The default is False.

        Returns
        -------
        None.

        """

        self.leaf_size = leaf_size
        self.verbose = verbose


    def author(self):
        """
        Auther string

        Returns
        -------
        string
            The GT username of the student.

        """

        return "lsoh3"  # Georgia Tech username


    def find_best_split(self, data):
        """
        Get the next feature to split and the split value.

        Parameters
        ----------
        data : numpy.ndarray
            numpy array representing training data and labels.
        Returns
        -------
        feature_index : int
            index of the feature to split.
        split_val : scalar value
            value used to split the nodes.

        """
        # List that contains the correlation values
        corr_list = []
        for i in range(data.shape[1] - 1):
            # Case when feature values are the same for that entire column.
            if np.unique(data[:, i]).size == 1:
                # For repeated occurrences of feature values, np.corrcoef() returns NaN. We'll treat
                # the result of np.corrcoef() as 0 in this case. Being the smallest, it'd not be selected
                # after the sorting.
                corr_list.append(0)
            else:
                # Correlation values are in a 2x2 matrix. Need to get one of the
                # diagonal values. That is why we need to request [0,1] - one of
                # these diagonal values.
                corr_list.append(abs(np.corrcoef(data[:, i], data[:, -1])[0,1]))
        corr_sorted_array_ind = np.argsort(corr_list)
        feature_index = corr_sorted_array_ind[-1]
        # When there are repeated occurences of values in the dataset, using np.median() causes the
        # 'split' to be on one side of the tree, and we'd hit "maximum recursion depth exceeded" error.
        # The use of np.mean(), on the other hand, gives a more balanced decision tree.
        split_val = np.mean(data[:, feature_index])

        return feature_index, split_val


    def build_tree(self, data):
        """
        Build the decision tree data structure.

        Parameters
        ----------
        data : numpy.ndarray
            numpy array representing traing data and labels.

        Returns
        -------
        numpy.ndarray
            Decision tree data structure.

        """
        # The cases when there is no need to split and you'd simply return the np array:
        # 1) There is only one row of data in the dataset
        # 2) There are repeated occurrences of the same label values (i.e. np.unique().size == 1)
        # Note that the result returned is a list of a list.
        if data.shape[0] <= self.leaf_size or np.unique(data[:,-1]).size == 1:
            return np.array([[-1, data[:, -1].mean(), None, None]])
         
        # Determine the best feature to split on
        feature_index, split_val = self.find_best_split(data)

        left_data = data[data[:, feature_index] <= split_val]
        right_data = data[data[:, feature_index] > split_val]

        # Build the left tree
        left_tree = self.build_tree(left_data)

        # Build the right tree
        right_tree = self.build_tree(right_data)

        # Must be declared as a list of a list
        root = [[feature_index, split_val, 1, left_tree.shape[0] + 1]]

        # The concatenation of the root, left tree and right tree is the result we want
        return np.concatenate((root, left_tree, right_tree), axis=0)


    def add_evidence(self, data_x, data_y):
        """
        Train and build a decision tree.

        Parameters
        ----------
        data_x : numpy.ndarray
            training data.
        data_y : numpy.ndarray
            labels.

        Returns
        -------
        None.

        """

        # Slap on 1s column so linear regression finds a constant term
        data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        data[:, 0:data_x.shape[1]] = data_x
        data[:,-1] = data_y

        self.dt = self.build_tree(data)


    def predict(self, point, dt, root_index):
        """
        Predict result for a single query.

        Parameters
        ----------
        point : numpy.ndarray
            features of a single query corresponding to one row.
        dt : numpy.ndarray
          decision tree
        root_index : int
            decision tree root node index.

        Returns
        -------
        scalar value
            The predicted result for the query.

        """
        
        feature_index, val, left_tree, right_tree = dt[root_index, :]
        feature_index = int(feature_index)
        if feature_index == -1: # leaf node
            return val
        left_tree = int(left_tree)
        right_tree = int(right_tree)
        if point[feature_index] <= val:
            return self.predict(point, dt, root_index + left_tree)
        else:
            return self.predict(point, dt, root_index + right_tree)

        
    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        Parameters
        ----------
        points : numpy.ndarray
            A numpy array with each row corresponding to a specific query. There are multiple rows in
            this numpy array.

        Returns
        -------
        result : numpy.ndarray
            The predicted result of the input data according to the trained model.

        """
        
        result = np.zeros([points.shape[0]])
        for i in range(points.shape[0]):
            result[i] = self.predict(points[i,:], self.dt, 0)
        return result



























