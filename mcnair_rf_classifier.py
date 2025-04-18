



def bootstrap_creator(this_tree_data, label_column_name, total_samples):
  bootstrap_for_this_tree = [] # holder array

  for i in range(total_samples): # match size of bootstrap to size of training data set
    random_index_to_add = np.random.randint(0 , total_samples) # generate a row index to grab from the training datatset
    bootstrap_for_this_tree.append(this_tree_data.iloc[random_index_to_add]) # add that dataset row to our array


    #create the bootstrap dataframe
  bootstrap = pd.DataFrame(bootstrap_for_this_tree) # I suppose I could have np.unique before creating the dataframe here - nevermind that didn't work how I intended

    # generate X and Y bootstraps from bootstrap
  X_bootstrap = bootstrap.drop(label_column_name, axis=1)
  y_bootstrap = bootstrap[label_column_name]

  return X_bootstrap, y_bootstrap, bootstrap



def oob_determination(tree_data, bootstrap, label_column_name):

    # drop bootstrap rows from the total tree training data set to be left with a dataframe of only the none-selected training samples
  oob_for_this_tree = tree_data.drop(index=bootstrap.index) # drop all the indices from the tree dataset/ training dataset that exist within the bootstrap for this specific tree

    # copy oob_for_this_tree incase we need to pass it as a return later
  all_oob_this_tree = oob_for_this_tree.copy()

    # make the X-oob and y_oob respective dataframes
  X_oob = oob_for_this_tree.drop(label_column_name, axis=1)
  y_oob = oob_for_this_tree[label_column_name]

  return X_oob, y_oob, all_oob_this_tree



def boot_strap_and_oob(tree_data, label_column_name, total_samples):
  X_bootstrap, y_bootstrap, bootstrap = bootstrap_creator(tree_data, label_column_name, total_samples)
  X_oob, y_oob, all_oob = oob_determination(tree_data, bootstrap, label_column_name)

  return X_bootstrap, y_bootstrap, X_oob, y_oob, bootstrap, all_oob

def split_point_finder(x):
  mid_points_in_column = []
  x_sorted = sorted(x)

  for i in range(0, len(x)-1):
    mid_point = ( (x_sorted[i] + x_sorted[i+1]) / 2)
    mid_points_in_column.append(mid_point)
  return mid_points_in_column

    

def branching_splits(x, y, split_point):
  mask = x >= split_point
  anti_mask = x < split_point
  x_right = x[mask]
  x_left = x[anti_mask]
  y_right = y[mask]
  y_left = y[anti_mask]
  return x_right, x_left, y_right, y_left


def best_branching_random_features(data, label_column_name, num_random_features_to_choose):

  y = data[label_column_name]
  possible_features_to_test = data.drop(label_column_name, axis=1) # drop the label column from the dataframe so only features are left

  randomized_features_this_node = rng.choice(possible_features_to_test.columns, num_random_features_to_choose, replace = False)
  # we use rng.choice to randomly choose (parameter number of features) from (all feature columns that are present in the dataframe) and don't allow duplicates

  this_node_data = possible_features_to_test[randomized_features_this_node].copy() # then we create a new dataframe only copying the feature columns selected by our random feature selector above
  # we get a smaller dataframe with only the number of features declared in the function parameter

  feature_error_dictionary = {}

  # In assignment 1 we wouldn't have ran into these errors, but I ran into an error for each of these variables not being initialized
  # min error should be set to inf, anythign smaller will replace it
  minimum_error = float('inf')

  # these 3 can create errors further down our decision tree building if no split points are found to reduce the minimum error, if they arent initialized now and we go to return them
  # then we get the errors I pasted in below - if your testing a split with only has 100 samples and for a feature they are all the same value, then no split point exists and these
  # values would be left un-initialized given our structure below
  node_mse = float('inf')
  best_feature = 'Kelton'
  best_split_point_current_feature = 992018683
  #UnboundLocalError: cannot access local variable 'best_feature' where it is not associated with a value
  #UnboundLocalError: cannot access local variable 'node_mse' where it is not associated with a value
  #UnboundLocalError: cannot access local variable 'best_split_point_current_feature' where it is not associated with a value

  for feature in this_node_data.columns:
    x = data[feature]

    split_points = split_point_finder(x)
    split_points_unique = np.unique(split_points)

    for split_point in split_points_unique:
      x_right, x_left, y_right, y_left = branching_splits(x, y, split_point)
      unique_errors = weighted_branch_MSE(y,y_right,y_left)
      feature_error_dictionary[feature, split_point] = [unique_errors]

        # recording the best feature and split point based on lowest weighted branch mse from the mse function
      if unique_errors[1] < minimum_error:
        minimum_error = unique_errors[1]
        node_mse = unique_errors[0]
        best_feature = feature
        best_split_point_current_feature = split_point

  return feature_error_dictionary, node_mse, minimum_error, best_feature, best_split_point_current_feature



def dt_builder(bootstrap_current_node, label_column_name, num_randomized_features, minimum_samples_per_leaf, min_samples_per_split, max_depth, depth = 0):

  before_split_sample_size = bootstrap_current_node.shape[0]# get the number of samples in the current parent node


  # this is like the max depth checker in the in-class example, but we must also check to make sure that there are enough samples in the parent node to even determine if splitting is worthwhile
  if depth >= max_depth or before_split_sample_size < min_samples_per_split: # if depth is >= max depth or there arent enough samples in the parent node then we return a prediction
    return {"prediction_this_node": np.mean(bootstrap_current_node[label_column_name])}

  # we have not reached max depth or the minimum number of samples in the parent node, so we call our function to find the best splitting at this node
  # we pass thorugh the current node sample, the label, and the number of features to randomly choose
  dictionary_of_errors_for_node, root_mse, branch_mse, feature, split_point = best_branching_random_features(bootstrap_current_node, label_column_name, num_randomized_features)


  # this is where I began getting errors for the uninitialized best feature, split and root mse in our branching function
  # I have initialized the best feature to my name, if this is returned that means there were no valid split points found in the features randomly selected
  # this seems to be due to a small number of samples in the node and the sample feature values being the same, like all having 1.0 floors as an example, no splits found so kelton is returned
  # in the case that this non-feature basecase is returned we can no longer split and return this node as a prediction node
  if feature == 'Kelton':
    # print(feature) # testing
    return {"prediction_this_node": np.mean(bootstrap_current_node[label_column_name])}

  # just like branching splits we use vectorized operations on the current node to get the data for the resulting left and right leaf nodes from the split
  # I suppose you could store these in the error dictionary from the best_branching_random_features from above
  left_node_data = bootstrap_current_node[bootstrap_current_node[feature] < split_point]

  right_node_data = bootstrap_current_node[bootstrap_current_node[feature] >= split_point]

  # this gets the size of the resulting split leaf nodes, left and right, from the lead node data sets, we get the sizes to check the leaf node sample sizes against the minimum
  left_sample_size = left_node_data.shape[0]
  right_sample_size = right_node_data.shape[0]

  # here we are checking if both the left and right resulting leaf node samples from splitting are greater than or equal to the minimum value parameter we have set
  # if either is less than the parameter threshold we do not do the node split and we return the node as a prediction node instead
  if left_sample_size < minimum_samples_per_leaf or right_sample_size < minimum_samples_per_leaf:
    return {"prediction_this_node": np.mean(bootstrap_current_node[label_column_name])}

 # these are our recrusive calls to this function, for the left nodes and right nodes, each time we increase the depth by 1 so we know if we need to stop the branching due to depth
 # these calls are what create the tree leaf structure below the root node which is created the first time we call this function
  left_leaf_node = dt_builder(left_node_data, label_column_name, num_randomized_features, minimum_samples_per_leaf, min_samples_per_split, max_depth, depth + 1)
  right_leaf_node = dt_builder(right_node_data, label_column_name, num_randomized_features, minimum_samples_per_leaf, min_samples_per_split, max_depth, depth + 1)

  # just like our tuple return from the in class version of this code, we return the feature, the split point, the left leaf node and right leaf node, in dictionary form instead of tuples
  return { "feature": feature, "split_point": split_point, "left_leaf_node": left_leaf_node, "right_leaf_node": right_leaf_node}
  # I had dynamic depth tracking in the key of the leaf nodes, but that was making tree navigation more difficult so I opted to remove it









# so with the functionality of our decision tree builder and prediction recursive functions established, the random forest regressor and the aggregate prediciton functions are not too difficult
# we just need to create loops to build enough trees to become a forest of X size, and then traverse all of them to make predictions with the forest structure
# here we need to pass in all of the parameters that we need, the training data, label, number of tree to build, the num of random features and the bootstramp sample size which should match out training sample size

def imported_random_forest_classifier(data, label_column_name, number_of_trees, num_randomized_features, bootstrap_sample_size):

  forest_of_trees_dict = {}

  # these 3 variables are the ones that we change to try and tune our trees which resulting in tuning the forest
  # I have been fiddling with these to get lower MSE's in our testing results
  # 8500 split values rounded up are 4250, 2125, 1063, 532, 266, 133, 67, 34, 17, 9, 5, 3, 2
  # at 13 splits we are getting leafs with very few samples
  # I was getting some of my best results at min samples 9, max depth 10 if I recall
  minimum_samples_per_leaf = 3
  min_samples_per_split = (minimum_samples_per_leaf * 2) + 1 # this is me rounding up when determining how many samples in the current node are needed to even try to split before testing leaf # samples
  max_depth = 7

  # storing each oob indices array as value #2 [value[1]] in the value tuple of the forest dictionary
  each_entry_oob =[]

  #build all the trees paramaterized
  for i in range(number_of_trees):
      
    # each tree gets its own bootstrap, everytime we call dt_builder we need a new bootstrap
    #dt_builder builds trees with randomized features at each node of the tree
    # then we store all of the OOB df indices to a list, you could store the whole dataframe, but when I access them later I only need the index values, so I am going to take only those now
    # you can use list(df indices) or the method that I used as .tolist() - I do remember seeing one comment on a website stating something about how list(df indices) might be safer or more reliable for some data structures? something to do with numpy I believe
    # we then store each tree in a  forest dictionary entry, the key is the tree number and the value is a tuple with value[0] = dictionary that is the tree itself, and value[1] is the array of indices of the oob values for the bootstrap sample used in building the tree
    X_bootstrap, y_bootstrap, X_oob, y_oob, bootstrap_current_node, all_oob = boot_strap_and_oob(data, label_column_name, bootstrap_sample_size)
    individual_tree_dict = dt_builder(bootstrap_current_node, label_column_name, num_randomized_features, minimum_samples_per_leaf, min_samples_per_split, max_depth, depth = 0)
    each_entry_oob_indices = all_oob.index.values.tolist()
    forest_of_trees_dict[i] = individual_tree_dict , each_entry_oob_indices

# then we return the forest dictionary
  return forest_of_trees_dict


















        

