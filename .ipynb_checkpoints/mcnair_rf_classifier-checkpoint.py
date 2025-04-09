# so with the functionality of our decision tree builder and prediction recursive functions established, the random forest regressor and the aggregate prediciton functions are not too difficult
# we just need to create loops to build enough trees to become a forest of X size, and then traverse all of them to make predictions with the forest structure
# here we need to pass in all of the parameters that we need, the training data, label, number of tree to build, the num of random features and the bootstramp sample size which should match out training sample size

def random_forest_regressor_from_scratch(data, label_column_name, number_of_trees, num_randomized_features, bootstrap_sample_size):

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