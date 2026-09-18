"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.

    if len(labels)==0:
        return 0.0
    classes,counts=np.unique(labels,return_counts=True)
    if len(classes)==1:
        return 0.0
    prop=counts/len(labels)
    prop=prop[prop>0]
    entropy=-np.sum(prop*np.log(prop))
    return float(entropy)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col=features[:, feature_index]
    mask=col<=threshold
    left_features=features[mask]
    left_labels=labels[mask]
    right_features=features[~mask]
    right_labels=labels[~mask]
    return left_features,left_labels,right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    parent_impurity=impurity(parent_labels)
    left_impurity=impurity(left_labels)
    right_impurity=impurity(right_labels)

    n=len(parent_labels)
    wl=len(left_labels)/n
    wr=len(right_labels)/n

    return parent_impurity-(wl*left_impurity+wr*right_impurity)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    result={'feature_index':None, 'threshold':None,'score':0.0}
    for fi in feature_indices:
        value=np.unique(features[:,fi])
        if len(value)<2:
            continue
        
        thresholds=(value[:-1]+value[1:])/2
        for t in thresholds:
            lf,ll,rf,rl=split_dataset(
                features,labels, fi,t
            )

            if len(ll)==0 or len(rl)==0:
                continue
            
            s=split_score(labels, ll,rl)
            if s>result['score']:
                result={
                    'feature_index':fi, 'threshold':t,'score':s
                }
    return result

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

