
# coding: utf-8

# In[1]:

import os
import pandas as pd

import sklearn.ensemble as ske

def categorise_dataframe(df):
    # Categories work better as indexes
    for obj_key in df.keys()[df.dtypes == object]:
        try:
            df[obj_key] = df[obj_key].astype('category')
        except TypeError:
            print("Couldn't categorise {}".format(obj_key))
            pass
    return df

def feature_extractor(df, target):
    data = df.drop(target, axis=1)
    reg = ske.RandomForestRegressor(n_jobs=4, n_estimators=512)
    reg.fit(data, df[target])
    return pd.Series(dict(zip(data.keys(), reg.feature_importances_)))


# The questions we want to answer are:
# 1. What metrics differentiate between what behaviours
# 2. Do these metrics cross over domains (i.e. comms impacting behaviour etc)
#
# To answer these questions we first have to manipulate the raw dataframe to be weight-indexed with behaviour(`var`) keys on the perspective from the observer to the (potential) attacker in a particular run (summed across the timespace)
#
# While this analysis simply sums both the upper and lower outliers, **this needs extended/readdressed**
#
# # IMPORTANT
# The July 3rd Simulation Run had a small mistake where the Ran

# In[2]:

observer = 'Bravo'
target = 'Alfa'
n_nodes = 6
n_metrics = 9

results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-03-16-45-26"
results_path = "/home/bolster/src/aietes/results/Malicious Behaviour Trust Comparison-2015-07-20-17-47-53"



with pd.get_store(os.path.join(results_path, "outliers.bkup.h5")) as store:
    weight_df = store.get('CombinedTrust1_3')
    # Select Perspective here
    weight_df = weight_df[weight_df.observer == observer]

weight_df = categorise_dataframe(weight_df)


# In[3]:

metric_keys = list(weight_df.keys()[-n_metrics:])
weight_df.set_index(metric_keys+['var','t'], inplace=True)
# REMEMBER TO CHECK THIS WHEN DOING MULTIPLE RUNS (although technically it shouldn't matter....)
weight_df.drop(['observer','run'], axis=1, inplace=True)
# Sum for each run
time_summed_weights = weight_df.groupby(level=list(weight_df.index.names[:-1])).sum().unstack('var')
target_weights = time_summed_weights.xs(target,level='target', axis=1).fillna(0.0) # Nans map to no outliers



# These results handily confirm that there is a 'signature' of badmouthing as RandomFlatWalk was incorrectly configured.
#
# Need to:
# 1. Perform multi-run tolerance analysis of metrics (i.e. turn the below into a boxplot)
# 2. Perform cross correlation analysis on metrics across runs/behaviours (what metrics are redundant)

# In[7]:

import operator

def target_weight_feature_extractor(target_weights):
    known_good_features_d = {}
    for basekey in target_weights.keys(): # Parallelisable
        print basekey
        # Single DataFrame of all features against one behaviour
        var_weights = target_weights.apply(lambda s: s/target_weights[basekey], axis=0).dropna()
        known_good_features_d[basekey] =             pd.concat([feature_extractor(s.reset_index(),var) for var,s  in var_weights.iteritems()],
                      keys=var_weights.keys(), names=['var','metric'])

    return known_good_features_d


def dataframe_weight_filter(df, keys):
    indexes = [(df.index.get_level_values(k)==0.0) for k in keys]
    return df.loc[reduce(operator.and_,indexes)]

phys_keys = ['INDD','INHD','Speed']
comm_keys = ['ADelay','ARXP','ATXP','RXThroughput','TXThroughput','PLR']


# In[8]:

#Comms Only Weights
comms_target_weights=dataframe_weight_filter(target_weights,phys_keys)
comms_target_weights.reset_index(level=phys_keys, drop=True, inplace=True)
comms_features_d = target_weight_feature_extractor(comms_target_weights)
for var,feat in comms_features_d.iteritems():
    feat.unstack().plot(kind='bar', title=var)


# In[9]:

#Phys Only Weights
phys_target_weights=dataframe_weight_filter(target_weights,comm_keys)
phys_target_weights.reset_index(level=comm_keys, drop=True, inplace=True)
phys_features_d = target_weight_feature_extractor(phys_target_weights)
for var,feat in phys_features_d.iteritems():
    feat.unstack().plot(kind='bar', title=var)
