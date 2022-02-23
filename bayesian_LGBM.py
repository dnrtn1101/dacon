import numpy as np
import pandas as pd
import datetime
import pickle

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
#########################################################################
dim_n_estimators = Integer(low = 2000, high = 40000, name = 'n_estimators')
dim_num_leaves = Real(low=0.1, high = 1, name = 'num_leaves')
dim_max_depth = Integer(low = 1, high = 9, name = 'max_depth')
dim_min_child_weight = Real(low=1e-5, high = 1, name = 'min_child_weight')
dim_min_child_samples = Integer(low = 20, high = 1000, name = 'min_child_samples')
dim_subsample = Real(low = 0.01, high = 1, name = 'subsample')
dim_colsample_bytree = Real(low = 0.01, high = 1, name = 'colsample_bytree')

dimensions=[dim_n_estimators,
            dim_num_leaves,
            dim_max_depth,
            dim_min_child_weight,
            dim_min_child_samples,
            dim_subsample,
            dim_colsample_bytree]

default_parameters = [28997, 0.5528210952049166, 7, 0.850359167657365, 
                      169, 0.1463978283110154, 0.9978180434130473]
#########################################################################
x_train = pd.read_csv('x_train.csv')
x_val = pd.read_csv('x_val.csv')
features_list = ['key', 'date', 'exclusive_use_area', 'common_area',
'total_household_count_of_area_type', 'floor/tallest', 'floor/lowest',
'supply_area', 'age', 'min_school_dist', 'min_subway_dist',
'transaction_month', 'total_household_count_in_sites', 'floor',
'total_parking_capacity_in_site', 'apartment_id', 'foundation_year',
'room_id', 'min_subway', 'cluster_N', 'year_of_completion',
'address_by_law_y', 'lowest_building_in_sites', 'longitude', 'latitude',
'apartment_building_count_in_sites', 'tallest_building_in_sites', 'sch_cnt',
'address_by_law_3', 'transaction_date_label_encoding', 'latitude_x',
'address_by_law_2', 'latitude_y', 'longitude_y', 'longitude_x',
'address_by_law', 'address_by_law_x', 'sub_cnt', 'lat_room_id',
'long_room_id', 'class_operation', 'room_count',
'front_door_structure_label_encoding', 'heat_type_label_encoding',
'bathroom_count', 'gender', 'heat_fuel_label_encoding', 'highschool_type',
'operation_type', '7', '3', '1', '5', 'B1', 'DL', 'B2', '4', 'station_id',
'transaction_year_month', 'lat_school_code']

x_train = x_train[x_train['transaction_year'] > 2012]
y_train = x_train['log_target']
x_train = x_train[features_list]
features = x_train.columns.values
x_train = x_train.values

y_val = x_val['log_target']
x_val = x_val[features_list].values

print("train shape : ", x_train.shape)
print("val shape : ", x_val.shape)
#########################################################################
path_best_model = 'best_model.pkl'
best_accuracy  = 99999999999999999.0
lst = []
cnt = 0
#########################################################################
@use_named_args(dimensions=dimensions)
def fitness(n_estimators, num_leaves, max_depth, min_child_weight,
            min_child_samples, subsample, colsample_bytree):
    
    global cnt
    cnt = cnt + 1
    print("{}회차".format(cnt))
    print("n_estimators : ", n_estimators)
    print("num_leaves : ", num_leaves)
    print("max_depth : ", max_depth)
    print("min_child_samples : ", min_child_samples)
    print("min_child_weight : ", min_child_weight)
    print("subsample : ", subsample)
    print("colsample_bytree : ", colsample_bytree)
    print(max(2, int(2**(num_leaves * max_depth) - 1)))
    print(2**max_depth)
    model =LGBMRegressor(n_estimators = n_estimators, 
                         num_leaves = max(2, int(2**(num_leaves * max_depth) - 1)), 
                         max_depth = max_depth,
                         min_child_samples = min_child_samples,
                         min_child_weight = min_child_weight,
                         subsample = subsample,
                         colsample_bytree = colsample_bytree)
    
    now = datetime.datetime.now()
    model.fit(x_train,y_train)
    print(datetime.datetime.now() - now)
    
    y_pred = model.predict(x_val)
    mse_val = mean_squared_error(np.expm1(y_val), np.expm1(y_pred))
    rmse_val = np.sqrt(mse_val)
    
#    train_pred = model.predict(x_train)
    #mse_train = mean_squared_error(np.expm1(y_train), np.expm1(train_pred))
    #rmse_train = np.sqrt(mse_train)
    #train.append(rmse_train)

    
    print()
    print("Test : ", rmse_val)
    print()
    
    global lst
    arr = [n_estimators, num_leaves, max_depth, min_child_samples, 
           min_child_weight, subsample, colsample_bytree, rmse_val]
    lst.append(arr)

    del model
    return rmse_val

search_resultsearch_r  = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=100,
                            x0=default_parameters)

DATA = pd.DataFrame(lst)

