import numpy as np
import pandas as pd

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv('x_train.csv')
val = pd.read_csv('x_val.csv')
features_list = ['exclusive_use_area', 'date', 'key', 'supply_area', 'common_area',
 'total_household_count_of_area_type', 'age',
 'total_household_count_in_sites', 'min_subway_dist', 'min_school_dist',
 'total_parking_capacity_in_site', 'apartment_id', 'floor/lowest',
 'transaction_year_month', 'floor/tallest', 'room_id', 'year_of_completion',
 'foundation_year', 'floor', 'lowest_building_in_sites' ,'address_by_law_y',
 'cluster_N', 'apartment_building_count_in_sites', 'min_subway',
 'tallest_building_in_sites' ,'longitude', 'latitude' ,'transaction_month',
 'long_room_id', 'lat_room_id' ,'sch_cnt', 'address_by_law_3',
 'address_by_law' ,'latitude_x' ,'longitude_x' ,'station_id', 'longitude_y',
 'address_by_law_x' ,'latitude_y', 'address_by_law_2' ,'lat_apartment_id',
 'lat_school_code' ,'long_apartment_id', 'long_school_code',
 'transaction_year', 'lat_station_id' ,'long_station_id' ,'room_count',
 'sub_cnt', 'transaction_date_label_encoding' ,'class_operation',
 'bathroom_count', 'front_door_structure_label_encoding',
 'heat_type_label_encoding' ,'heat_fuel_label_encoding', 'school_class',
 'gender' ,'highschool_type' ,'operation_type' ,'3' ,'5', 'DL', '1', 'B1', '7',
 'B2', '4']

train = train[train['transaction_year'] > 2012]
y_train = train['log_target']
x_train = train[features_list]
features = x_train.columns.values
x_train = x_train.values

y_val = val['log_target']
x_val = val[features_list].values

n_estimators =  31768
num_leaves = 0.985829854822637 
max_depth =  5
min_child_samples =  696
min_child_weight =  0.906411399605945
subsample =  0.0453525064523448
colsample_bytree = 0.184887660899373


itterrow = 50
features_list = []
features_list.append(features)
test_score = []

for i in range(20):
    print(i, '-th iter with ', x_train.shape[1], 'features')    
    model =LGBMRegressor(n_estimators = n_estimators, 
                         num_leaves = max(2, int(2**(num_leaves * max_depth) - 1)), 
                         max_depth = max_depth,
                         min_child_samples = min_child_samples,
                         min_child_weight = min_child_weight,
                         subsample = subsample,
                         colsample_bytree = colsample_bytree)
    now = datetime.now()
    model.fit(x_train, y_train)
    print(datetime.now() - now)
    
    y_pred = model.predict(x_val)
    mse_val = mean_squared_error(np.expm1(y_val), np.expm1(y_pred))
    rmse_val = np.sqrt(mse_val)
    
    test_score.append(rmse_val)
    
    meta = pd.DataFrame(np.concatenate((features.reshape(-1,1), model.feature_importances_.reshape(-1,1)) , axis = 1))
    meta = meta.sort_values(by = 1, ascending = False).reset_index(drop = True)
    meta.columns = ['feature', 'importance']
    features = meta.loc[: meta.shape[0] - 2, 'feature'].values
    features_list.append(features)
    
    x_train = train[features].values
    x_val = val[features].values
    print(rmse_val)