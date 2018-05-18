# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:38:48 2018

@author: matti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, date
import time
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from math import ceil
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras.optimizers
import json


input_folder=r'input'

train = pd.read_csv(f'{input_folder}\\sales_train_v2.csv')
test = pd.read_csv(f'{input_folder}\\test.csv')
submission = pd.read_csv(f'{input_folder}\\sample_submission.csv')
items = pd.read_csv(f'{input_folder}\\items.csv')
item_cats = pd.read_csv(f'{input_folder}\\item_categories.csv')
shops = pd.read_csv(f'{input_folder}\\shops.csv')

test_shops = test.shop_id.unique()
train = train[train.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train = train[train.item_id.isin(test_items)]

MAX_BLOCK_NUM = train.date_block_num.max()
MAX_ITEM = len(test_items)
MAX_CAT = len(item_cats)
MAX_YEAR = 3
MAX_MONTH = 4 # 7 8 9 10
MAX_SHOP = len(test_shops)

save_path=r'runs'
load_path=r'runs\2018-05-15_1_40_43_salesPredict\cv_0'
do_descriptive=False
do_featurePreparation=False
do_loadFeatures=True

class logger_saver():
    def __init__(self,results_path):
        folder_name = "/{0}_{1}_{2}_{3}_salesPredict". \
        format(date.today(), datetime.now().hour, datetime.now().minute, datetime.now().second)        
        self.folder_path=results_path + folder_name     
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.cv_round=0
        
    def next_cv_folder(self):        
        cv_folder =  '/cv_{}'.format(self.cv_round)
        self.cv_folder_path=self.folder_path+cv_folder
        self.log_path=self.cv_folder_path  + '/log.txt'
        self.img_path=self.cv_folder_path
        if not os.path.exists(self.cv_folder_path):
            os.mkdir(self.cv_folder_path)
        self.cv_round +=1    
    def log_numpy(self, name, obj_numpy):
        np.save(f'{self.img_path}\\{name}',obj_numpy)    
    def log_collection(self, name, obj_coll):
        with open(f'{self.img_path}\\{name}', 'w') as myfile:
            json.dump(obj_coll, myfile)
    def log_image(self, name, myplt=None):
        name=name.replace(':','_')
        name=name.replace('/','__')
        if not myplt:
            plt.savefig(self.cv_folder_path + r'/' + name)
        else: 
            myplt.savefig(self.cv_folder_path + r'/' + name)
            myplt.close()    
    def log_dict(self, dictionary):
        with open(self.log_path, 'a') as log:            
            for prop in dictionary.keys():
                log.write("{}: {}\n".format(prop, dictionary[prop]))        

logger=logger_saver(save_path)

if do_descriptive:
    grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
    fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
    num_graph = 10
    id_per_graph = ceil(grouped.shop_id.max() / num_graph)
    count = 0
    for i in range(5):
        for j in range(2):
            sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
            count += 1
            
    plt.show()

if do_featurePreparation:
    # add categories
    train = train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
    
    train['month'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
    train['year'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))

if do_descriptive:
    fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
    num_graph = 10
    id_per_graph = ceil(train.item_category_id.max() / num_graph)
    count = 0
    for i in range(5):
        for j in range(2):
            sns.pointplot(x='month', y='item_cnt_day', hue='item_category_id', 
                          data=train[np.logical_and(count*id_per_graph <= train['item_category_id'], train['item_category_id'] < (count+1)*id_per_graph)], 
                          ax=axes[i][j])
            count += 1
            
    plt.show()
    
    fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
    num_graph = 10
    id_per_graph = ceil(train.item_category_id.max() / num_graph)
    count = 0
    for i in range(5):
        for j in range(2):
            sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id', 
                          data=train[np.logical_and(count*id_per_graph <= train['item_category_id'], train['item_category_id'] < (count+1)*id_per_graph)], 
                          ax=axes[i][j])
            count += 1
            
    plt.show()

maxlen = 4 # 4 months
step = 1

scaler = StandardScaler()
cnt_scaler = StandardScaler()

scaler.fit(train.item_price.as_matrix().reshape(-1, 1))
cnt_scaler.fit(train.item_cnt_day.as_matrix().reshape(-1, 1))
    

if do_featurePreparation:
    train = train.drop('date', axis=1)
    train = train.drop('item_category_id', axis=1)
    train = train.groupby(['shop_id', 'item_id', 'date_block_num', 'month', 'year']).sum()
    train = train.sort_index()
    
    scaler = StandardScaler()
    cnt_scaler = StandardScaler()
    
    scaler.fit(train.item_price.as_matrix().reshape(-1, 1))
    cnt_scaler.fit(train.item_cnt_day.as_matrix().reshape(-1, 1))
    
    train.item_price = scaler.transform(train.item_price.as_matrix().reshape(-1, 1))
    train.item_cnt_day = cnt_scaler.transform(train.item_cnt_day.as_matrix().reshape(-1, 1))
    
    train.reset_index().groupby(['item_id', 'date_block_num', 'shop_id']).mean()
    
    price = train.reset_index().set_index(['item_id', 'shop_id', 'date_block_num'])
    price = price.sort_index()
    
    def convert(date_block):
        mydate = datetime(2013, 1, 1)
        mydate += relativedelta(months = date_block)
        return (mydate.month, mydate.year)
    
    def closest_date_block(current_day, item_id, shop_id):
        """Find the block_date which is closest to the current_day, given item_id and shop_id. Returns index integer"""
        if (item_id, shop_id) in price.index:
            search_lst = np.array(price.loc[(item_id, shop_id)].index)        
            return search_lst[np.abs(current_day - search_lst).argmin()]
        return -1
                    
    def closest_price(current_day, item_id, shop_id):
        closest_date = closest_date_block(current_day, item_id, shop_id)
        if closest_date != -1:
            return price.loc[( item_id, shop_id, closest_date )]['item_price']
        return np.nan
    
    def closest_price_lambda(x):
        return closest_price(34, x.item_id, x.shop_id)
    
    assert closest_date_block(18, 30, 5) == 18
    
    # Some simple math to know what date_block_num to start learning
    print(convert(6))
    print(convert(18))
    print(convert(30))
    
    maxlen = 4 # 4 months
    step = 1
    # 0: train, 1: val, 2:test
    sentences = [[],[],[]]
    next_chars = [[], []]
    BLOCKS = [6, 18, 30]
    
    month=None
    year=None
    
    start_time = time.time()
    
    for s in test_shops:
        shop_items = list(train.loc[s].index.get_level_values(0).unique())
        for it in shop_items:
            for i_index, i in enumerate(BLOCKS):
                sentence = []
                closest_pc = closest_price(i, it, s)            
                for j in range(maxlen+1):
                    if j < maxlen:
                        if (s, it, i+j) in train.index:
                            r = train.loc[(s, it, i + j)].to_dict(orient='list')                    
                            closest_pc = r['item_price'][0]
                            item_cnt_day = r['item_cnt_day'][0]
                            row = {'shop_id': s, 'date_block_num': i+j, 'item_cnt_day': item_cnt_day, 
                                   'month': month, 'item_id': it, 'item_price': closest_pc, 'year': year}
                        else:
                            month, year = convert(i+j)                    
                            row = {'shop_id': s, 'date_block_num': i+j, 'item_cnt_day': 0, 
                                   'month': month, 'item_id': it, 'item_price': closest_pc, 'year': year}
                        sentence.append(row)
                    elif i_index < 2:   # not in test set
                        next_chars[i_index].append(row)
                sentences[i_index].append(sentence)
    
    elapsed_time = time.time() - start_time
    print(f'tempo calcolo sequences: {elapsed_time}')
    
    x_train_o = np.array(sentences[0])
    x_val_o = np.array(sentences[1])
    x_test_o = np.array(sentences[2])
    y_train = np.array([x['item_cnt_day'] for x in next_chars[0]])
    y_val = np.array([x['item_cnt_day'] for x in next_chars[1]])
    
    logger.next_cv_folder()
    logger.log_numpy('x_train_o',x_train_o)
    logger.log_numpy('x_val_o',x_val_o )
    logger.log_numpy('x_test_o',x_test_o)
    logger.log_numpy('y_train',y_train)
    logger.log_numpy('y_val',y_val)    
    logger.log_numpy('next_chars',next_chars)
    logger.log_numpy('sentences',sentences)        
    load_path=logger.img_path

if do_loadFeatures:    
    x_train_o = np.load(f'{load_path}\\x_train_o.npy')
    x_val_o = np.load(f'{load_path}\\x_val_o.npy')
    x_test_o = np.load(f'{load_path}\\x_test_o.npy')
    next_chars= np.load(f'{load_path}\\next_chars.npy')
    y_train = np.array([x['item_cnt_day'] for x in next_chars[0]])
    y_val = np.array([x['item_cnt_day'] for x in next_chars[1]])
    


length = MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 + 1 + 1

from sklearn import preprocessing

shop_le = preprocessing.LabelEncoder()
shop_le.fit(test_shops)
shop_dm = dict(zip(test_shops, shop_le.transform(test_shops)))

item_le = preprocessing.LabelEncoder()
item_le.fit(test_items)
item_dm = dict(zip(test_items, item_le.transform(test_items)))

month_le = preprocessing.LabelEncoder()
month_le.fit(range(7,11))
month_dm = dict(zip(range(7,11), month_le.transform(range(7,11))))

#cat_le = preprocessing.LabelEncoder()
#cat_le.fit(item_cats.item_category_id)
#cat_dm = dict(zip(item_cats.item_category_id.unique(), cat_le.transform(item_cats.item_category_id.unique())))


def vectorize(inp):
    print('Vectorization...')   
    x = np.zeros((len(inp), maxlen, length), dtype=np.float32)
    for i, sentence in enumerate(inp):
        for t, char in enumerate(sentence):            
            x[i][t][ shop_dm[char['shop_id']] ] = 1        
            x[i][t][ MAX_SHOP + item_dm[char['item_id']] ] = 1
            x[i][t][ MAX_SHOP + MAX_ITEM + month_dm[char['month']] ] = 1
            x[i][t][ MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 ] = char['item_price']
            x[i][t][ MAX_SHOP + MAX_ITEM + MAX_MONTH + 1 + 1] = char['item_cnt_day']    
    return x

x_train = vectorize(x_train_o)
x_val = vectorize(x_val_o)
x_test = vectorize(x_test_o)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(32, input_shape=(maxlen, length)))
model.add(Dense(1, activation='relu'))

myoptimizer = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=myoptimizer)

#model.fit(x_train, y_train, batch_size=128, epochs=13)

#####fit on the train and evaluates the score
#import math
#from sklearn.metrics import mean_squared_error

# make predictions
#predict_train = model.predict(x_train)
#predict_val = model.predict(x_val)
# invert predictions
#predict_train = cnt_scaler.inverse_transform(predict_train)
#y_train = cnt_scaler.inverse_transform(y_train)
#predict_val = cnt_scaler.inverse_transform(predict_val)
#y_val = cnt_scaler.inverse_transform(y_val)
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(predict_train, y_train))
#print('Train Score: %.2f RMSE' % (trainScore))
#valScore = math.sqrt(mean_squared_error(predict_val, y_val))
#print('Test Score: %.2f RMSE' % (valScore))
#For 1 epoch
#Train Score: 3.85 RMSE
#Test Score: 4.29 RMSE
    
    
model.fit(x_train, y_train, batch_size=256, epochs=500)
   
predict_test = model.predict(x_test)
predict_test = cnt_scaler.inverse_transform(predict_test)


test = test.set_index(['shop_id', 'item_id'])
test['item_cnt_month'] = 0

for index, sentence in enumerate(x_test_o):
    (shop_id, item_id) = (sentence[0]['shop_id'], sentence[0]['item_id'])
    test.loc[(shop_id, item_id)]['item_cnt_month'] = predict_test[index]
    
test = test.reset_index().drop(['shop_id', 'item_id'], axis=1)
test.to_csv(f'{save_path}\\submission.csv', index=False)
