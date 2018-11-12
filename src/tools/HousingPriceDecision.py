# /data/pyenv/bin python3
# -*- coding:utf-8 -*-
# Author: wngruihuanbeijing@126.com

import os
import sys
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def house_price_decision(fpath):
    melt_data = pd.read_csv(fpath)
    melt_data.describe()
    melbourne_data = melt_data.dropna(axis=0)
    y = melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    melbourne_model = DecisionTreeRegressor(random_state=1)
    print(melbourne_model.fit(X, y))
    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))


if __name__ == "__main__":
    pre_dirname = os.path.abspath(os.path.dirname(os.getcwd()))
    fname = "melb_data.csv"
    fpath = "{}/data/{}".format(pre_dirname, fname)
    house_price_decision(fpath)
