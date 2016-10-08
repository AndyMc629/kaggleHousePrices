Began: 8/10/2016.

Kaggle house price prediction competition (playground = no prize money).

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data .

Data set has quite a lot of features, want to automatically dummy encode the
string entries whilst keeping the numerical entries. pd.get_dummies() doesn't
seem to play nice in that it drops the numerical columns ... and it also
expands the data frame massively, prefer what sklearn's label encoding
seems to do but am playing around with that.

