import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import dataSciTools as ds 
from sklearn import preprocessing
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# READ IN AND CLEAN DATA
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#read in data.
data=pd.read_csv("Data/train.csv")

# don't like get_dummies, creates a far
# larger data frame and seems to get 
# rid of pure numeric columns.
# sklearn's label_encoder seems better
# - just working out how to apply this 
# to a pandas dataframe rather than a numpy
# array.
#data=pd.get_dummies(data,dummy_na=False)
preprop=preprocessing.LabelEncoder()
data.apply(preprocessing.LabelEncoder().fit_transform)
print data
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## DATA CLEANED AND SUMMARY STATS PLOTTED
## APPLY ML ALGO
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## NORMALISE, SPLIT INTO TEST/TRAIN
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ds.normalise(data)

#convert train dataframe to numpy array
#x_np_train=data.values[:,:-2].astype(np.float32)
#y_np_train=data.values[:,-1].astype(np.float32)

#print y_np_train
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## MODEL - NOTE: I PLAYED AROUND WITH A FEW FOR FUN.
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##Uncomment below if you want kfold cross validation.
#from sklearn.cross_validation import *

##LINEAR REGRESSION
#from sklearn import linear_model
#create lin reg object.
#linear=linear_model.LinearRegression()
##ridge_regression
#ridge=linear_model.Ridge(alpha=1.0)
##lasso_regression
#lasso=linear_model.Lasso(alpha=1.0)
##Stochastic grad descrent regressor rather than closed form.
#linear=linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)

##SVM for REGRESSION
#from sklearn import svm
#svr = svm.SVR(kernel='linear')

##RANDOM FORESTS for REGRESSION
#from sklearn import ensemble
#extraTrees=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)

##GAUSSIAN PROCESSES
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel as C
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gpr=GaussianProcessRegressor(kernel=kernel,alpha=0,optimizer=None, normalize_y=True)
#gpr.fit(x_np_train,y_np_train)

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## TRAINING AND CROSS-VALIDATION
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#print "LinReg: ", train_and_evaluate(linear,x_np_train,y_np_train), "\n"
#print "ridge: ", train_and_evaluate(ridge,x_np_train,y_np_train), "\n"
#print "lasso: ", train_and_evaluate(lasso,x_np_train,y_np_train), "\n"
#print "SVR: ", train_and_evaluate(svr,x_np_train,y_np_train)
#print "Extra trees: ", train_and_evaluate(extraTrees, x_np_train,y_np_train)
#print "gpr: ", train_and_evaluate(gpr,x_np_train, y_np_train)

##Predict Output
#predicted=linear.predict(x_np_test)

## Plot predicted vs true data.
#plt.figure()
#plt.scatter(y_np_test,predicted)
#plt.plot([y_np_test.min(), y_np_test.max()], [y_np_test.min(), y_np_test.max()], 'k--', lw=4)
#plt.xlabel('Measured')
#plt.ylabel('Predicted')
#plt.savefig("Prediction.pdf")
#plt.close()

##Rerun with most strongly correlated features 
## - PropertyType and InLondon
##convert train dataframe to numpy array
#x_np_train=train.values[:,[2,3,4]].astype(np.float32)

##convert test dataframe to numpy array
#x_np_test=test.values[:,[2,3,4]].astype(np.float32)

#print "LinReg: ", train_and_evaluate(linear,x_np_train,y_np_train), "\n"

##Predict Output
#predicted=linear.predict(x_np_test)

## Plot predicted vs true data.
#plt.figure()
#plt.scatter(y_np_test,predicted)
#plt.plot([y_np_test.min(), y_np_test.max()], [y_np_test.min(), y_np_test.max()], 'k--', lw=4)
#plt.xlabel('Measured')
#plt.ylabel('Predicted')
#plt.savefig("Prediction.pdf")
#plt.show()
#plt.close()



