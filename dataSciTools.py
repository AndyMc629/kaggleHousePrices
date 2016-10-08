import numpy as np
#Function to normalise data columns to range (0,1).
# takes a pandas dataframe as input, returns pd df
# output.
def normalise(df):
	result = df.copy()
	for feature_name in df.columns:
        	max_value = df[feature_name].astype(np.float32).max()
        	min_value = df[feature_name].astype(np.float32).min()
        	result[feature_name] = (df[feature_name].astype(np.float32) - min_value) / (max_value - min_value)
    	return result	

##function to train the data and output Coefficient of det on train.
## clf=sklearn model object.
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    print "Coefficient of determination on training set:",clf.score(X_train, y_train)
    Coeffs=linear.coef_
    print "Coeffs=", Coeffs
    #xtics = dataframe column names
    labels=list(train.columns.values)
    print "labels",labels[1:]
    x=np.linspace(1,Coeffs.size,Coeffs.size)
    plt.figure()
    plt.xlim(x[0]-0.5,x[-1]+0.5)
    plt.plot(x,Coeffs, linestyle='None',marker='D',markersize=10)
    plt.xticks(x,labels[1:]) #assumes 1st col is target variable.
    plt.savefig("Coeffs.pdf")
    plt.close()


