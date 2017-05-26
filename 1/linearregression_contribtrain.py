 import tensorflow as tf
 import numpy as np

#Declare list of features. 
 feature = [tf.contrib.layers.real_valued_column("x", dimension=1)]
"""
 An estimator is the front end to invoke training and evaluation(inference). There are many predefined types like linear regression, 
 logistic regression, linear classification, logistic classification and many neural network classifiers and regressors. 
 The following code provides an estimator that des linear regression. 

"""
 estimator = tf.contrib.learn.LinearRegressor(feature_columns=feature)

 x = np.array([1.,2.,3.,4.])
 y = np.array([0.,-1.,-2.,-3.])
 input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x},y,batch_size=4,num_epochs=1000)
 estimator.fit(input_fn=input_fn, steps=1000)

 print(estimator.evaluate(input_fn=input_fn))