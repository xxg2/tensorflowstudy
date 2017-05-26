import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
#Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. 
#Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively.
#Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
print(node1, node2)
sess = tf.Session() 
#[3.0, 4.0]
print(sess.run([node1, node2]))
node3 = tf.add(node1, node2)
#7.0
print(sess.run(node3))
"""
1. 定义假设函数 htheta(x) = theta0+theta1x
2. 算方差并求和
3. 调整参数
4. 
4. 算梯度下降
"""
#--------------------------placeholder------------------------
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
added_node=a+b # provides a shortcut for tf.add(a, b)
#7.5
print(sess.run(added_node, {a:3, b:4.5}))
#[3., 7.]
print(sess.run(added_node, {a: [1, 3], b:[2,4]}))
#---------------------------------------------------------------
add_and_triple = added_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))
#------------------------------variables-------------------------------
#Constants are initialized when you call tf.constant, and their value can never change. 
#By contrast, variables are not initialized when you call tf.Variable. 
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
#[1    0.30000001	0.60000002    0.90000004]
print(sess.run(linear_model, {x:[1,2,3,4]}))
#-------------------------------------square error---------------------------------
y = tf.placeholder(tf.float32)
#linear_model - y creates a vector where each element is the corresponding example's error delta
square_deltas = tf.square(linear_model - y)
#we sum all the squared errors to create a single scalar that abstracts the error of all examples using reduce_sum()
loss = tf.reduce_sum(square_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#---------------------------------reassigning the values of W and b-----------------
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#-----------------------------gradient descent optimizer-------------------------------------
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W,b]))