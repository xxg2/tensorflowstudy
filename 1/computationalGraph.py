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
#--------------------------placeholder------------------------
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
added_node=a+b # provides a shortcut for tf.add(a, b)
#7.5
print(sess.run(adder_node, {a:3, b:4.5}))
#[3., 7.]
print(sess.run(adder_node, {a: [1, 3], b:[2,4]}))
#---------------------------------------------------------------
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))
#------------------------------variables-------------------------------
#Constants are initialized when you call tf.constant, and their value can never change. 
#By contrast, variables are not initialized when you call tf.Variable. 
W = tf.Variables([.3], tf.float32)
b = tf.Variables([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variable_initializer()
sess.run(init)
#[1    0.30000001	0.60000002    0.90000004]
print(sess.run(linear_model, {x:[1,2,3,4]}))