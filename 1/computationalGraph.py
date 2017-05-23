import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
#Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect. 
#Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively.
print(node1, node2)