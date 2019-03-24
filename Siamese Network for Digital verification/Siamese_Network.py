from Cnn_model import *
import tensorflow as tf
import numpy as np


def next_batch():
    X_left = np.load('X_left.npy')
    X_right = np.load('X_right.npy')
    label = np.load('label.npy')

    return X_left,X_right,label


X_l,X_r,lab = next_batch()

print(X_l.shape,X_r.shape,lab.shape)



    

X_l = X_l.reshape((-1,100,100,1))
X_r = X_r.reshape((-1,100,100,1))
label_sim = lab.reshape((-1,1))




placeholder_shape = [None,100,100,1]



left = tf.placeholder(tf.float32, placeholder_shape, name='left')
right = tf.placeholder(tf.float32, placeholder_shape, name='right')
label = tf.placeholder(tf.int32, [None, 1], name='label')
label_float = tf.to_float(label)
margin = 0.5
left_output = cnn_model(left, reuse=False)
right_output = cnn_model(right, reuse=True)
loss = contrastive_loss(left_output, right_output, label_float, margin)
epochs = 7


# Setup Optimizer
global_step = tf.Variable(0, trainable=False)

# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

# Start Training
saver = tf.train.Saver()
with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		
		
		#train iter
		for epoch in range(epochs):

			#get the next training set
			#batch_left, batch_right, batch_similarity = next_batch()

			_, l = sess.run([train_step, loss],feed_dict={left:X_l, right:X_r, label: label_sim})
			
			print('Epoch {0} : Loss = {1}'.format(epoch,l))

	        
