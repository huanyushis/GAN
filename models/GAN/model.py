import tensorflow as tf

def Generator(noise): #生成器
	# z:[b,100]-->[b,1024]-->[b,2048]-->[b,1024]-->[b,204]
	out_put = tf.keras.layers.Dense(1024,activation="relu")(noise)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(2048,activation="relu")(out_put)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(1024,activation="relu")(out_put)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(204,activation="relu")(out_put)
	return out_put


def Discriminator(input): #判别器
	# [b,204]-->[b,512]-->[b,1024]-->[b,512]-->[b,1]
	out_put = tf.keras.layers.Dense(512,activation="relu")(input)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(1024,activation="relu")(out_put)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(512,activation="relu")(out_put)
	out_put = tf.keras.layers.BatchNormalization()(out_put)
	out_put = tf.keras.layers.Dense(1, activation="softmax")(out_put)
	return out_put

