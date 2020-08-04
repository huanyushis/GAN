from GAN.model import Generator, Discriminator
from GAN.Config import cfg
from GAN.dataset import load_data
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time

tf.keras.backend.set_floatx('float64')
input_tensor = tf.keras.layers.Input([100])
output_tensors = Generator(input_tensor)
model_generator = tf.keras.Model(input_tensor, output_tensors)

input_tensor = tf.keras.layers.Input([204])
output_tensors = Discriminator(input_tensor)
model_discriminator = tf.keras.Model(input_tensor, output_tensors)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=10)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=10)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=model_generator,
                                 discriminator=model_discriminator)


def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss


def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def train_step(images):
	noise = tf.random.uniform([cfg.batch_size, cfg.noise_dim], minval=0, maxval=10000)
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = model_generator(noise, training=True) #[512,204]
		real_output = model_discriminator(images) #
		fake_output = model_discriminator(generated_images)
		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, model_generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, model_generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model_discriminator.trainable_variables))
	return gen_loss, disc_loss


def data_generator(data):
	batches = len(data) // 32
	while (True):
		for i in range(batches):
			X = data[i * 32: (i + 1) * 32]
			yield X


def train():
	dataset = load_data("E:/高光谱GAN神经网络项目/data/8classd.dat")
	gen = data_generator(dataset)
	for epoch in range(cfg.epochs):
		start = time.time()
		gen_loss, disc_loss = train_step(next(gen))
		tf.print('Time for epoch {} is {} sec gen_loss:{} disc_loss:{}'.format(epoch + 1, time.time() - start, gen_loss,
		                                                                       disc_loss))
		# 每 15 个 epoch 保存一次模型
		if (epoch + 1) % 15 == 0:
			tf.print('epoch {} save_checkpoint'.format(epoch + 1))
			checkpoint.save(file_prefix=checkpoint_prefix)

	checkpoint.save(file_prefix=checkpoint_prefix)
