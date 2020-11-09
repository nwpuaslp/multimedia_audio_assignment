import tensorflow as tf
import numpy as np
import random
import os

from data_preprocessing import get_audio_dataset_features_labels, get_audio_test_dataset_filenames, get_audio_test_dataset_features_labels
def shuffle_randomize(dataset_features, dataset_labels):
	dataset_combined = list(zip(dataset_features, dataset_labels))
	random.shuffle(dataset_combined)
	dataset_features[:], dataset_labels[:] = zip(*dataset_combined)
	return dataset_features, dataset_labels


def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


DATASET_PATH = 'dataset'
ALLOWED_LABELS = ['one', 'two', 'three', 'four', 'five', 'zero']
ALLOWED_LABELS_MAP = {}
for i in range(0, len(ALLOWED_LABELS)):
	ALLOWED_LABELS_MAP[str(i)] = ALLOWED_LABELS[i]

dataset_train_features, dataset_train_labels, labels_one_hot_map = get_audio_dataset_features_labels(DATASET_PATH, ALLOWED_LABELS, type='train')
audio_filenames = get_audio_test_dataset_filenames(DATASET_PATH)

print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_labels.shape:', dataset_train_labels.shape)

# randomize shuffle
print('Shuffling training dataset')
dataset_train_features, dataset_train_labels = shuffle_randomize(dataset_train_features, dataset_train_labels)

# divide training set into training and validation
dataset_validation_features, dataset_validation_labels = dataset_train_features[11155:dataset_train_features.shape[0], :], dataset_train_labels[11155:dataset_train_labels.shape[0], :]
dataset_train_features, dataset_train_labels = dataset_train_features[0:11155, :], dataset_train_labels[0:11155, :]
print('dataset_validation_features.shape:', dataset_validation_features.shape, 'dataset_validation_labels.shape:', dataset_validation_labels.shape)

CLASSES = ['one', 'two', 'three', 'four', 'five', 'zero']
NUM_CLASSES = len(CLASSES)
NUM_EXAMPLES = dataset_train_features.shape[0]
NUM_CHUNKS = dataset_train_features.shape[1]	# 161
CHUNK_SIZE = dataset_train_features.shape[2]	# 99 
NUM_EPOCHS = 100
BATCH_SIZE = 32

x = tf.placeholder(tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE])
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])


def DNN(x):
	'''
	##############################################################
	define your DNN here
	x: model input with shape:(batch_size, frame_num, feature_size)
	frame_num is how many frame one wav have
    feature_size is the dimension of the feature
	##############################################################
	'''
	return 

logits = DNN(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		# initialize all global variables, which includes weights and biases

	# training start
	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			'''
			###############################################################################
			this is the training part, in this part you should get one batch data and train.
			you must calculate loss for each batch then add them up as total_loss
			###############################################################################
			'''

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		sum_accuracy_validation = 0.0
		sum_i = 0
		for i in range(0, int(dataset_validation_features.shape[0]/BATCH_SIZE)):
			batch_x = get_batch(dataset_validation_features, i, BATCH_SIZE)
			batch_y = get_batch(dataset_validation_labels, i, BATCH_SIZE)

			y_predicted = tf.nn.softmax(logits)
			correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
			accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
			accuracy_validation = accuracy_function.eval({x:batch_x, y:batch_y})

			sum_accuracy_validation += accuracy_validation
			sum_i += 1
			if(sum_i == 95):
				print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation, 'sum_i:', sum_i, 'sum_accuracy_validation:', sum_accuracy_validation)
		# training end

		# testing
		if epoch > 0 and epoch%2 == 0:
			y_predicted_labels = []
			audio_files_list = []
			dataset_test_features = []
			test_samples_picked = 0
			y_predicted = tf.nn.softmax(logits)

			for audio_file in audio_filenames:
				audio_files_list.append(audio_file)
				dataset_test_features.append(get_audio_test_dataset_features_labels(DATASET_PATH, audio_file))

				if len(audio_files_list) == 488:
					dataset_test_features = np.array(dataset_test_features)

					for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
						batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
						temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
						for element in temp:
							y_predicted_labels.append(element) 

					test_samples_picked += 488
					print('test_samples_picked:', test_samples_picked)

					# writing predicted labels into a csv file
					with open('run'+str(epoch)+'.csv','a') as file:	
						for i in range(0, len(y_predicted_labels)):
							file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
							file.write('\n')

					y_predicted_labels = []
					dataset_test_features = []
					audio_files_list = []

			# last set
			dataset_test_features = np.array(dataset_test_features)

			for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
				batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
				temp = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
				for element in temp:
					y_predicted_labels.append(element) 

			test_samples_picked += 488
			print('test_samples_picked:', test_samples_picked)

			# writing predicted labels into a csv file
			with open('run'+str(epoch)+'.csv','a') as file:	
				for i in range(0, len(y_predicted_labels)):
					file.write(str(audio_files_list[i]) + ',' + str(ALLOWED_LABELS_MAP[str(int(y_predicted_labels[i]))]))
					file.write('\n')


		
