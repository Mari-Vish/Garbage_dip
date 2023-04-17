import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

data_dir = 'Garbage classification/'

total_dir = len(os.listdir(data_dir))
total_files = 0
for dirname, _, filenames in os.walk(data_dir):
    files_counter = 0
    for file in filenames:
        files_counter += 1
    total_files += files_counter
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=100
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=100
)
class_names = train_ds.class_names
print(class_names)
train_batch = train_ds.as_numpy_iterator().next()
validation_batch = validation_ds.as_numpy_iterator().next()
input_shape = (256,256,3)
base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape)
base_model.trainable = True
base_model.summary()
tuning_layer_name = 'conv5_block1_preact_bn'
tuning_layer = base_model.get_layer(tuning_layer_name)
tuning_index = base_model.layers.index(tuning_layer)
for layer in base_model.layers[:tuning_index]:
    layer.trainable =  False
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
], name='data_augmentation')
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(7, activation='softmax')
])

learning_rate = 0.00001
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    metrics=['accuracy']
)
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=30
)
model.save('model.h5');