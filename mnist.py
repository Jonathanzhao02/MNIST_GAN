import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Conv2DTranspose
import logging
import numpy
tf.get_logger().setLevel(logging.FATAL)

mnist = tf.keras.datasets.mnist

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-3)
acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)

"""28x28"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.one_hot(y_train, 10, on_value=1.0, off_value=0.0)
y_test = tf.one_hot(y_test, 10, on_value=1.0, off_value=0.0)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(100)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l0 = Flatten()
        self.l1 = Dense(128, activation='relu')
        self.l2 = Dropout(0.2)
        self.l3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

model = MyModel()

epochs = 500

@tf.function
def train_step (x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss(logits, y)
        acc_metric.update_state(y, logits)
        loss_metric.update_state(y, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

@tf.function
def validate_step (x, y):
    logits = model(x, training=False)
    acc_metric.update_state(y, logits)
    loss_metric.update_state(y, logits)

def train (dataset):
    for step, (x, y) in enumerate(dataset):
        train_step(x, y)
    print('Epoch', i, 'Loss:', float(loss_metric.result()), 'Acc:', float(acc_metric.result()))
    loss_metric.reset_states()
    acc_metric.reset_states()

def validate (dataset):
    for step, (x, y) in enumerate(dataset):
        validate_step(x, y)
    print('Validation Loss:', float(loss_metric.result()), 'Acc:', float(acc_metric.result()))
    loss_metric.reset_states()
    acc_metric.reset_states()

for i in range(0, epochs):
    epoch_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True)
    train(epoch_dataset)
    validate(valid_dataset)

model.save('mnist_conv.h5')
