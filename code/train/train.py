import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow.keras.callbacks import Callback

from azureml.core import Run
run = Run.get_context()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, acc = model.evaluate(x_test,y_test)
run.log("Validation Accuracy", acc)

os.makedirs('./outputs/model', exist_ok=True)

model_json = model.save("./outputs/model/")

print("model saved in ./outputs/model folder")