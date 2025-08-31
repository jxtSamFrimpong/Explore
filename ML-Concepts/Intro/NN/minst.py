import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
model.summary()

predictions = model.predict(x_test)
for idx, item  in enumerate(predictions):
  # print("Predicted: ", item, "Original: ", y_test[idx])
  # pred = model.predict(x_test[0:1])  # probability vector
  predicted_class = np.argmax(item)  # index of the largest probability
  print("Predicted class:", predicted_class, "Original: ", y_test[idx])
  print("Predicted Probabilities: ", item)

  # --- plot image ---
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')  # show as grayscale
  plt.title(f"True: {y_test[idx]}, Pred: {predicted_class}")
  plt.axis('off')

  # --- plot prediction probabilities ---
  plt.subplot(1, 2, 2)
  plt.bar(range(10), item)
  plt.xticks(range(10))
  plt.title("Prediction Probabilities")

  plt.show()

