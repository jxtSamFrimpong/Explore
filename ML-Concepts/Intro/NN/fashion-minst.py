import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape, y_test.shape)
print(x_train.shape, y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=7)


test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
model.summary()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


predictions = model.predict(x_test[:10])

for idx, prediction in enumerate(predictions):
    predicted_class = np.argmax(prediction)
    print('Predicted class: ', class_names[predicted_class])
    print('Actual class: ', class_names[y_test[idx]])

    # --- plot image ---
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')  # show as grayscale
    plt.title(f"True: {class_names[y_test[idx]]}, Pred: {class_names[predicted_class]}")
    plt.axis('off')

    # --- plot prediction probabilities ---
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.title("Prediction Probabilities")

    plt.show()