import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('final_model.keras', compile=False)

# Compile the model with categorical_crossentropy
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'test',  # Path to the test directory
    image_size=(224, 224),  # Adjust based on your model's input size
    batch_size=32,
    shuffle=False
)

# One-hot encode the labels
def one_hot_encode(image, label):
    num_classes = 2  # Number of classes (e.g., With Mask, Without Mask)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

test_dataset = test_dataset.map(one_hot_encode)

# Evaluate the model
loss, acc = model.evaluate(test_dataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {acc}")