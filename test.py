# Assuming you have already split your data into training and test sets
# and have trained your model

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test data
predictions = model.predict(test_images)

# To see the predicted categories for the first few test images
predicted_categories = np.argmax(predictions, axis=1)
print(predicted_categories[:10])

# To see the actual categories for the first few test images
print(test_labels[:10])