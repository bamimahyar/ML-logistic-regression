# ML-logistic-regression
Image Classification with Logistic Regression

This project demonstrates a simple image classification pipeline using Python, NumPy, scikit-learn, and PIL. It applies logistic regression to grayscale images in .bmp format to classify them into two categories.
📁 Dataset

    20 BMP images (1.bmp to 20.bmp) located in the dataset/ directory

    Each image has dimensions 64x64 pixels

    Grayscale information is extracted from the red channel

🧪 Workflow

    Load training images and extract the red channel.

    Normalize pixel values to the range [0, 1].

    Train/Test split using train_test_split.

    Train a Logistic Regression model using scikit-learn.

    Visualize input images and predictions using matplotlib.

    Predict a new image (t2.bmp) using the trained model.

🔍 Key Features

    Uses PIL to open .bmp files

    Extracts grayscale data from the red channel

    Trains logistic regression on flattened 64x64 images

    Displays image samples before and after prediction

    Computes accuracy based on correct and incorrect predictions

🧠 Example

# Train model
model.fit(X_train, y_train)

# Predict on test image
prediction = model.predict(new_image)

# Show input and prediction
plt.imshow(new_image.reshape(64, 64))
plt.show()

📊 Evaluation

    Calculates number of correct (T) and incorrect (F) predictions

    Computes classification accuracy as percentage

📦 Dependencies

    numpy

    pandas

    matplotlib

    PIL

    seaborn

    scikit-learn

✅ Run the Project

Make sure your dataset/ folder includes:

    1.bmp to 20.bmp (for training)

    t2.bmp (for testing)

Then run:

python your_script_name.py

