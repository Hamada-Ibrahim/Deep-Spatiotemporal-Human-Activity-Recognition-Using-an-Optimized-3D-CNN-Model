# Deep-Spatiotemporal-Human-Activity-Recognition-Using-an-Optimized-3D-CNN-Model

Description:
The proposed models implements human action recognition on video datasets using 3D Convolutional Neural Networks (Conv3D), Bidirectional LSTMs, and an Attention mechanism. The workflow includes:
a) Video-frames extraction from UCF50 dataset.
b) Frames preprocessing such as resizing and normalization.
c) Dataset splitting and parapartion for models.
e) Model training and tuning using different Keras Tuner such as Hyperband, Bayesian Optimization, and Random Search.
f) Model evaluation using accuracy, classification reports, and confusion matrices.

The project trains three different models:
a) A baseline Conv3D model.
b) A Conv3D + hyperparameter tuning model using Bayesian optimization.
c) A Conv3D + BiLSTM + Attention model for temporal sequence modeling.

‎Dataset Information:
UCF50 [1] action recognition dataset. Where videos are categorized into 50 action classes. The data format are .avi or .mpg video files organized by class folders.

Code Information:
The following functions are used for video frame extraction and visualization
a) frames_from_video_file()
b) to_gif()
c) display_gif()

Then the frames extracted are split into 70% for training, 20% for testing and 10% for validation using tf.data.Dataset pipelines for batching.

In training step, it uses early stopping, model checkpointing, and CSV logger. Tracks and plots loss & accuracy curves.

For the Hyperband model these method are used to create the Hyperband which then used to search for the best model hyperparameters
tuner = kt.Hyperband(...)
tuner.search(train_ds, validation_data=valid_ds, epochs=50)

The same steps are performed for Bayesian optimization model and Conv3D + BiLSTM + Attention model.
tuner = kt.BayesianOptimization(...)
tuner.search(train_ds, validation_data=valid_ds, epochs=50)

tuner = RandomSearch(...)
tuner.search(train_ds, validation_data=valid_ds, epochs=50)

In evaluation step, the method classification_report() is used for precision, recall, F1-score. Also, confusion_matrix() method is used to plotted with seaborn.

Usage Instructions:
The used dataset are placed in the correct path in Kaggle input section.

Requirements:
The proposed models need the following dependencies:
pip install tensorflow keras keras-tuner opencv-python scikit-learn seaborn matplotlib pandas

The core libraries used are:
a) tensorflow (Deep Learning, tf.data, model building)
b) keras-tuner (Hyperparameter tuning)
c) opencv-python (Video frame extraction)
d) numpy, pandas (Data handling)
e) matplotlib, seaborn (Visualization)
f) scikit-learn (Evaluation metrics, train-test split)

Methodology:
1) Data Preprocessing
a) Extract frames from each video.
b) Resize and normalize frames.
c) Store as NumPy arrays.

2) Dataset Creation
a) Create train/validation/test splits.
b) Use TensorFlow pipelines with batching & caching.

3) Modeling
a) Conv3D layers capture spatialtemporal features.
b) Batch Normalization & Dropout for regularization.
c) Dense layers for classification.
d) BiLSTM + Attention to capture temporal dependencies and emphasize important frames.

Hyperparameter tuning is used to tune the number of Conv3D layers, filters, dropout rate, dense units, learning rate, optimizer. The tuning methods includes Hyperband, Bayesian Optimization, and Random Search.

-----------------
[1] https://www.crcv.ucf.edu/data/UCF50.php
