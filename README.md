Brain Tumor Detection System

This project is a web-based application designed to identify brain tumors from MRI scans. It utilizes a Convolutional Neural Network (CNN) to analyze uploaded images and provide a classification (Tumor Detected vs. No Tumor) along with a confidence percentage.
Project Description

The system provides an automated way to screen MRI scans using Deep Learning. By processing the visual patterns within the scan, the model distinguishes between healthy tissue and potential tumors. The application is built with a clear separation between the AI processing (backend) and the user interaction (frontend) to ensure a smooth "Vibe Check" experience.
Project Components

    app.py: The backend server built with Flask. It handles API requests, performs image preprocessing (resizing and normalization), and runs the prediction using the TensorFlow model.

    index.html: The frontend user interface. It allows users to upload MRI images and displays the results retrieved from the backend.

    brain_tumor_model.h5: The pre-trained Deep Learning model containing the weights and architecture optimized for tumor detection.

    train_model.ipynb: The Jupyter Notebook containing the source code for data augmentation, model training, and performance evaluation.

    requirements.txt: A list of all Python dependencies required for the project environment.

Team Members

    Faiza Farooqui

    Insha Fatima

    Alafiya Irshad

    Aman Mirza

Disclaimer

This software is developed for educational purposes as part of a college project. It is not a certified medical diagnostic tool and should not be used for actual clinical decision-making.