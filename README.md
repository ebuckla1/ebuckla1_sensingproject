# Sleep Quality Monitoring

This project implements a **non-contact sleep quality monitoring system** using a Raspberry Pi Pico 2W, ambient sensors, and a machine learning model. It collects sound, vibration, light, motion, temperature, and humidity data overnight, processes the data in 15 minute intervals, and predicts sleep scores without requiring the user to wear any device.

## Scripts

* **`main.py`** – Runs on the Pico to collect and log overnight sensor data into `sleep_data.csv`.
* **`sleep_train.py`** – Trains a Random Forest model on historical labeled data, saving the model and scaler.
* **`plot_sleep_scores.py`** – Extracts features from recorded data, applies the trained model, and visualizes sensor readings with predicted sleep scores.
