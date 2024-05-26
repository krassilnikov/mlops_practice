#!/bin/bash

# Creating data
python3 data_creation.py

# Preprocessing data
python3 model_preprocessing.py

# Training model
python3 model_preparation.py

# Testing model
python3 model_testing.py