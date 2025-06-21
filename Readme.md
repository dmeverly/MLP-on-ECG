## ECG Cardiac Ischemia Prediction Using MLP of Various Levels of Complexity

Author: David Everly
Date: June 2025
Submitted as partial requirement for completion of CS 615 - Deep Learning
Drexel University

# Description
Main.py is a script which can run various experiments from command line. Each experiment trains and validates 3 MLPs on a specific dataset contained in the Data directory. Once the model is validated, an image the results is stored in the Images directory, one level above the code directory on the OS.  

# Language
Python

# Dependencies
1. framework and Data directory

# Execution
Program is intended to be run using Unix-like terminal such as Linux, macOS Terminal (untested), or MINGW64 (Git Bash) on Windows.  

The script recognizes the following commands:  

python main.py [model to run]  
  
Models to run:  
ecg200  
ecg5000  
ecg5000multiclass  
mit  
full  