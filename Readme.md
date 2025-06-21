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

# References  

Al-Zaiti, S. S., Martin-Gill, C., Zègre-Hemsey, J. K., Bouzid, Z., Faramand, Z., Alrawashdeh, M. O., Gregg, R. E., Helman, S., Riek, N. T.,
Kraevsky-Phillips, K., Clermont, G., Akcakaya, M., Sereika, S. M., Van Dam, P., Smith, S. W., Birnbaum, Y., Saba, S., Sejdic, E., & Callaway,
C. W. (2023). Machine learning for ECG diagnosis and risk stratification of occlusion myocardial infarction. Nature Medicine, 29(7), 1804–1813. https://doi.org/10.1038/s41591-023-02396-3
Dau, H. A., Keogh, E., Kamgar, K., Yeh, C.-C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., et al. (2019). The UCR Time Series Classification
Archive. University of California, Riverside. https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E.
(2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*,
101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215
Institute of Medicine (Ed.). (2000). To err is human: Building a safer health system. National Academies Press. http://www.nap.edu/catalog/9728
Institute of Medicine (Ed.). (2009). Crossing the quality chasm: A new health system for the 21st century (9. print). National Acad. Press.
Moody, G.B., & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50. doi:10.1109/51.932724
Xiong, P., Lee, S. M.-Y., & Chan, G. (2022). Deep Learning for Detecting and Locating Myocardial Infarction by Electrocardiogram: A
Literature Review. Frontiers in Cardiovascular Medicine, 9, 860032. https://doi.org/10.3389/fcvm.2022.860032