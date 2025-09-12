## Telco Customer Churn Prediction Machine Learning


## Dataset
dataset kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


===========================================================
Documentation: Converting Jupyter Notebook to HTML Report
Introduction

The project was originally developed in a Jupyter Notebook file named project.ipynb. To make the analysis and results easier to share without requiring Jupyter, the notebook was converted into an HTML report. This document explains the steps taken to perform the conversion.

Tools Used

Python (version 3.10.11)

Jupyter Notebook

nbconvert (a tool for converting Jupyter notebooks into various formats)

Required libraries installed:

pip install jupyter nbconvert

Conversion Process

Opened a terminal (or command prompt) in the folder containing project.ipynb.

Ran the following command to convert the notebook into an HTML file:

jupyter nbconvert --to html project.ipynb


If jupyter was not recognized, the command below was used instead:

python -m jupyter nbconvert --to html project.ipynb


After successful execution, a new file named project.html was created in the same folder.

Output

The HTML report includes both the code cells and their outputs (such as tables, charts, and printed results).

The file can be opened directly in any modern web browser (e.g., Chrome, Edge, Firefox).

Conclusion

By using jupyter nbconvert, the notebook project.ipynb was successfully converted into an HTML report. This allows the project’s results to be shared and viewed easily without requiring Jupyter Notebook.




=================================================================
Type of data: 

data is unnon class data or non class data

miss type data

max type is like example: category 5 varibles & numeric 5 variables

if dependent variable is Churn is a 

want to prepare group of data use unsupervised clustering

want to 

if we decide to analysis data we need to find type of data is non class data or unnon class data


dependent variable must be category so we can use unsuperivised learning  k-mean, ann,
dependent variable must be numeric so we can use supervised learning are k-mean, ann, svm

for ann the method that can use is MLP or perceptrons tell too what that i need to see