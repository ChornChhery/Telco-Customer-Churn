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