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


==============================================
pattern recognize and machine learning

pattern in scatter plot like in scatter plot have 2 pattern


scatter plot help for prepare group of numerical data of our x variable



dimension reduction: for reduction like to decrease parameter of data , selected all dimension 
feature selection: select the important x variable for analyst like choose some or few variable

like selected the important variable for analyst


in statistic is called variable
in machine learning is called feature or dimension or attribute




components of standard must be = that
example: 4 dimension is mean that a standardivation value must = 4 
and if want to cut dimension must watch like this: 
> prcomp(x)
Standard deviations (1, .., p=4):
[1] 2.0562689 0.4926162 0.2796596 0.1543862

Rotation (n x k) = (4 x 4):
                     PC1         PC2         PC3        PC4
Sepal.Length  0.36138659 -0.65658877  0.58202985  0.3154872
Sepal.Width  -0.08452251 -0.73016143 -0.59791083 -0.3197231
Petal.Length  0.85667061  0.17337266 -0.07623608 -0.4798390
Petal.Width   0.35828920  0.07548102 -0.54583143  0.7536574
> z <- scale(x)
> prcomp(z)
Standard deviations (1, .., p=4):
[1] 1.7083611 0.9560494 0.3830886 0.1439265

Rotation (n x k) = (4 x 4):
                    PC1         PC2        PC3        PC4
Sepal.Length  0.5210659 -0.37741762  0.7195664  0.2612863
Sepal.Width  -0.2693474 -0.92329566 -0.2443818 -0.1235096
Petal.Length  0.5804131 -0.02449161 -0.1421264 -0.8014492
Petal.Width   0.5648565 -0.06694199 -0.6342727  0.5235971

like pc1 in dimension x1,x3,x4  => the most important variable
like pc2 in dimension x2 => important less than x1,x3,x4

