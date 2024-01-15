Instructions.


all files must be run from the root of the repository with python 3.11 or older 

steps:

1. with pip install verify the installation of the following packages:
 * csv
 * matplotlib
 * scikit-learn
 * tensorflow
 * numpy
2. Run cleanerANDnominaliser.py
3. Run complete.py It will take about an hour. To run the column avg based completer run complete2.py.
4. Run classify.py here you can se the reports on the classification accuracy, if you want to use the column avg dataset use 'tasks/Part_A/task_1/cleaned_and_filled2.csv' in line 21 instead of 'tasks/Part_A/task_1/cleaned_and_filled.csv'.
5. Run correcterandfeatures.py
6. Run merger.py
7. Run clustering.py , comment the line 91 if you want clustering without PCA