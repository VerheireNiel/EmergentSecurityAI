'''
The goal of this case is training a classifier to detect fraudulent transactions for credit card data.
More information of the case can be found in the pdf on leho.

Firstly we import some packages
'''

# Importing some packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.utils import shuffle

'''
The full creditcard dataset is avaiable. 
The data is present in src -> data_ex. 
read in 'creditcard.csv' and check what collumns you have and their datatypes.
Print the first 5 rows of the dataframe.
'''

creditcard_data = pd.read_csv("creditcard.csv")
print(creditcard_data.dtypes)
print(creditcard_data.head())



'''
Drop the two collumns 'Index' and 'Time' and store your result. 
Check if you succeeded by displaying the resulting dataframe. 
'''
creditcard_data.reset_index(drop=True, inplace=True)
df = creditcard_data.drop("Time", 1)
print(df.head())

'''
This dataset was partly anonimized, only the collumns Time and amount are original. 
Looking at the resulting dataframe of the previous codeblock:

Is there any categorical/numerical data ? The class column is categorical

What preprocessing can we do to this dataset ? We have to split our dataset into attributes and labels

After answering these questions for yourself, 
make a decision for your preprocessing step and execute this on the resulting dataframe of the previous codeblock.

Note : Preprocessing functions from scikit-learn usually return a numpy array.
       You can put the result back into a pandas dataframe and don't forget to add column names. 
'''

headers = list(df)
attributes = df.iloc[:, :-1].values
labels = df.iloc[:, 29].values

classes = df['Class']

df = df.drop(['Class'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.DataFrame(scaler.fit_transform(df))
df['Class'] = classes
df.columns = headers

print(df.head())


'''
It's best practice to randomly shuffle the rows of a dataset. Scikit-learn has all sorts of usefull 
preprocessing functions. In the code below the function shuffle is used. Everytime it's called it returns a randomly
shuffle dataset. You can adapt 'df' in the code below if needed.

Note: The shuffle function will return a pandas dataframe if a pandas dataframe is given. 
'''

df = shuffle(df)

'''
The 'Class' collumn of the dataframe contains the labels. The value 1 indicates fraud, 0 normal transactions.
This collumn can be used to filter the dataframe df into two parts. 
Filter the dataset and store them into two dataframes: df_normal and df_fraud.
'''

df_normal = df.loc[df['Class'] == 0]
df_fraud = df.loc[df['Class'] == 1]


'''
We will use a part of the data for training and a part for testing. 
A good split of training and test data is a 70%/30% split.

The first 70% of df_normal and df_fraud should be stored in df_train
The last 30% should be stored in df_test. 

Note : You can use pd.concat() to stick together dataframes.  
'''

df_train = pd.concat([df_normal.head(int(len(df_normal)*(70/100))), df_fraud.head(int(len(df_fraud)*(70/100)))])
df_test = pd.concat([df_normal.tail(int(len(df_normal)*(30/100))), df_fraud.tail(int(len(df_fraud)*(30/100)))])


'''
We will use the K-nearest neighbour algorithm. 
See: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

You should try out different values for 'n_neighbors', 'weights', 'p', 'metric'. 
Stick to the rule of thumb to start with simple KNN parameters. Improve accuracy by trying out different parameters. 

The explanation for n_neighbors, weights and metric should be clear from the slides.

The parameter 'p' relates to one specific metric family, the 'minkwoski' metric. 
setting p=1,2,3 gives different metrics if metric='minkowski'. This is the default value for the metric argument

Setting the argument 'n_jobs=-1' will use all of your cpu cores. This gives you faster results but make sure you don't 
have any other heavy applications open while running the code with this option.

So make a KNN object now starting with some simple arguments and store it. 
'''

classifier = KNeighborsClassifier(n_neighbors=1, p=2, n_jobs=-1, metric='minkowski', weights='distance')

'''
Use the KNN model to train on the df_train dataset. 
The function for training expects two arguments:  a dataframe and labels. 
Make sure the dataframe you pass into the function doesn't contain the collumn for labels.
'''

x_train = pd.DataFrame(df_train.iloc[:, :-1].values)
y_train = pd.DataFrame(df_train.iloc[:, 29].values)

classifier.fit(x_train, y_train)

'''
Using the trained KNN, predict on the test dataset and store the results.
'''

x_test = pd.DataFrame(df_test.iloc[:, :-1].values)
y_test = pd.DataFrame(df_test.iloc[:, 29].values)

y_pred = classifier.predict(x_test)

'''
It's time to make a visualization of the results. We will compare the results of the prediction versus the ground 
truth of the labels.

1) Make two arrays of size of test dataset. 
One has values from 0 to size of test dataset. Second array has integer values '1' .

2) Make a boolean array that is true for fraud rows of the test dataset and false otherwise. 

3) Make two scatter plots. A scatter plot of matplotlib expects two numpy arrays. One for the x-values and one for 
   the y-values. For an example of this look at 'src->case->graph.jpg'.
   - Make one scatter plot with points y = 1 and the other with y= 1.5
   - Only the predicted fraud points should be plotted. plot these for y=1 . For the plot with y=1.5 plot points
     with label fraud.
   - Give the predicted fraud points the color red. Give the labeled fraud points the color yellow.
   - give arguments: 'marker', 's (=point size)', 'label'. To both scatter plots.
   - give the plot a title, show the labels and set a ylim such that all points can be seen.

   => show the plot and check if everything is as expected.
'''

arr1 = np.arange(x_test.shape[0])
arr2 = np.ones(x_test.shape[0])

class_df_testing_row = y_test.iloc[:, -1].values
boolArr = [0] * len(class_df_testing_row)

for i in range(0, len(boolArr)):
    if round(class_df_testing_row[i]) == 1:
        boolArr[i] = True
    else:
        boolArr[i] = False

predicted_fraud_points = []
for i in range(0, len(y_pred)):
    if round(y_pred[i]) == 1:
        predicted_fraud_points.append(i)

y_vals = [1] * len(predicted_fraud_points)
plt.scatter(x=predicted_fraud_points, y=y_vals, c="red", marker="D", label="Predicted as Fraud")

label_fraud_points = []
for i in range(0, len(y_pred)):
    if boolArr[i] == 1:
        label_fraud_points.append(i)

y_vals = [1.5] * len(label_fraud_points)
plt.scatter(x=label_fraud_points, y=y_vals, c="yellow", marker="D", s=5, label="Fraud")

plt.title("Predicted and Fraud")
plt.ylim(0.5, 2)
plt.legend()
plt.show()


'''
Calculate the confusion matrix. 
Since we have only two classes for this classifier problem the matrix will be of shape = (2,2). 

Determine 'True Positives', 'True negatives','False positives','False negatives' for this matrix.
Make a dictionary of these four values and store it. display the dictionary.

repeat the process of changing KNN-parameters and producing results to improve your algorithm.
At most there should be 50 missclassified data points. 

Note: The order of the collumns can be set for the confusion_matrix. 
      Make sure to set these, because the first value encountered will be set as first collumns. 
      This will be random depending on the shuffling of the dataset!
'''

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
Provide following plots and answer the questions in the comment block below:

- The plots of your best and second best parameter choice. You can save the plot by right-clicking on it
  (see plot_save.jpg).


Questions:
1) Data reading: After dropping the collumns 'Index' and 'Time', what collumns do we have left?
2) Data preprocessing: We've dropped the collumn 'Time'. What information do you lose when doing this?
If we keep the 'Time' and shuffle the dataset what information do we lose ?
3) KNN object: Give the parameters that give the best and second best results for you.
4) Prediction: If we train on the entire dataset (with k=1) and tried to make predictions on points within
our dataset, what prediction accuracy would we have ?  
5) copy the printout of the dictionary with confusion matrix values, for your best and second best results.
'''

'''
1) V1-V28, Amount & Class
2) We lose the order of transactions but this shouldn't matter.
3) n_neighbors=7, p=2, metric='minkowski', weights='distance'
4) The prediction accuracy is lower as we use less neighbours to compare against to.
5)

BEST:
[[85283    11]
 [   44   103]]

2nd BEST:
[[85282    12]
 [   35   112]]
 
'''