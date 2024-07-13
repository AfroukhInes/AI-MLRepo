from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)
df = X.copy()
df['target'] = y

# Save the DataFrame to a CSV file
csv_file_path = 'Diabetes_dataset.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset has been saved to '{csv_file_path}'")

print(X.head())
print(y.head())
dataset=pd.concat([X,y],axis=1)
print(dataset)


#Exploring the Dataset
print(dataset.describe())
#min=87  max=346  mean=152

'''
====> histogram of age and bmi

plt.figure(figsize=(10, 6))
df.hist(column='bmi', by="age")
plt.title('Histogram of BMI by age')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

=====> a lot of histograms to take in consid
'''


'''
====> histogram of sex and bmi

plt.figure(figsize=(10, 6))
df.hist(column='bmi', by="sex")
plt.title('Histogram of BMI by sex')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

=====> 2 histograms male and female!!
'''

# Preprocess the Data

#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#look for missing data
print(dataset[dataset.isnull().any(axis=1)])
'''
Empty DataFrame
Columns: [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target]
Index: [] 

======> no missing values'''


#Converting to binary classF abd spliting dataset to training  set and testing set 20%

y_binary = (y > y.mean()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

'''
100 trees serve as a good starting point.
====> random forst ========>> 74.15%
'''
randf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
randf_classifier.fit(X_train, y_train)
#making predection
y_pred = randf_classifier.predict(X_test)


# ====> Decision Tree Classifier  ========>> 66.29%

'''
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
#making predection
y_pred = dt_classifier.predict(X_test)
'''


#====> KNN Classifierr  ========>> 70.78%

'''
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
#making predection
y_pred = knn_classifier.predict(X_test)
'''


#====> SVM   ========>> 73.03%

'''
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
#making predection
y_pred = svm_model.predict(X_test)
'''




# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")
print(classification_report(y_test, y_pred))















