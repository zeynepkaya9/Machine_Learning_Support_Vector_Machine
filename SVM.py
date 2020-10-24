import pandas as pd
import matplotlib.pyplot as plt

names = ['Id number: 1 to 214', 'RI: refractive index', 'Na: Sodium', 'Mg: Magnesium', 'Al: Aluminum', 'Si: Silicon', 'K: Potassium'
         , 'Ca: Calcium', 'Ba: Barium' , 'Fe: Iron', 'Type of glass']
data = pd.read_csv("GlassData.csv", names=names)

x = data.iloc[:, 1:10]
y = data.iloc[:,10:11]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(x_train, y_train)

result = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print(accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,result))