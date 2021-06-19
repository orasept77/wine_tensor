import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

wine_names = ['Klasa', 'Alkohol', 'Kwas jab³kowy', 'Popió³', 'Alkalicznosc', 'Magnez', 'Zawartosc fenoli', 'Flawonoidy', 'Fenole nieflawonoidowe', 'Proantocyjanidyny', 'Intensywnoœæ koloru', 'Odcieñ', 'OD280/OD315', 'Prolina']
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names)

X = wine.drop('Klasa',axis=1)
y = wine['Klasa']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#print(mlp.coefs_)