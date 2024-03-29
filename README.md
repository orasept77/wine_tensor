# wine_tensor
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data for download file 


Сначала код импортирует необходимые библиотеки, такие как pandas, sklearn, и MLPClassifier из sklearn.neural_network.

Затем он загружает данные о вине из UCI репозитория с помощью функции pd.read_csv() и записывает названия столбцов в список wine_names. Затем данные разбиваются на X (входные данные) и y (метки классов).

Далее X и y разбиваются на обучающую и тестовую выборки с помощью функции train_test_split().

Затем данные нормализуются с помощью StandardScaler из sklearn.preprocessing. Сначала обучающий набор используется для вычисления среднего и стандартного отклонения каждой функции, а затем данные нормализуются.

Затем создается MLPClassifier с тремя скрытыми слоями, каждый из которых содержит 13 нейронов. Это задается аргументом hidden_layer_sizes.

Затем MLP обучается на обучающих данных с помощью метода fit().

Далее модель используется для предсказания меток классов на тестовом наборе данных, и выводится матрица ошибок с помощью confusion_matrix из sklearn.metrics и отчет классификации с помощью classification_report.

Наконец, код выводит веса между каждыми слоями MLP, используя mlp.coefs_.

Интересная фишка этого кода заключается в том, что он показывает пример использования MLP для многоклассовой классификации, а также демонстрирует, как использовать различные функции из библиотек sklearn для подготовки и обучения модели.
