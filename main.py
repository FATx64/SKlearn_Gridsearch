#SKLearn dengan Teknik Grid Search

import pandas as pd

df = pd.read_csv('Salary_Data.csv')

df.info()
df.head()

import numpy as np
 
# memisahkan atribut dan label
X = df['YearsExperience']
y = df['Salary']
 
# mengubah bentuk atribut
X = X[:,np.newaxis]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
 
# membangun model dengan parameter C, gamma, dan kernel
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.5, 0.05,0.005]
}
grid_search = GridSearchCV(model, parameters)
 
# melatih model dengan fungsi fit
grid_search.fit(X,y)

# menampilkan parameter terbaik dari objek grid_search
print(grid_search.best_params_)

# membuat model SVM baru dengan parameter terbaik hasil grid search
model_baru  = SVR(C=100000, gamma=0.005, kernel='rbf')
model_baru.fit(X,y)

# memvisualisasikan SVR dengan parameter hasil grid search. 
#Dapat dilihat dari hasil plot bahwa grid search berhasil mencari 
#parameter yang lebih baik sehingga meningkatkan performa dari model.
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model_baru.predict(X))
