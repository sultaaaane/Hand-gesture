import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt

data_dict = pickle.load(open('./data.pickle', 'rb'))

data_dict['data'] = np.array(data_dict['data'])
data_dict['labels'] = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data_dict['data'], data_dict['labels'], test_size=0.2,shuffle=True,stratify=data_dict['labels'])

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)

print('Accuracy: {}'.format(score))

f = open('model.pickle', 'wb')
pickle.dump(model, f)
f.close()