import pandas as pd 
import numpy as np 
import tensorflow as tf  
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as py
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier 
from keras import backend as K


raw_data=pd.read_csv("zoo_data.csv")
zoo_1=['aardvark', 'antelope','bear','boar','buffalo','calf',
                  'cavy','cheetah','deer','dolphin','elephant',
                  'fruitbat','giraffe','girl','goat','gorilla','hamster',
                  'hare', 'leopard', 'lion','lynx', 'mink', 'mole', 'mongoose',
                  'opossum','oryx','platypus', 'polecat', 'pony',
                  'porpoise','puma','pussycat','raccoon','reindeer',
                  'seal', 'sealion', 'squirrel','vampire','vole','wallaby','wolf']
zoo_2=['chicken', 'crow', 'dove', 'duck', 'flamingo', 'gull', 'hawk',
                  'kiwi','lark','ostrich','parakeet','penguin','pheasant',
                  'rhea','skimmer','skua','sparrow','swan','vulture','wren']
zoo_3=['pitviper', 'seasnake', 'slowworm', 'tortoise', 'tuatara']
zoo_4=['bass','carp','catfish','chub','dogfish','haddock',
                  'herring','pike', 'piranha','seahorse','sole','stingray','tuna']
zoo_5=['frog','frog','newt','toad']
zoo_6=['flea','gnat','honeybee','housefly','ladybird','moth','termite','wasp']
zoo_7=['clam','crab','crayfish','lobster','octopus',
                  'scorpion','seawasp','slug','starfish','worm']

output=raw_data['ANIMAL']
y=np.array(output)

train_x=pd.read_csv('x_train.csv')
x=np.array(train_x)

i=0
while i<101:
    if y[i] in zoo_1:
        y[i]=1
    if y[i] in zoo_2:
        y[i]=2
    if y[i] in zoo_3:
        y[i]=3
    if y[i] in zoo_4:
        y[i]=4
    if y[i] in zoo_5:
        y[i]=5
    if y[i] in zoo_6:
        y[i]=6
    if y[i] in zoo_7:
        y[i]=7    
           
    i=i+1
i=0

   
''' now we will be applying the K fold methods '''
kf = KFold(n_splits=5)
kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
'''def precision(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 '''   
model = Sequential()
model.add(Dense(5, input_dim=17, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='softmax'))   
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10) 
scores = model.evaluate(x_train, y_train)
print(model.metrics_names[1], scores[1]*100)
score=model.predict(x_test)
print(score)
score=score.round()
print(scores)
y_target=list(y_test)
from mlxtend.evaluate import confusion_matrix



cm = confusion_matrix(y_target=[7, 4, 2, 1, 7, 4, 2, 6, 5, 3, 3, 4, 1, 1, 2, 1, 6, 1, 7, 2], 
                      y_predicted=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                      binary=False)

'''y_actu = pd.Series([7, 4, 2, 1, 7, 4, 2, 6, 5, 3, 3, 4, 1, 1, 2, 1, 6, 1, 7, 2],name='Actual')
y_pred = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
'''
print(cm)
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()