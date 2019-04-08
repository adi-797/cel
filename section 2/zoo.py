#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:54:15 2019

@author: devang
"""
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier 



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
y_train=list(y_train[0:])
y_test=list(y_test)
y=list(y)
''' now we will be getting the average value of recall and precison'''
from sklearn.metrics import precision_recall_fscore_support


     
linear_svc = svm.LinearSVC()
linear_svc.fit(x_train,y_train)
y_score =linear_svc.predict(x_test)
score=precision_recall_fscore_support(y_test, y_score) 
''' now we will be predicting the score'''
py.plot(score[0],score[1])
print(score[2])
py.show()

print(linear_svc.score(x_test,y_test))
linear_svc1=svm.SVC(kernel='rbf')
linear_svc1.fit(x_train,y_train)
y_score=linear_svc1.predict(x_test)
score=precision_recall_fscore_support(y_test, y_score) 
''' now we will be predicting the score'''
py.plot(score[0],score[1])
print(score[2])
py.show()


print(linear_svc.score(x_test,y_test))
clf1=svm.SVC(kernel='poly')
clf1.fit(x_train,y_train)
y_score=clf1.predict(x_test)
score=precision_recall_fscore_support(y_test, y_score) 
''' now we will be predicting the score'''
py.plot(score[0],score[1])
print(score[2])
py.show()

print(clf1.score(x_test,y_test))
clf2=svm.SVC(kernel='sigmoid')
clf2.fit(x_train,y_train)
y_score=clf2.predict(x_test)
score=precision_recall_fscore_support(y_test, y_score) 
''' now we will be predicting the score'''
py.plot(score[0],score[1])
print(score[2])
py.show()

print(clf2.score(x_test,y_test))
    
