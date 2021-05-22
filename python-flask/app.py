# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:11:37 2020

@author: hp
"""
from flask import Flask
import numpy as np
from flask import request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index2():
    return render_template('index2.html')

"""@app.route('/')
def home():
    return render_template('form.html')"""

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        l=[]
        int_features=[]       
        pregnancies=request.json['pregnancies']
        l.append(pregnancies)
        int_features.append(int(pregnancies))
        ##
        glucose=request.json['glucose']
        l.append(glucose)
        if glucose=='low':
            glucose=0
        elif glucose=='medium':
            glucose=1
        elif glucose=='high':
            glucose=2
        int_features.append(int(glucose))
        ##
        bp=request.json['bp']
        l.append(bp)
        if bp=='low':
            bp=0
        elif bp=='medium':
            bp=1
        elif bp=='high':
            bp=2
        
        int_features.append(int(bp))
        
        skinthickness=request.json['skinthickness']
        l.append(skinthickness)
        int_features.append(int(skinthickness))
        
        insulin=request.json['insulin']
        l.append(insulin)
        if insulin=='low':
            insulin=0
        elif insulin=='medium':
            insulin=1
        elif insulin=='high':
            insulin=2
        int_features.append(int(insulin))
        
        bmi=request.json['bmi']
        l.append(bmi)
        if bmi=='low':
            bmi=0
        elif bmi=='medium':
            bmi=1
        elif bmi=='high':
            bmi=2
        int_features.append(int(bmi))
        
        DPF=request.json['DPF']
        l.append(DPF)
        if DPF=='normal':
            DPF=0
        elif DPF=='prediabetes':
            DPF=1
        elif DPF=='diabetes':
            DPF=2
        int_features.append(int(DPF))
        
        age=request.json['age']
        
        l.append(age)
        if age=='young':
            age=0
        elif age=='middle':
            age=1
        elif age=='old':
            age=2
        int_features.append(int(age))
        #print(int_features)
        final_features = [np.array(int_features)]
        #print(final_features)
        prediction = model.predict(final_features)
        #print(prediction)
        output = int(prediction)
        #print(output)
        
        if output==0:
            #return render_template('form1.html',prediction_text='Be Safe! You have tested NEGATIVE for diabetes',res=0)
            return {'message':'Be Safe! You have tested NEGATIVE for diabetes'}
        else:
            #return render_template('form1.html',res=1,prediction_text='We are sorry to say, you have tested POSITIVE for diabetes' )
            return {'message':'We are sorry to say, you have tested POSITIVE for diabetes'}


if __name__=="__main__":
    app.run(debug=True,use_reloader=False)
