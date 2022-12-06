from flask import Flask, render_template, session, request, redirect
import os
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = APP_ROOT + "/static"
app = Flask(__name__)
import pickle
import numpy as np
import pandas as pd
lr = pickle.load(open("Log_reg2.pkl",'rb'))
pca = pickle.load(open("pca.pkl",'rb'))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=['post'])
def upload():
    try:
        row_index = request.form.get('row_index')
        row_index = int(row_index)
        file = request.files.get('file')
        upload_path = APP_ROOT+"/"+str(file.filename)
        file.save(upload_path)
        dataframe = pd.read_csv('static/'+str(file.filename))
        dataframe.drop(['Activity', 'subject'], axis=1, inplace=True)

        value = dataframe.iloc[[row_index]].values
        print(value)
        print(value.shape)
        my_pca = pca.transform(value)
        result = lr.predict(my_pca)

        if result == 0:
            msg = 'LAYING'
            print('LAYING ')
        elif result == 1:
            msg = 'SITTING'
            print('SITTING')
        elif result == 2:
            msg = 'STANDING'
            print('STANDING ')
        elif result == 3:
            msg = 'WALKING'
            print('WALKING')
        elif result == 4:
            msg = 'WALKING DOWNSTAIRS'
            print('WALKING_DOWNSTAIRS')
        elif result == 5:
            msg = 'WALKING_UPSTAIRS'
            print('WALKING_UPSTAIRS')
        else:
            msg = 'Invalid input'
            print('Invalid input')

        return render_template("msg.html", msg=msg)
    except:
        return render_template("msg.html", msg='Something went wrong')

app.run(debug=True)
