from flask import Flask, render_template, url_for, request, redirect, json
import os
from datetime import datetime
from main import inferweb
from flask_mysqldb import MySQL
#from Model import Model, DecoderType

#allowed=['jpg','jpeg','png','pdf']

app = Flask(__name__)

app.config["IMAGE_UPLOADS"]="../src/static/images/"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'DigiForm'
app.config['MYSQL_PASSWORD'] = 'group1'
app.config['MYSQL_DB'] = 'DigiForm'

mysql = MySQL(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            print(image)
            ext = os.path.splitext(image.filename)[1]
            fn=image.filename
            # print(ext)
            # xt.lower()
            # print(ext)
            # newFileName = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "."+ext
            print(fn)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], fn))
            ipath='../src/static/images/%s'%fn
            recognized = inferweb(ipath)
            #recognized.extend(('test1','test2','test3','test4'))
            return render_template('fill.html',recognized=json.dumps(recognized),name='images/%s'%fn)

@app.route("/success",methods=["GET","POST"])
def dataEntry():
    if request.method == "POST":
        details = request.form
        firstName = details['fname']
        MiddleName = details['mname']
        lastName = details['lname']
        year = details['year']
        dept = details['dept']
        roll = details['roll']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO Data(firstName, MiddleName, lastName, Year, Department, RollNo) VALUES (%s, %s, %s, %s, %s, %s)", (firstName, MiddleName, lastName, year, dept, roll))
        mysql.connection.commit()
        cur.close()
        return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)
