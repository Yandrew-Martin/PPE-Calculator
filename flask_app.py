from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

#import pandas as pd
#from sklearn.linear_model import LogisticRegression

#def maskinfo():
    #data = pd.read_csv('maskdata.csv', sep=',')
    #return data

#def logRegression():
#    data = maskinfo()
#    reg = LogisticRegression(random_state=9)
#    logisticRegr.fit(x_train,y_train)
#    return reg

#def logRegPrediction(a):
#    reg = logRegression()
#    prediction = reg.predict(a)
#    return prediction

app = Flask(__name__)
app.config["DEBUG"] = True

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="amartin",
    password="cdrh%40fda123",
    hostname="amartin.mysql.pythonanywhere-services.com",
    databasename="amartin$publications",
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Submission(db.Model):
    __tablename__ = "publications"

    author = db.Column(db.String(4096))
    title = db.Column(db.String(4096), primary_key=True)
    journal = db.Column(db.String(4096))
    url = db.Column(db.String(4096))
    submission = db.Column(db.String(4096))
#mydb = mysql.connector.connect(
#  host='amartin.mysql.pythonanywhere-services.com',
#  user='amartin',
#  password='cdrh%40fda123',
#  database='amartin$publications',)

@app.route('/')
def home():
    return render_template('calculator.html')

@app.route('/', methods=['GET','POST'])
def calc():
#    method = request.form.get("method")
#    microbial = request.form.get("microbial")
    if request.method == 'POST':
        mfp = request.form["mfp"]
        thick = request.form["thick"]
        runs = request.form['runs']
        method = request.form.get("method")
        microbial = request.form.get("microbial")
#        if 'submit_button' in request.form:
#            method = request.form["method"]
#            microbial = request.form.get["microbial"]
        result = list((mfp,thick,method,runs,microbial))
        return render_template('calculator.html',result=result)

@app.route('/PPERisk/COU')
def cou():
    return render_template('COU.html')

@app.route('/PPERisk/FAQ')
def faq():
    return render_template('FAQ.html')

@app.route('/PPERisk/Changelog')
def change():
    return render_template('Changelog.html')
#@app.route('/PPERisk/Publications')
#def publications():
#    return render_template('Publications.html')

@app.route('/PPERisk/Publications', methods=['GET','POST'])
def submission():
#    if request.method == 'POST':
#        author = request.form["author"]
#        title = request.form["title"]
#        journal = request.form['journal']
#        url = request.form["url"]
#        date = "test"
#        submission = Submission(author=author, title=title, journal=journal,url=url,date=date)
#        db.session.add(submission)
#        db.session.commit()
    if request.method == "GET":
        return render_template('Publications.html', publications=Submission.query.all())
    pubauthor = request.form["author"]
    pubtitle = request.form["title"]
    pubjournal = request.form['journal']
    puburl = request.form["url"]
    subdate = "test"
    submission = Submission(author=pubauthor, title=pubtitle, journal=pubjournal,url=puburl,submission=subdate)
    db.session.add(submission)
    db.session.commit()
    return redirect(url_for('submission'))
@app.route('/PPERisk/Table')
def table():
    return render_template('Table.html')