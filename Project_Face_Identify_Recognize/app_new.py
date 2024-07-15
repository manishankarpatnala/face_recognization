from flask import Flask, render_template, url_for, redirect, request, session
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
import bcrypt
from functools import wraps
import os

# from werkzeug import secure_filename



app = Flask(__name__)
app.config['SECRET_KEY'] = 'testing'
# app.SECRET_KEY = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'

app.config["UPLOAD_FOLDER"] = "static/sample"
app.config["UPLOAD_FOLDER"] = "static/example"


client = pymongo.MongoClient('localhost', 27017)
db = client.flaskdatabase
usersdata = db.flaskcollection



# client = MongoClient('mongodb://localhost:27017/')
# db = client["firstdatabase"]

# collection = db["firstcollection"]


# user_details={ "name": "sym", "email": "sym@gmail.com", "password": 123456}
# usersdata.insert_one(user_details)


@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/index')
def index():
   
    if "email" in session:
        email = session["email"]
    return render_template('index.html', email=email)


@app.route('/signin')
def signin():
    return render_template('sign-in.html')

@app.route("/postsignin", methods=['post', 'get'])
def postsignin():
    message = 'Please login to your account'
    # print(session)
    if "email" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        email_found = usersdata.find_one({"email": email})
        if email_found:
            
            email_val = email_found['email']
            passwordcheck = email_found['password']
            
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('index'))
            else:
                if "email" in session:
                    
                    return redirect(url_for("index.html"))
                message = 'Wrong password'
                return render_template('sign-in.html', message=message)
        else:
            message = 'Email not found'
            return render_template('sign-in.html', message=message)
    return render_template('sign-in.html', message=message)




@app.route('/signup')
def signup():
    return render_template('sign-up.html')

@app.route("/postsignup", methods=['post', 'get'])
def postsignup():

    client = pymongo.MongoClient('localhost', 27017)
    db = client.flaskdatabase
    usersdata = db.flaskcollection
    message = ''

    # if "email" in session:
    #     return redirect(url_for("signin"))

    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        

        user_found = usersdata.find_one({"name": user})
        email_found = usersdata.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('sign-in.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('sign-in.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('sign-in.html', message=message)
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed}
            usersdata.insert_one(user_input)
            
            user_data = usersdata.find_one({"email": email})
            new_email = user_data['email']
   
            return render_template('sign-in.html', email=new_email)
    return render_template('sign-in.html')


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("sign-in.html")
    else:
        return render_template('uploadimages.html')

@app.route("/post_forgot_password", methods=["POST", "GET"])
def post_forgot_password(request):

    try:
        if request.method == "POST":
            email = request.POST.get("email")
            authe.send_password_reset_email(email)
            print("try success")
            return render(request, "post_forgot_password.html")
   
    except:
        message = "Invalid Email! Please re-check your Email-Id"
        return render(
            request, "forgot_password.html", {"msg": message}
        )

    print("try failed")
    return render(request, "forgot_password.html")



@app.route('/uploadimages')
def uploadimages():
    return render_template('uploadimages.html')

# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       return 'file uploaded successfully'


# @app.route('/display', methods = ['GET', 'POST'])
# def display_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         filename = secure_filename(f.filename)

#         f.save(app.config['UPLOAD_FOLDER'] + filename)

#         file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
#         content = file.read()   
        
#     return render_template('uploadimages.html', content=content ) 



@app.route('/extractinfo')
def extractinfo():
    return render_template('extractinfo.html')

@app.route('/facedetection')
def facedetection():
    return render_template('facedetection.html')
    
@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/content')

def content():
    # filename = "C:/Users/dell/flask/flaskdemo/projects/demo/static/sample.txt"
    # with open(filename) as diary_file:
    #     n = 1
    #     for data in diary_file:
    #             print(data)
    #             n += 1
    # return render_template('textfile.html', data=data)
    with open('C:/Users/dell/flask/flaskdemo/projects/demo/static/sample.txt', 'r') as f:
        data = f.read().replace('\n','<br>')
        print(data)
    return render_template('textfile.html', data=data)


    # print("name_file")
    # name_file = open('static/example.doc', 'r+')
    # print(name_file)
    # with open('static/sample.txt', 'r') as f:
    #     data = f.readlines()
    #     print(data)
    # return render_template('textfile.html', data=data)
    
    # with open('the-zen-of-python.txt') as f:
    
    # contents = f.read()
    # print(contents)
    # return render_template('textfile.html')


	# with open('sample.txt', 'r') as f:
    #     content = f.read()
	# return render_template('textfile.html') 



if __name__ == "__main__":
    app.run(debug=True)


