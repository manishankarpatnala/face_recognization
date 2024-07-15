import cv2
import cvlib as cv
import numpy as np
import easyocr
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import os
import shutil
import face_recognition
import dlib
from pymongo import MongoClient
import gridfs
from mtcnn.mtcnn import MTCNN
import zipfile
from scraper import search_and_download
import time
import imageio.v3 as iio
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gc
import torch
import glob
from datetime import datetime
import bcrypt
import requests
from imutils import paths, resize
from transformers import pipeline,AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
gc.collect() 
torch.cuda.empty_cache()
import json
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Users\venkatesh.ch\Tesseract-OCR\tesseract.exe"
from flask_mail import Mail, Message
from bson.objectid import ObjectId
from flask import session, request


app = Flask(__name__)

# Secret key for sessions encryption
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dlib.DLIB_USE_CUDA = True

client = MongoClient("mongodb://localhost:27017")
url_db = client['Raw_images_URL_DB']
zipfile__db = client['Raw_images_ZipFile_DB']
url_db_fs = gridfs.GridFS(url_db)
zipfile_db_fs = gridfs.GridFS(zipfile__db)
output_image_db = client["Predicted_Images_DataBase"]
predicted_fs = gridfs.GridFS(output_image_db)
db = client.flaskdatabase
usersdata = db.flaskcollection
dashboard_data = db.dashboard_data
# dashboard_data_temp = db.dashboard_data_temp


mail = Mail(app) # instantiate the mail class
   
# configuration of mail
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'connect@whiteitc.in'
app.config['MAIL_PASSWORD'] = 'ckgblrkqvkhlbioz'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# @app.route('/')
# def home():
#     return render_template("index.html", title="Image Reader")


@app.route('/')
def signin():
    return render_template('sign-in.html', title="Image Reader")


@app.route('/signin')
def signin_page():
    print(session)
    if "email" in session:
        email = session["email"]
        return redirect(url_for("uploadimages"))
    else:
        return render_template('sign-in.html', title="Image Reader")


@app.route("/postsignin", methods=['post', 'get'])
def postsignin():
    message = 'Please login to your account'
    # print(session)
    if "email" in session:
        email = session["email"]
        return redirect(url_for("uploadimages"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        email_found = usersdata.find_one({"email": email})
        if email_found:
            
            email_val = email_found['email']
            passwordcheck = email_found['password']
            
            if str(passwordcheck) == str(password):
                session["email"] = email_val
                return redirect(url_for('uploadimages'))
            else:
                if "email" in session:
                    return redirect(url_for("uploadimages.html"))
                message = 'Wrong password'
                return render_template('sign-in.html', message=message)
        else:
            message = 'Email not found'
            return render_template('sign-in.html', message=message)
    return render_template('sign-in.html', message=message, title="Image Reader")


@app.route('/signup')
def signup():
    return render_template('sign-up.html')


@app.route("/postsignup", methods=['post', 'get'])
def postsignup():
    message = ''
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
            # hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': password1}
            usersdata.insert_one(user_input)
            
            user_data = usersdata.find_one({"email": email})
            new_email = user_data['email']
            message="Account Signed-UP Successfully"
            return render_template('sign-in.html', email=new_email, message=message)
    return render_template('sign-in.html')


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return redirect(url_for("signin"))
    else:
        return render_template('uploadimages.html')

@app.route('/forgotpassword', methods=['post', 'get'])
def forgotpassword():
    return render_template('forgotpassword.html')

@app.route('/changepassword', methods=['post', 'get'])
def changepassword():
    if request.method == "POST":
        
        email = request.form.get("email")

        email_found = usersdata.find_one({"email": email})
        if email_found:
            msg = Message('Password Reset Requested', sender = 'connect@whiteitc.in', recipients = [email])
            msg.body = "By clicking on link You can change your password"  " http://127.0.0.1:5000/passwordset/" + str(email_found['_id'])
            mail.send(msg)
            # return "Sent"
            return render_template('emailsent.html')
        else:
            message="Enter Valid Email Address"
            return render_template('forgotpassword.html', message=message)


@app.route('/mailsent')
def mailsent():
    return render_template('emailsent.html')

@app.route('/passwordset/<user_id>')
def passwordset(user_id):
    # print(user_id) 
    session['temp_user'] = user_id
    return render_template('passwordset.html')

@app.route('/changepass', methods=['post', 'get'])
def changepass():
    value = session['temp_user']
    print(value)

    if request.method == "POST":
        password1 = request.form.get("ps1")
        password2 = request.form.get("ps2")
        
        # temp_name = request.session['temp_user'] 
        # print(temp_name)
        print(password1)
        print(password2)
        data = ObjectId(str(value))
        user_details = usersdata.find_one({"_id": ObjectId(data)})
        print(user_details)
        
        if password1 == password2:
                usersdata.update_one({"_id": ObjectId(data)},{"$set":{"password":str(password1)}})
                print(user_details)
                return render_template("reset_completed.html")
        else:
            # print("123")
            message="Both passwords should match"
            return render_template("passwordset.html", {"msg":message})
    else:
        pass
    return render_template('passwordset.html')


@app.route('/uploadimages')
def uploadimages():
    try:
        shutil.rmtree(r"images")
        # print("Images folder deleted") 
    except:
        pass
    try:
        shutil.rmtree(r"static\Output")
    except:
        pass
    
    if (os.path.exists('Output.zip')):
        os.remove('Output.zip')
        # print("File Removed")
    else:
        pass
    # print("Output.zip file deleted")
    try:
        if "email" in session:
            email = session["email"]
            return render_template("uploadimages.html", email=email)
        else:
            return redirect("signin")
    except:
        pass
    return render_template("uploadimages.html", title="Image Reader", email=email)


@app.route('/extractinfo')
def extractinfo():
    try:
        email = session["email"]
        if not os.path.exists(os.path.dirname(__file__) + r"\images"):
            print("Images Folder Not Available")
            message = "Please Upload Images"
            return render_template("uploadimages.html", message=message, email=email)
        elif os.path.exists(r"static\Output\images"): 
            gallery_images = glob.glob(r'static\Output\images\*')
            return render_template("facedetection.html", filename=gallery_images, email=email) 
        else:
            print("Images Folder Available")
            return render_template("image_loading.html", title="Image Reader", email=email) 
                
    except:
        return redirect("signin")

@app.route('/textinfo')
def textinfo():
    if "email" in session:
        email = session["email"]
        pass
    else:
        return redirect("signin")
    if os.path.exists(os.path.dirname(__file__) + r"\static\Output\Extracted_text.txt"):
        with open(os.path.dirname(__file__) + r"\static\Output\Extracted_text.txt", encoding="utf8") as f:
            text_file = f.read().replace('\n', '<br>')
        return render_template("textfile.html", data=text_file, email=email)
    else:
        return redirect("uploadimages")


@app.route('/sentimentanalysis')
def sentimentanalysis():
    email = session["email"]
    if os.path.exists(os.path.dirname(__file__) + r"\static\Output\images"):
        source_data = dashboard_data.find({}, {'_id': False})
        return render_template("sentiment_analysis.html", data=source_data, email = email)
    else:
        return redirect("uploadimages")


class MITE:

    # def __init__(self):
    #     pass

    def bb_to_rect(bb):
        try:
            top=bb[1]
            left=bb[0]
            right=bb[0]+bb[2]
            bottom=bb[1]+bb[3]
            return (top, right, bottom, left)
        except Exception as bbe:
            # print(bbe)
            return bb

    def opencv_write(im_data, name, top, right, bottom, left):
        try:
            color = list(np.random.random(size=3) * 256)
            cv2.rectangle(im_data, (left, top), (right, bottom), color, 2)
            cv2.rectangle(im_data, (left, bottom + 15), (right + len(name), bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im_data, name, (left + 1, bottom + 10), font, 0.6, (255, 225, 255), 2)
            return im_data
        except Exception as img_wr_er:
            # print(img_wr_er)
            return im_data

    def img_gender(image, x, y, x2, y2):
        try:
            cv2.rectangle(image, (y2, x), (y, x2), (0, 255, 0), 2)
            crop = np.copy(image[x:x2, y2:y])
            (label, confidence) = cv.detect_gender(crop)
            idx = np.argmax(confidence)
            label = label[idx]
            label = "unknown_{0}".format(label)
            cv2.putText(image, label, (y2+1, x2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            return image
        except Exception as img_gen_er:
            # print(img_gen_er)
            return image

    def no_output_detected(save_folder_path, file_name, output_image_db, name, output_text_dict, predicted_fs, model):
        try:
            file_path = os.path.join(save_folder_path, "{2}_no_output_detected_{0}.{1}".format(os.path.split(file_name)[1].split(".")[-2], file_name.split(".")[-1], model))
            shutil.copy(file_name, file_path)
            db_collection = output_image_db[name]
            image = cv2.imread(file_path)
            imageString = image.tostring()
            imageID = predicted_fs.put(imageString, encoding='utf-8')
            # create our image meta data
            meta = {
                'name': "{2}_no_output_detected_{0}.{1}".format(os.path.split(file_name)[1].split(".")[-2], file_name.split(".")[-1], model),
                'images': [
                    {
                        'imageID': imageID,
                        'shape': image.shape,
                        'dtype': str(image.dtype)
                    }
                ],
                'Text': output_text_dict[file_name]
            }
            db_collection.insert_one(meta)
            return True
        except Exception as err:
            # print(err)
            return False

    def face_detection_ml(file_name, small_frame, loc, known_face_encodings, known_face_names, file_path, output_image_db, predicted_fs, output_text_dict, model_dict, model):
        try:
            start = time.time()
            face_encodings = face_recognition.face_encodings(small_frame, loc)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings,
                                                         face_encoding)  # , tolerance=0.95)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                output = {}
                for (top, right, bottom, left), name in zip(loc, face_names):
                    output["{}_{}".format(name, top)] = [top, right, bottom, left]

                db_collection = output_image_db[name]
                image = cv2.imread(file_path)
                imageString = image.tostring()
                imageID = predicted_fs.put(imageString, encoding='utf-8')
                # create our image meta data
                meta = {
                    'name': "{3}_{0}_{1}.{2}".format(
                        os.path.split(file_name)[1].split(".")[-2], name, file_name.split(".")[-1], model),
                    'images': [
                        {
                            'imageID': imageID,
                            'shape': image.shape,
                            'dtype': str(image.dtype)
                        }
                    ],
                    'Text': output_text_dict[file_name]
                }
                db_collection.insert_one(meta)
            model_dict[model] = output
            # # print("--- %s {model} Model seconds ---" % (time.time() - start))
            return True
        except Exception as detect_err:
            # print(detect_err)
            return False


@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    email = session["email"]
    try:
        if os.path.exists(os.path.dirname(__file__) + r"images"):
            return redirect("uploadimages")
        else:
            pass
    except:
        pass

    save_folder_path = r"static\Output"
    images_directory = r"images"
    try:
        if (os.path.exists('Output.zip')):
            os.remove('Output.zip')
            # print("File Removed")
        else:
            pass
        # print("Output.zip file deleted")
        try:
            shutil.rmtree(save_folder_path)
        except OSError:
            os.remove(save_folder_path)

        # print("Output folder images deleted")
        try:
            shutil.rmtree(images_directory)
            # print("Images folder deleted") 
        except OSError:
            os.remove(images_directory)
            # print("Images folder not deleted") 

        # try:
        #     shutil.rmtree(zip_directory)
        # except OSError:
        #     os.remove(zip_directory)
        # print("Zip Files folder images deleted") 
    except:
        pass

    try:
        if request.method == 'POST':
            image_data = request.files['file']
            url_data = request.form['text']

            if request.form['text'] != '':
                if not os.path.exists(r"images\url_images"):
                    os.makedirs(r"images\url_images")
                search_and_download(search_term=url_data, number_images=2)
                file_names_folder =  os.path.join(os.path.dirname(__file__), r"images\url_images")
                # print(file_names_folder)
                file_names = []
                for file_name in os.listdir(file_names_folder):
                    file_path = os.path.join(file_names_folder, file_name)
                    file_names.append(file_path)
                    with open(file_path, 'rb') as url_file_data:
                        url_db_fs.put(url_file_data, filename=file_name)

                file_length = len(file_names)
                input_file_name = url_data
                # print(input_file_name)
                
            elif request.files['file'].filename != '':
                if not os.path.exists(r"images"):
                    os.mkdir(r"images")
                zipfile_db_fs.put(image_data, filename=image_data.filename)
                with zipfile.ZipFile(image_data, 'r') as zip_ref:
                    zip_ref.extractall(r"images")
                file_names_folder =  os.path.join(os.path.dirname(__file__), "images")
                # print(file_names_folder)
                file_names = []
                for folder in os.listdir(file_names_folder):
                    for file_name in os.listdir(os.path.join(file_names_folder, folder)):
                        file_path = os.path.join(file_names_folder, folder, file_name)
                        file_names.append(file_path)
                        with open(file_path, 'rb') as zip_file_data:
                            zipfile_db_fs.put(zip_file_data, filename=file_name)
                file_length = len(file_names)
                input_file_name = image_data.filename
                # print(input_file_name)

            else:
                message = "Please Upload Images"
                return render_template("uploadimages.html", message=message, email=email)
            
            data = file_length
            input_file = input_file_name
            # print(input_file)
            return render_template("uploadimages.html", data=file_length, input_file = input_file_name, email=email)
    except Exception as e:
        # print(e)
        return render_template("uploadimages.html", email=email)


@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    save_folder_path = r"static\Output\images"
    try:
        if "email" in session:
            pass
        else:
            return redirect("signin")
    except:
        pass
    try:
        if os.path.exists(save_folder_path):
            return render_template("uploadimages.html", title="Image Reader")
        else:
            pass
    except:
        pass

    try:
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        file_dict = dict()
        start_time = time.time()
        known_face_encodings = list()
        known_face_names = list()
        for image_file in os.listdir(r"Face_encoding_data"):
            train_image = face_recognition.load_image_file(os.path.join(r"Face_encoding_data", image_file))
            known_face_encodings.append(face_recognition.face_encodings(train_image)[0])
            known_face_names.append(image_file.split(".")[-2])

        
        output_text_dict = {}
        output_text_dict_1 = {}
        reader = easyocr.Reader(['ta', 'en'], gpu=True) 
        

        file_names_folder =  os.path.join(os.path.dirname(__file__), r"images")
        file_names = []
        for folder in os.listdir(file_names_folder):
            for file_name in os.listdir(os.path.join(file_names_folder, folder)):
                file_path = os.path.join(file_names_folder, folder, file_name)
                file_names.append(file_path)
        detector = MTCNN()
        cnn_face_detector = dlib.cnn_face_detection_model_v1(r"CNN_face_detector\mmod_human_face_detector.dat")
        size = 512
        # print(file_names)
        for file_name in file_names:
            model_dict = {}
            text_time = time.time()
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                bounds = reader.readtext(iio.imread(file_name))
                temp = ''
                text_eval = 0
                for ind in range(len(bounds)):
                    temp = temp + ' ' + bounds[ind][1]
                    text_eval = float(text_eval) + float(bounds[ind][2])
                output_text_dict[file_name] = temp
                file_path_temp, text_file_name = os.path.split(file_name)

                response=requests.post('http://127.0.0.1:8000/batch_translate',json=({"text": temp, "language": "ta", "auth_token": "2fe23e072a1fc945891778f78acad05b"}))

                text = pytesseract.image_to_string(file_name, lang="eng")
                for index in range(len(text)):
                    temp_1 = temp + ' ' + str(index)
                    # text_eval = float(text_eval) + float(bounds[ind][2])
                print(temp_1)
                output_text_dict_1[file_name] = temp_1
                response_1=requests.post('http://127.0.0.1:8000/batch_translate',json=( {
                "text": temp_1,
                "language": "ta",
                "auth_token": "2fe23e072a1fc945891778f78acad05b"
                })
                )
                if float(sentiment_pipeline(output_text_dict[file_name])[0]["score"]) > float(sentiment_pipeline(output_text_dict_1[file_name])[0]["score"]):
                    with open(r'static\Output\Extracted_text.txt', 'a', encoding="utf-8") as f:
                        f.write(f"Source : {text_file_name}")
                        f.write('\n')
                        f.write(f"Text_EasyOCR : {output_text_dict[file_name]}")
                        f.write('\n')
                        f.write(f"Translated_text_EasyOCR : {response.json()['translation']}")
                        f.write('\n')
                        if len(output_text_dict[file_name]) > 1:
                            f.write(f"Sentiment_Analysis_easyocr : {sentiment_pipeline(temp)}")
                        else:
                            f.write(f"Sentiment_Analysis_easyocr : '' ")
                        f.write('\n')
                        if temp != '':
                            f.write(f"Text_Evaluation usig EasyOCR : {text_eval/len(bounds)}")
                        else:
                            f.write(f"Text_Evaluation usig EasyOCR : '' ")
                        f.write('\n')
                        f.write('\n')
                else:
                    with open(r'static\Output\Extracted_text.txt', 'a', encoding="utf-8") as f:
                        f.write(f"Source : {text_file_name}")
                        f.write('\n')
                        f.write(f"Text_Tessaract : {temp_1}")
                        f.write('\n')
                        f.write(f"Translated_text_Tessaract : {response_1.json()['translation']}")
                        f.write('\n')
                        if len(output_text_dict_1[file_name]) > 1:
                            f.write(f"Sentiment_Analysis_Tessaract : {sentiment_pipeline(temp_1)}")
                        else:
                            f.write(f"Sentiment_Analysis_Tessaract : '' ")
                        f.write('\n')
                        # if temp_1 != '':
                        #     f.write(f"Text_Evaluation usig Tessaract : {text_eval/len(temp_1)}")
                        # else:
                        #     f.write(f"Text_Evaluation usig Tessaract : '' ")
                        # f.write('\n')
                        f.write('\n')

                if len(output_text_dict[file_name]) > 1:
                    dashboard_datas = {"Source" : str(text_file_name), "Sentiment_Analysis" : str(sentiment_pipeline(output_text_dict[file_name])[0]["label"])}
                    dashboard_data.insert_one(dashboard_datas)
                else:
                    dashboard_datas = {"Source" : str(text_file_name), "Sentiment_Analysis" : "None"}
                    dashboard_data.insert_one(dashboard_datas)

                # with open(r'static\Output\Dashboard.txt', 'a', encoding="utf-8") as f:
                #     f.write("{Source : "+str(text_file_name)+"}")
                #     f.write('\n')
                #     if len(output_text_dict[file_name]) > 1:
                #         f.write("{Sentiment_Analysis : "+str(sentiment_pipeline(output_text_dict[file_name])[0]["label"])+"}")
                #     else:
                #         f.write("{Sentiment_Analysis : None}")
                #     f.write('\n')
                # print("--- Text Time : %s seconds ---" % (time.time() - text_time))
                
                small_frame = cv2.imread(file_name)
                name = "Unknown"

                ##DLIB
                img = dlib.load_rgb_image(file_name)
                dets = cnn_face_detector(img, 1)
                dlib_loc =[]
                for d in dets:
                    dlib_loc.append((d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()))

                if len(dlib_loc) == 0:
                    MITE.no_output_detected(save_folder_path, file_name, output_image_db, name, output_text_dict, predicted_fs, "Dlib")

                else:
                    MITE.face_detection_ml(file_name, small_frame, dlib_loc, known_face_encodings, known_face_names, file_path, output_image_db, predicted_fs, output_text_dict, model_dict, "Dlib")

                ##MTCNN
                name = "No_output_detected"
                faces = detector.detect_faces(small_frame)
                mtcnn_loc = []
                for face in faces:
                    face_loc = MITE.bb_to_rect(face['box'])
                    mtcnn_loc.append(face_loc)

                if len(mtcnn_loc) == 0:
                    MITE.no_output_detected(save_folder_path, file_name, output_image_db, name, output_text_dict,
                                        predicted_fs, "MTCNN")

                else:
                    MITE.face_detection_ml(file_name, small_frame, mtcnn_loc, known_face_encodings, known_face_names,
                                        file_path, output_image_db, predicted_fs, output_text_dict, model_dict, "MTCNN")

                ##FACE_RECOGNITION
                face_locations = face_recognition.face_locations(small_frame)

                if len(face_locations) == 0:
                    MITE.no_output_detected(save_folder_path, file_name, output_image_db, name, output_text_dict,
                                        predicted_fs, "Face_recognition")

                else:
                    MITE.face_detection_ml(file_name, small_frame, face_locations, known_face_encodings, known_face_names,
                                        file_path, output_image_db, predicted_fs, output_text_dict, model_dict, "Face_recognition")
                
            file_dict[file_name] = model_dict

        for f_name in file_dict.keys():
            for mod in file_dict[f_name].keys():
                im_data = cv2.imread(f_name)
                for cls_name in file_dict[f_name][mod].keys():
                    if "unknown" in cls_name.lower():
                        out_im_data = MITE.img_gender(im_data, file_dict[f_name][mod][cls_name][0], file_dict[f_name][mod][cls_name][1], file_dict[f_name][mod][cls_name][2], file_dict[f_name][mod][cls_name][3])
                    else:
                        out_im_data = MITE.opencv_write(im_data, cls_name, file_dict[f_name][mod][cls_name][0], file_dict[f_name][mod][cls_name][1], file_dict[f_name][mod][cls_name][2], file_dict[f_name][mod][cls_name][3])
                file_path = os.path.join(save_folder_path, "{2}_{0}.{1}".format(os.path.split(f_name)[1].split(".")[-2], f_name.split(".")[-1], mod))
                cv2.imwrite(file_path, out_im_data)
                    
    except Exception as e:
        print(e)

    gc.collect()
    torch.cuda.empty_cache()

    # gallery_images = glob.glob(r'static\Output\images\*')
    # print("--- Text Time : %s seconds ---" % (time.time() - text_time))
    return redirect("extractinfo")


@app.route('/download', methods=['GET', 'POST'])
def download():

    try:
        if "email" in session:
            email = session["email"]
            pass
        else:
            return redirect("signin")
    except:
        pass

    client = MongoClient("mongodb://localhost:27017")
    output_db = client['OutputDB']
    output_db_fs = gridfs.GridFS(output_db)
    import os, zipfile

    folder_path = r"static\Output"
    folder, name = os.path.split(folder_path)
    zip_name = name + '.zip'

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, folder_path))

    zip_ref.close()
    with open(os.path.dirname(__file__) + r"\Output.zip", 'rb') as output_file:
        output_db_fs.put(output_file, filename=zip_name)
    path = send_file(os.path.dirname(__file__) + r"\Output.zip",  as_attachment=True)
    return path


if __name__ == '__main__':
    app.run(debug=True)