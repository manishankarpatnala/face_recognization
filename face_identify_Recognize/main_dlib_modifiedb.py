import datetime
from distutils import extension
from fileinput import filename
from tkinter import E
import cv2
import numpy as np
import easyocr
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import os
import shutil
import urllib.request
import face_recognition
import dlib
from pymongo import MongoClient
from io import BytesIO
import gridfs
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import zipfile
from scraper import search_and_download

app = Flask(__name__)

# Secret key for sessions encryption
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@app.route('/')
def home():
    return render_template("index.html", title="Image Reader")

def bb_to_rect(bb):
    top=bb[1]
    left=bb[0]
    right=bb[0]+bb[2]
    bottom=bb[1]+bb[3]
    return (top, right, bottom, left)
    
@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    save_folder_path = r"Output"
    try:
        if (os.path.exists('Output.zip')):
            os.remove('Output.zip')
            print("File Removed")
        else:
            pass
        try:
            directory = r"images"
            for predict_file_name in os.listdir(directory):
                os.remove(os.path.join(directory, predict_file_name))
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
        except:
            pass
        for folder in os.listdir(save_folder_path):
            path = os.path.join(save_folder_path, folder)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    except:
        pass

    client = MongoClient("mongodb://localhost:27017")
    url_db = client['Raw_images_URL_DB']
    zipfile__db = client['Raw_images_ZipFile_DB']
    url_db_fs = gridfs.GridFS(url_db)
    zipfile_db_fs = gridfs.GridFS(zipfile__db)
    print(os.path.dirname(__file__))

    if request.method == 'POST':
        try:
            start_time = datetime.datetime.now()
            image_data = request.files['file']
            url_data = request.form['text']

            known_face_encodings = list()
            known_face_names = list()

            for image_file in os.listdir(r"Face_encoding_data"):
                train_image = face_recognition.load_image_file(os.path.join(r"Face_encoding_data", image_file))
                known_face_encodings.append(face_recognition.face_encodings(train_image)[0])
                known_face_names.append(image_file[:-4])
            output_text_dict = {}
            reader = easyocr.Reader(['ta', 'en'])

            if len(url_data) != 0:
                search_and_download(search_term=url_data, number_images=10)
                file_names_folder =  os.path.join(os.path.dirname(__file__), "images")
                print(file_names_folder)
                file_names = []
                for file_name in os.listdir(file_names_folder):
                    file_path = os.path.join(file_names_folder, file_name)
                    file_names.append(file_path)
                    with open(file_path, 'rb') as url_file_data:
                        url_db_fs.put(url_file_data, filename=file_name)
                    # url_db_fs.put('r"{}"'.format(file_path), filename=file_name)
                print(file_names)
                # for file in file_names:
                #     file1, file2 = os.path.split(file)
                #     url_db_fs.put(file, filename=file2)
                
            else:
                zipfile_db_fs.put(image_data, filename=image_data.filename)
                file_like_object = image_data.stream._file  
                zipfile_ob = zipfile.ZipFile(file_like_object)
                file_names = zipfile_ob.namelist()
                print(file_names)   

            client = MongoClient("mongodb://localhost:27017")
            output_image_db = client["Predicted_Images_DataBase"]
            predicted_fs = gridfs.GridFS(output_image_db)


            for file_name in file_names:
                image_extension = file_name.split(".")[-1]
                f1, f2 = os.path.split(file_name)
                file_name_split = f2.split(".")[-2]
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    bounds = reader.readtext(file_name)
                    temp = ''
                    for ind in range(len(bounds)):
                        temp = temp + bounds[ind][1]
                    output_text_dict[file_name] = temp

                    with open(r'Output\Extracted_text.txt', 'a', encoding="utf-8") as f:
                        f.write(f"Source : {file_name}")
                        f.write('\n')
                        f.write(f"Text : {output_text_dict[file_name]}")
                        f.write('\n')
                        f.write('\n')

                    small_frame = cv2.imread(file_name)
                    # mtcnn_img = plt.imread(file_name)

                    try:
                        detector = MTCNN()
                        faces = detector.detect_faces(small_frame)
                        mtcnn_loc = []
                        for face in faces:
                            face_loc = bb_to_rect(face['box'])
                            mtcnn_loc.append(face_loc)

                        if not os.path.exists(os.path.join(save_folder_path, file_name_split)):
                            os.mkdir(os.path.join(save_folder_path, file_name_split))

                        if len(mtcnn_loc) == 0:
                            file_path = os.path.join(save_folder_path, "{0}\MTCNN_no_output_detected_{1}.{2}".format(
                                file_name_split, file_name_split, image_extension))
                            shutil.copy(file_name, file_path)
                            with open(file_path, 'rb') as url_file_data:
                                predicted_fs.put(url_file_data, filename="MTCNN_no_output_detected_{0}.{1}".format(
                                file_name_split, image_extension))
                        else:
                            face_encodings = face_recognition.face_encodings(small_frame, mtcnn_loc)
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
                                for (top, right, bottom, left), name in zip(mtcnn_loc, face_names):
                                    color = list(np.random.random(size=3) * 256)
                                    cv2.rectangle(small_frame, (left, top), (right, bottom), color, 2)
                                    cv2.rectangle(small_frame, (left, bottom + 15), (right + len(name), bottom),
                                                  color, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(small_frame, name, (left + 1, bottom + 10), font, 0.6,
                                                (255, 225, 255), 2)
                                    file_path = os.path.join(save_folder_path, "{0}\MTCNN_{1}_{2}.{3}".format(
                                        file_name_split, file_name_split, name, image_extension))
                                    cv2.imwrite(file_path, small_frame)
                                    with open(file_path, 'rb') as url_file_data:
                                        predicted_fs.put(url_file_data, filename="MTCNN_{0}_{1}.{2}".format(
                                        file_name_split, name, image_extension))
                                    

                    except Exception as e:
                        print(f"MTCNN_ error : {e}")
                        
                    try:
                        if "Unknown" in face_names:
                            cnn_face_detector = dlib.cnn_face_detection_model_v1(r"CNN_face_detector\mmod_human_face_detector.dat")
                            img = dlib.load_rgb_image(file_name)
                            dets = cnn_face_detector(img, 1)
                            dlib_loc =[]

                            for i, d in enumerate(dets):
                                dlib_loc.append((d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()))

                            if not os.path.exists(os.path.join(save_folder_path, os.path.split(file_name)[1][:-6])):
                                os.mkdir(os.path.join(save_folder_path, os.path.split(file_name)[1][:-6]))

                            if len(dlib_loc) == 0:
                                file_path = os.path.join(save_folder_path, "{0}\Dlib_no_output_detected_{1}.{2}".format(file_name_split, file_name_split, image_extension))
                                shutil.copy(file_name, file_path)
                                with open(file_path, 'rb') as url_file_data:
                                    predicted_fs.put(url_file_data, filename="Dlib_no_output_detected_{0}.{1}".format(file_name_split, image_extension))
                            else:
                                face_encodings = face_recognition.face_encodings(small_frame, dlib_loc)
                                face_names = []
                                for face_encoding in face_encodings:
                                    matches = face_recognition.compare_faces(known_face_encodings,face_encoding)  # , tolerance=0.95)
                                    name = "Unknown"
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        name = known_face_names[best_match_index]
                                    face_names.append(name)
                                    for (top, right, bottom, left), name in zip(dlib_loc, face_names):
                                        color = list(np.random.random(size=3) * 256)
                                        cv2.rectangle(small_frame, (left, top), (right, bottom), color, 2)
                                        cv2.rectangle(small_frame, (left, bottom + 15), (right + len(name), bottom), color, cv2.FILLED)
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(small_frame, name, (left + 1, bottom + 10), font, 0.6, (255, 225, 255), 2)
                                        file_path = os.path.join(save_folder_path, "{0}\Dlib_{1}_{2}.{3}}".format(file_name_split, file_name_split, name, image_extension))
                                        cv2.imwrite(file_path, small_frame)
                                        with open(file_path, 'rb') as url_file_data:
                                            predicted_fs.put(url_file_data, filename="Dlib_{0}_{1}.{2}}".format(file_name_split, name, image_extension))

                    except Exception as e:
                        print(f"Dlib_ error : {Exception}")

                    try:
                        if "Unknown" in face_names:
                            face_names = []
                            face_locations = face_recognition.face_locations(small_frame)

                            if not os.path.exists(os.path.join(save_folder_path, os.path.split(file_name)[1][:-6])):
                                os.mkdir(os.path.join(save_folder_path, os.path.split(file_name)[1][:-6]))

                            if len(face_locations) == 0:
                                file_path = os.path.join(save_folder_path, "{0}\Face_recog_no_output_detected_{1}.{2}".format(
                                    file_name_split, file_name_split, image_extension))
                                shutil.copy(file_name, file_path)
                                with open(file_path, 'rb') as url_file_data:
                                    predicted_fs.put(url_file_data, filename="Face_recog_no_output_detected_{0}.{1}".format(
                                    file_name_split, image_extension))
                            else:
                                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                                for face_encoding in face_encodings:
                                    matches = face_recognition.compare_faces(known_face_encodings,
                                                                             face_encoding)  # , tolerance=0.95)
                                    name = "Unknown"
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        name = known_face_names[best_match_index]
                                    face_names.append(name)
                                    for (top, right, bottom, left), name in zip(dlib_loc, face_names):
                                        color = list(np.random.random(size=3) * 256)
                                        cv2.rectangle(small_frame, (left, top), (right, bottom), color, 2)
                                        cv2.rectangle(small_frame, (left, bottom + 15), (right + len(name), bottom), color, cv2.FILLED)
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(small_frame, name, (left + 1, bottom + 10), font, 0.6, (255, 225, 255), 2)
                                        file_path = os.path.join(save_folder_path, "{0}\Face_recog_{1}_{2}.{3}".format(file_name_split,file_name_split,  name, image_extension))
                                        cv2.imwrite(file_path, small_frame)
                                        with open(file_path, 'rb') as url_file_data:
                                            predicted_fs.put(url_file_data, filename="Face_recog_{0}_{1}.{2}".format(file_name_split,  name, image_extension))
                    
                    except Exception as e:
                        print(f"Face_recog_ error : {e}")

                            # client = MongoClient("mongodb://localhost:27017")
                            # output_image_db = client["Predicted_Images_DataBase"]
                            # predicted_fs = gridfs.GridFS(output_image_db)

                            # save_folder_path = os.path.dirname(__file__) + r"\Output"
                            # db_collection = output_image_db[name]
                            # for save_file_name in os.listdir(save_folder_path):
                            #     # with open(os.path.join(save_folder_path, folder, file_name), 'rb') as f:
                            #     #     contents = f.read()
                            #     #     predicted_fs = gridfs.GridFS(output_image_db)
                            #     # predicted_fs.put(contents, filename=file_name)
                            #     if save_file_name.endswith(".jpeg"):
                            #         image = cv2.imread(os.path.join(save_folder_path, save_file_name))
                            #         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            #         # convert ndarray to string
                            #         imageString = image.tostring()

                            #         # store the image
                            #         imageID = predicted_fs.put(imageString, encoding='utf-8')
                            #         print(session['data']['text'])
                            #         # create our image meta data
                            #         meta = {
                            #             'name': save_file_name,
                            #             'images': [
                            #                 {
                            #                     'imageID': imageID,
                            #                     'shape': image.shape,
                            #                     'dtype': str(image.dtype)
                            #                 }
                            #             ],
                            #             'Image_Text': session['data']['text']
                            #         }

                            #         # insert the meta data
                            #         db_collection.insert_one(meta)

                                # # get the image meta data
                                # image = testCollection.find_one({'name': file2})['images'][0]

                                # # get the image from gridfs
                                # gOut = fs.get(image['imageID'])

                                # # convert bytes to ndarray
                                # img = np.frombuffer(gOut.read(), dtype=np.uint8)

                                # # reshape to match the image size
                                # img = np.reshape(img, image['shape'])
                                # cv2.imwrite(os.path.join(output_dir, file2), img)
                            
                        # shutil.rmtree(os.path.join(save_folder_path, folder))
                
            # print(output_text_dict)
            # with open(image_data, 'rb') as input_file:
            #     content = input_file.read()
            # file_folder, file_name = os.path.split(file)
            # input_fs.put(content, filename=file_name)
            # image_db_url_col = image_db["Input_URl_Collection"]
            # file_url = {
            #     "url" : image_data
            # }
            # image_db_url_col.insert_one(file_url)
        #     print("file Saved")

        #     import requests
        #     f = open(r"Input_url_Image\Input.jpg", 'wb')
        #     f.write(requests.get(image_data).content)
        #     f.close()
        #     print("file Not Saved")
            
               
        except Exception as e:
            print(e)
    return redirect(url_for('result'))


@app.route('/result')
def result():
    return render_template("result.html")

    # if "data" in session:
    #     data = session['data']
    #     return render_template(
    #         "result.html",
    #         title="Result",
    #         time=data["time"],
    #         text=data["text"],
    #         words=len(data["text"].split(" "))
    #     )
    # else:
    #     return "Wrong request method."

@app.route('/download', methods=['POST'])
def download():
    # shutil.make_archive("Output", 'zip', r"C:\varun\projects\Optical_character_recognition")
    import os, zipfile

    name = r'Output'
    zip_name = name + '.zip'

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

    zip_ref.close()
    path = send_file(os.path.dirname(__file__) + r"\Output.zip",  as_attachment=True)
    print(path)
    return path

if __name__ == '__main__':
    # Setup Tesseract executable path
    app.run(debug=True)
