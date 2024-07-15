from datetime import datetime
from logging import exception
import cv2
import numpy as np
import os 
import face_recognition
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import shutil
from pymongo import MongoClient
from io import BytesIO
import gridfs
import datetime


client = MongoClient("mongodb://localhost:27017")
image_db = client['images']
input_fs = gridfs.GridFS(image_db)



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])

def after():

    try:
        if (os.path.exists('predicted_files.zip')):
            file_path = "predicted_files.zip"
            os.remove('predicted_files.zip')
            print("File Removed")   
        else:
            pass
        try:
            directory = r"static\file_folder"
            for predict_file_name in os.listdir(directory):
                os.remove(os.path.join(directory, predict_file_name))
        except:
            pass
        
        save_folder_path = r"static\predicted_files"
        for folder in os.listdir(save_folder_path):
            path = os.path.join(save_folder_path, folder)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    except:
        pass

    if 'files[]' not in request.files:
        print("no files")
        flash('No file part')
        return redirect(request.url)
        
    files = request.files.getlist('files[]')
    file_names = []
    input_file_dir = r"static/file_folder/"
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            input_fs.put(file, filename=file.filename)
            for ID in input_fs.find({'filename': file.filename}).distinct('_id'):
                output = input_fs.get(ID).read()
                bytesio_object = BytesIO(output)
                # Write the stuff
                with open(os.path.join(input_file_dir, file.filename), "wb") as output_f:
                    output_f.write(bytesio_object.getbuffer())

            # file.save('static/file_folder/' + filename)
    print("="*50)
    print("IMAGE'S SAVED")
    print(os.path.dirname(__file__))

    modi_image_1 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Modi\modi2.jpg")
    modi_face_encoding_1 = face_recognition.face_encodings(modi_image_1)[0]

    # # Load a second sample picture and learn how to recognize it.
    modi_image_2 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Modi\modi32.jpg")
    modi_face_encoding_2 = face_recognition.face_encodings(modi_image_2)[0]

    modi_image_3 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Modi\modi54.jpg")
    modi_face_encoding_3 = face_recognition.face_encodings(modi_image_3)[0]

    modi_image_4 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Modi\modi94.jpg")
    modi_face_encoding_4 = face_recognition.face_encodings(modi_image_4)[0]

    modi_image_5 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Modi\modi137.jpg")
    modi_face_encoding_5 = face_recognition.face_encodings(modi_image_5)[0]


    amit_image_1 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Amit\amit_47.jpg")
    amit_face_encoding_1 = face_recognition.face_encodings(amit_image_1)[0]

    # Load a second sample picture and learn how to recognize it.
    # amit_image_2 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\amit\\amit_temp_train\\amit_49.jpg")
    # amit_face_encoding_2 = face_recognition.face_encodings(amit_image_2)[0]

    # amit_image_3 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\amit\\amit_temp_train\\amit_60.jpg")
    # amit_face_encoding_3 = face_recognition.face_encodings(amit_image_3)[0]

    # amit_image_4 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\amit\\amit_temp_train\\amit_89.jpg")
    # amit_face_encoding_4 = face_recognition.face_encodings(amit_image_4)[0]

    # amit_image_5 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\amit\\amit_temp_train\\amit_352.jpg")
    # amit_face_encoding_5 = face_recognition.face_encodings(amit_image_5)[0]

    kcr_image_1 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Kcr\kcr_22.jpg")
    kcr_face_encoding_1 = face_recognition.face_encodings(kcr_image_1)[0]

    # Load a second sample picture and learn how to recognize it.
    # kcr_image_2 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\kcr\\kcr_temp_train\\kcr_39.jpg")
    # kcr_face_encoding_2 = face_recognition.face_encodings(kcr_image_2)[0]

    # kcr_image_3 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\kcr\\kcr_temp_train\\kcr_40.jpg")
    # kcr_face_encoding_3 = face_recognition.face_encodings(kcr_image_3)[0]

    # kcr_image_4 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\kcr\\kcr_temp_train\\kcr_132.jpg")
    # kcr_face_encoding_4 = face_recognition.face_encodings(kcr_image_4)[0]

    # kcr_image_5 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\kcr\\kcr_temp_train\\kcr_226.jpg")
    # kcr_face_encoding_5 = face_recognition.face_encodings(kcr_image_5)[0]


    jagan_image_1 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Jagan\jagan1_496.jpg")
    jagan_face_encoding_1 = face_recognition.face_encodings(jagan_image_1)[0]

    # Load a second sample picture and learn how to recognize it.
    # jagan_image_2 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\jagan\\jagan_temp\\jagan_19.jpg")
    # jagan_face_encoding_2 = face_recognition.face_encodings(jagan_image_2)[0]

    # jagan_image_3 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\jagan\\jagan_temp\\jagan_88.jpg")
    # jagan_face_encoding_3 = face_recognition.face_encodings(jagan_image_3)[0]

    # jagan_image_4 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\jagan\\jagan_temp\\jagan_106.jpg")
    # jagan_face_encoding_4 = face_recognition.face_encodings(jagan_image_4)[0]

    # jagan_image_5 = face_recognition.load_image_file("C:\\Users\\venkatesh.ch\\FACE DETECTION AND RECOGNITION PROJECT\\jagan\\jagan_temp\\jagan_110.jpg")
    # jagan_face_encoding_5 = face_recognition.face_encodings(jagan_image_5)[0]


    # stalin_image_1 = face_recognition.load_image_file(r"C:\Users\venkatesh.ch\FACE DETECTION AND RECOGNITION PROJECT\Python_Files\temp_train_new_folder\stalin\stalin_13.jpg")
    # stalin_face_encoding_1 = face_recognition.face_encodings(stalin_image_1)[0]

    # # Load a second sample picture and learn how to recognize it.
    # stalin_image_2 = face_recognition.load_image_file(r"C:\Users\venkatesh.ch\FACE DETECTION AND RECOGNITION PROJECT\Python_Files\temp_train_new_folder\stalin\stalin_138.jpg")
    # stalin_face_encoding_2 = face_recognition.face_encodings(stalin_image_2)[0]

    # stalin_image_3 = face_recognition.load_image_file(r"C:\Users\venkatesh.ch\FACE DETECTION AND RECOGNITION PROJECT\Python_Files\temp_train_new_folder\stalin\stalin_131.jpg")
    # stalin_face_encoding_3 = face_recognition.face_encodings(stalin_image_3)[0]

    # stalin_image_4 = face_recognition.load_image_file(r"C:\Users\venkatesh.ch\FACE DETECTION AND RECOGNITION PROJECT\Python_Files\temp_train_new_folder\stalin\stalin_138.jpg")
    # stalin_face_encoding_4 = face_recognition.face_encodings(stalin_image_4)[0]

    stalin_image_5 = face_recognition.load_image_file(os.path.dirname(__file__) + r"\Stalin\stalin_1.jpg")
    stalin_face_encoding_5 = face_recognition.face_encodings(stalin_image_5)[0]

    # FumioKishida_image_5 = face_recognition.load_image_file(r"C:\Users\venkatesh.ch\FACE DETECTION AND RECOGNITION PROJECT\Python_Files\temp_train_new_folder\Fumio Kishida\FumioKishida19.jpg")
    # FumioKishida_face_encoding_5 = face_recognition.face_encodings(FumioKishida_image_5)[0]


    # Create arrays of known face encodings and their names
    known_face_encodings = [
        modi_face_encoding_1,
        modi_face_encoding_2,
        modi_face_encoding_3,
        modi_face_encoding_4,
        modi_face_encoding_5,
        amit_face_encoding_1,
        # amit_face_encoding_2,
        # amit_face_encoding_3,
        # amit_face_encoding_4,
        # amit_face_encoding_5,
        kcr_face_encoding_1,
        # kcr_face_encoding_2,
        # kcr_face_encoding_3,
        # kcr_face_encoding_4,
        # kcr_face_encoding_5,
        jagan_face_encoding_1,
        # jagan_face_encoding_2,
        # jagan_face_encoding_3,
        # jagan_face_encoding_4,
        # jagan_face_encoding_5,
        # stalin_face_encoding_1,
        # stalin_face_encoding_2,
        # stalin_face_encoding_3,
        # stalin_face_encoding_4,
        stalin_face_encoding_5
        # FumioKishida_face_encoding_5
    ]

    known_face_names = [
        "Modi",
        "Modi",
        "Modi",
        "Modi",
        "Modi",
        "AmitShah",
        # "Amit Shah",
        # "Amit Shah",
        # "Amit Shah",
        # "Amit Shah",
        "kcr",
        # "kcr",
        # "kcr",
        # "kcr",
        # "kcr",
        "Jagan"
        # "Jagan",
        # "Jagan",
        # "Jagan",
        # "Jagan",
        # "Stalin",
        # "Stalin",
        # "Stalin",
        "Stalin"
        # "Stalin"
        # "FumioKishida"  
    ]

    face_locations = list()
    face_encodings = list()
    face_names = list()
    error_files = list()

    # Initialize some variables
    directory = r'static/file_folder/'
    print(directory)


    # for folder in os.listdir(directory):
    for filename in os.listdir(directory):
        try:
            # if filename.endswith(".jpg" or ".jpeg"):
            frame = os.path.join(directory, filename)
            print(frame)
            small_frame = cv2.imread(frame)
            # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding) #, tolerance=0.95)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                try:
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)
                except:
                    error_folder_path = r'static\predicted_files\not_predicted_files'
                    if not os.path.exists(error_folder_path):
                        os.mkdir(error_folder_path)
                    shutil.copy(frame, os.path.join(error_folder_path, filename))
#                     top, right, bottom, left = face_locations[0]
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    color = list(np.random.random(size=3) * 256)
                    cv2.rectangle(small_frame, (left, top), (right, bottom), color, 2)
                    # (w, h) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.rectangle(small_frame, (left, bottom + 15), (right + len(name), bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(small_frame, name, (left + 1, bottom + 10), font, 0.6, (255, 225, 255), 2)
                    # cv2.imshow('image', small_frame)

                    save_folder_path = r'static\predicted_files' + f"\{name}"
                    if not os.path.exists(save_folder_path):
                        os.mkdir(save_folder_path)
                    save_folder_path_1, save_folder_path_2 = os.path.split(save_folder_path)
                    print(save_folder_path_2)
                    file_path = os.path.join(save_folder_path, filename)
                    print(file_path)
                    cv2.imwrite(file_path, small_frame)
                    cv2.waitKey(0)  
#                         pil_image = Image.fromarray(small_frame)
# #                 pil_image.show()
# #                 file_path = os.path.join(file_folder, str(count) + ".jpg")
#                         pil_image.save(file_path)
                                                    

        except Exception as e:
            error_files.append([filename, e])
            print(error_files)


    try:
        zip_folde_path1, zip_folder_path2 = os.path.split(save_folder_path_1)
        shutil.make_archive(zip_folder_path2, 'zip', save_folder_path_1)
        try:


            directory = r"static\file_folder"
            for predict_file_name in os.listdir(directory):
                os.remove(os.path.join(directory, predict_file_name))

        
            client = MongoClient("mongodb://localhost:27017")
            output_image_db = client["Predicted_Images_DB"]
            predicted_fs = gridfs.GridFS(output_image_db)
            
            save_folder_path = r"static\predicted_files"
            for folder in os.listdir(save_folder_path):
                db_collection = output_image_db[folder]
                for file_name in os.listdir(os.path.join(save_folder_path, folder)):
                    # with open(os.path.join(save_folder_path, folder, file_name), 'rb') as f:
                    #     contents = f.read()
                    #     predicted_fs = gridfs.GridFS(output_image_db)
                    # predicted_fs.put(contents, filename=file_name)
                    image = cv2.imread(os.path.join(save_folder_path, folder, file_name))
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # convert ndarray to string
                    imageString = image.tostring()

                    # store the image
                    imageID = predicted_fs.put(imageString, encoding='utf-8')

                    # create our image meta data
                    meta = {
                        'name': file_name,
                        'images': [
                            {
                                'imageID': imageID,
                                'shape': image.shape,
                                'dtype': str(image.dtype)
                            }
                        ]
                    }

                    # insert the meta data
                    db_collection.insert_one(meta)

                    # # get the image meta data
                    # image = testCollection.find_one({'name': file2})['images'][0]

                    # # get the image from gridfs
                    # gOut = fs.get(image['imageID'])

                    # # convert bytes to ndarray
                    # img = np.frombuffer(gOut.read(), dtype=np.uint8)

                    # # reshape to match the image size
                    # img = np.reshape(img, image['shape'])
                    # cv2.imwrite(os.path.join(output_dir, file2), img)
                
                shutil.rmtree(os.path.join(save_folder_path, folder))
        except:
            return render_template('index.html')

    except:
        return render_template('index.html')

    
    return render_template('predict.html')
    



@app.route('/download', methods=['POST'])
def download():
    path = send_file(os.path.dirname(__file__) + r"\predicted_files.zip",  as_attachment=True)
    print(path)
    return path

# @app.route('/download', methods=['GET', 'POST'])
# def download(filename):
#     # Appending app path to upload folder path within app root folder
#     # uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
#     # Returning file from appended path\
#     filename = "predicted_files.zip"
#     return send_from_directory(filename=filename)

if __name__ == "__main__":
     app.run(debug=True)