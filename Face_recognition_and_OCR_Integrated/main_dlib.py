import datetime
import cv2
import numpy as np
import easyocr
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import os
import shutil
import urllib.request
import face_recognition
import dlib

app = Flask(__name__)

# Secret key for sessions encryption
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@app.route('/')
def home():
    return render_template("index.html", title="Image Reader")


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
            directory = r"static\Input_url_Image"
            for predict_file_name in os.listdir(directory):
                os.remove(os.path.join(directory, predict_file_name))
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

    if request.method == 'POST':
        start_time = datetime.datetime.now()
        image_data = request.form['text']
        try:
            urllib.request.urlretrieve(image_data, r"Input_url_Image\Input.jpg")
        except:
            import requests
            f = open(r"Input_url_Image\Input.jpg", 'wb')
            f.write(requests.get(image_data).content)
            f.close()
        reader = easyocr.Reader(['ta', 'en'])
        bounds = reader.readtext(r"Input_url_Image\Input.jpg")
        temp = ''
        for ind in range(len(bounds)):
            temp = temp + bounds[ind][1]
        session['data'] = {
            "text": temp,
            "time": str((datetime.datetime.now() - start_time).total_seconds())
        }

        with open(r'Output\Extracted_text.txt', 'w', encoding="utf-8") as f:
            if "data" in session:
                f.write(f"Source : {image_data}")
                f.write('\n')
                f.write(f"Text : {session['data']['text']}")

        known_face_encodings = list()
        known_face_names = list()

        for image_file in os.listdir(r"Face_encoding_data"):
            train_image = face_recognition.load_image_file(os.path.join(r"Face_encoding_data", image_file))
            known_face_encodings.append(face_recognition.face_encodings(train_image)[0])
            known_face_names.append(image_file[:-4])

        small_frame = cv2.imread(r"Input_url_Image\Input.jpg")

        try:
            cnn_face_detector = dlib.cnn_face_detection_model_v1(r"CNN_face_detector\mmod_human_face_detector.dat")
            img = dlib.load_rgb_image(r"Input_url_Image\Input.jpg")
            dets = cnn_face_detector(img, 1)
            rects =[]
            print("Number of faces detected: {}".format(len(dets)))
            for i, d in enumerate(dets):
                rects.append((d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()))

            if len(rects) == 0:
                file_path = os.path.join(save_folder_path, "Face_detected_input.jpg")
                shutil.copy(r"Input_url_Image\Input.jpg", file_path)
            face_encodings = face_recognition.face_encodings(small_frame, rects)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings,face_encoding)  # , tolerance=0.95)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                for (top, right, bottom, left), name in zip(rects, face_names):
                    color = list(np.random.random(size=3) * 256)
                    cv2.rectangle(small_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(small_frame, (left, bottom + 15), (right + len(name), bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(small_frame, name, (left + 1, bottom + 10), font, 0.6, (255, 225, 255), 2)
                    file_path = os.path.join(save_folder_path, "Face_detected_{0}.jpg".format(name))
                    cv2.imwrite(file_path, small_frame)
        except Exception as e:
            print(e)
    return redirect(url_for('result'))


@app.route('/result')
def result():
    if "data" in session:
        data = session['data']
        return render_template(
            "result.html",
            title="Result",
            time=data["time"],
            text=data["text"],
            words=len(data["text"].split(" "))
        )
    else:
        return "Wrong request method."

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
