import datetime
import io

from PIL import Image
import easyocr
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import os
app = Flask(__name__)

# Secret key for sessions encryption
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def home():
    return render_template("index.html", title="Image Reader")


@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    if request.method == 'POST':
        start_time = datetime.datetime.now()
        image_data = request.form['text']

        reader = easyocr.Reader(['ta', 'en'])
        bounds = reader.readtext(image_data)
        temp = ''
        for ind in range(len(bounds)):
            temp = temp + bounds[ind][1]

        session['data'] = {
            "Source" : image_data,
            "text": temp,
            "time": str((datetime.datetime.now() - start_time).total_seconds())
        }

        with open('Extracted_text.txt', 'w', encoding="utf-8") as f:
            if "data" in session:
                f.write(f"Source : {image_data}")
                f.write('\n')
                f.write(f"Text : {session['data']['text']}")

        #return redirect(url_for('result'))
        return jsonify(session['data'])


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
    path = send_file(os.path.dirname(__file__) + r"\Extracted_text.txt",  as_attachment=True)
    print(path)
    return path

if __name__ == '__main__':
    # Setup Tesseract executable path
    app.run(debug=True)
