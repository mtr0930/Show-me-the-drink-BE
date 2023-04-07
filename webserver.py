import os

import flask
from flask import render_template, request
import json
from prediction import predict
import pymysql
from ocr import ocr_prediction
from set_vars import drink_names


db = pymysql.connect(host="localhost", user="root", passwd="****", db="graduation_project", charset="utf8", port=3300)
cur = db.cursor()
sql = "SELECT * from drinks"
cur.execute(sql)
data_list = cur.fetchall()

app = flask.Flask(__name__)
@app.route('/')
def home_page():
	return render_template('index.html', data_list=data_list)



@app.route('/uploadfile', methods = ['POST', 'GET'])
def handle_request():
    file = request.files['file']
    drink = {"drink_name" : "none"}
    if file != None:
        file_path = "./upload/" + file.filename
        file.save(os.path.join("upload", file.filename))
        result = ocr_prediction(file_path)
        # 예측 결과가 none 이라면
        if result == "none":
            drink = {"name": "none", "type": "none", "flavor": "none",
                     "cautions": "none"}
            return json.dumps(drink)

        # 예측 결과가 none 아니면 아래 실행
        drink_sql = "SELECT * FROM drinks WHERE name=" + '"' + result + '"'
        cur.execute(drink_sql)
        drink_result = cur.fetchall()
        drink_result = drink_result[0]
        print(drink_result)
        drink_name = drink_names[result]
        drink = {"name": drink_name, "type" : drink_result[1], "flavor" : drink_result[2], "cautions" : drink_result[3]}

    return json.dumps(drink)



app.run(host="0.0.0.0", port=5555, debug=True)
