## Flask 웹서버

### 1. Environment

<aside>
 **📌IDE : PyCharm
     Language : Python**

</aside>

### 2. 파일 별 기능 설명

### ocr.py

> Google Cloud Vision API 라이브러리 함수를 활용해서 OCR 기능을 구현하였다.
이미지 분류 모델은 12개 종류의 음료수 외에 구분할 수 없는 dump 이미지들을 학습시키고 `none`이라는 
레이블을 부여 했고  `none` 이라는 레이블로 분류될 경우에만 OCR을 API를 호출하여 분류의 정확도를 높였다.
> 
- [ocr.py](http://ocr.py) 코드
    
    ```python
    def ocr(img_path):
        client = vision.ImageAnnotatorClient()
    
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
    
        img = vision.Image(content=content)
    
        response = client.text_detection(image=img)
        labels = response.text_annotations
    
        # OCR 결과에서 음료명 찾기
        for label in labels:
            print("label description", label.description)
            if label.description in drink_info.keys():
                return drink_info[label.description]
            if label.description.lower() in drink_info.keys():
                return drink_info[label.description.lower()]
        else:
            return -1  # OCR 결과도 없는 경우 -1 반환
    
    def with_model(model, path):
        img_path = path
        changed_image_path = change_image(img_path)
    
        img = image.load_img(changed_image_path, target_size=(150, 150))
    
        x = image.img_to_array(img)
        x = x / 255.  # 이미지 rescale
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
    
        predict = model.predict(images, batch_size=4, verbose=0)
        score = tf.nn.softmax(predict[0])
        print()
        np.set_printoptions(precision=3, suppress=True)
    
        result = predict.argmax()
    
        # 결과가 none일 경우 OCR 실행
        if classes[result] == 'none':
            result = ocr(img_path)
            print("ocr requested")
            print(result)
            if result == -1:
                print("IMAGE NAME: ---------- OCR 결과 없음, 다른 사진 요청")
    
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(classes[result], 100 * np.max(score))
        )
        print("RESULT: {:7} !!!!!!!!!".format(classes[result]))
        return classes[result]
    
    def ocr_prediction(path):
    
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\g_chokuho\\Downloads\\drinks-ocr-api-ba17cbabe064.json'
    
        model = load_model("./model", compile=False)
        predict_result = with_model(model, path=path)
        return predict_result
    ```
    

### prediction.py

> TensorFlow 라이브러리를 이용해서 학습 시킨 모델을 연동하고 predict 함수를 통해
모델의 입력 조건에 맞게 데이터를 넣어주면 13개 분류 결과 중 하나를 출력한다
음료수 종류 
**"cider", "coke", "fanta", "milkis"
, "monster", "mtdew", "pepsi"
, "soda", "sprite", "toreta", "welchis", "none”**
> 
- prediction.py 코드
    
    ```python
    def predict(data_path):
    
        img = image.load_img(data_path, target_size=(img_height, img_width))
    
        img_array = image.img_to_array(img)
        img_array = img_array / 255.
        img_array = tf.expand_dims(img_array, axis=0)
    
        #predict함수가 결과예측해줌.
        predictions = model.predict(img_array)
        print(predictions[0][1])
    
        score = tf.nn.softmax(predictions[0])
        max_index = np.argmax(score)
    
        print(score)
        class_names = ["cider", "coke", "fanta", "milkis", "monster", "mtdew", "pepsi", "soda", "sprite", "toreta", "welchis", "none"]
        pr_result = class_names[max_index]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        return pr_result
    ```
    

### webserver.py

> Flask 서버를 구동하기 위한 몸통 코드이다.  MySQL DB와 연동하고 `/uploadfile` 
경로로 요청이 들어오면 메인 로직으로 처리하고 결과를 JSON타입으로 리턴해준다.
> 
- webserver.py 코드
    
    ```python
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
    ```
