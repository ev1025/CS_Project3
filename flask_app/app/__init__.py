from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import json

ca_json = os.path.join(os.getcwd(), 'category.json')
ca_model= os.path.join(os.getcwd(), 'category.pkl')


def create_app():
    app = Flask(__name__)

    from app.routes import bp1 # from(app폴더.routes폴더) import bp파일
    app.register_blueprint(bp1.mbp1) # bp파일.bp변수
    
    @app.route('/')
    def main():
        return '안녕하세요'

    @app.route('/json')
    def json_data():
        json_file = json.load(open(ca_json, 'r'))

        return json_file

    @app.route('/model')
    def model_data():
        model = pickle.load(open(ca_model, 'rb'))
        return model

    @app.route('/html')
    def index():
        pic = pickle.load(open(ca_model, 'rb')) # 모델 예측               
        return render_template('index.html', pic=pic) # app폴더안의 templates폴더에 html파일
                                             # app폴더안의 main폴더에 있다면 'main/index.html'

        
    return app

# if __name__ == "__main__":
#     app.run(debug=True)

    # @app.route('/')
    # def main():
    #     return render_template('home.index')


    # @app.route('/predict', methods=['POST'])
    # def home():
    #     data1 = request.form['a'] # 각 데이터 연결
    #     data2 = request.form['b']
    #     data3 = request.form['c']
    #     data4 = request.form['d']
    #     arr = np.array([[data1, data2, data3, data4]]) # 데이터의 값들
    #     pred = model.predict(arr)                      # 모델 예측
    #     return render_template('after.html', data=pred)