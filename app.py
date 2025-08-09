from flask import Flask,request
from flask import render_template   
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)
@app.route('/')
def hello():
    return render_template('hello.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('predict.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)