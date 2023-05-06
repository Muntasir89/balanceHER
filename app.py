from flask import Flask,request,jsonify
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "Woman Work Life Balance"

@app.route('/predict', methods=['POST'])
def predict():
    vegetables = request.form.get('vegetables') #1
    stress = request.form.get('stress')         #2
    newPlace = request.form.get('newPlace')     #3
    people = request.form.get('people')         #4
    helpedPeople = request.form.get('helpedPeople') #5
    interaction = request.form.get('interaction')   #6
    achievements = request.form.get('achievements') #7
    donations = request.form.get('donations')       #8
    bmi = request.form.get('bmi')                   #9
    workCompletion = request.form.get('workCompletion') #10
    actions = request.form.get('actions')               #11
    stepsWalks = request.form.get('stepsWalks')         #12
    liveVision = request.form.get('liveVision')         #13
    sleepHours = request.form.get('sleepHours')         #14
    vacation = request.form.get('vacation')             #15
    anger = request.form.get('anger')                   #16
    income = request.form.get('income')                 #17
    awards = request.form.get('awards')                 #18
    passionHours = request.form.get('passionHours')     #19
    meditaionsHours = request.form.get('meditaionsHours')   #20
    age = request.form.get('age')                           #21
    gender = request.form.get('gender')                     #22

    input_query=np.array([[vegetables,stress,newPlace,people,helpedPeople,interaction,achievements,donations,bmi,
                           workCompletion,actions,stepsWalks,liveVision,sleepHours,vacation,anger,income,awards,passionHours,meditaionsHours,age, gender]], dtype=int)
    result=model.predict(input_query)[0]

    return jsonify({'result':str(result)})


if __name__=='__main__':
    app.run(debug=True)