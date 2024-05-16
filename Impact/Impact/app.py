import numpy as np
from flask import Flask, request, jsonify, render_template  
import pickle
app = Flask(__name__,template_folder='template')
model = pickle.load(open(r'C:\Users\hriti\Desktop\j\final\Bethany\fungal.pkl','rb'))
model2 = pickle.load(open(r'C:\Users\hriti\Desktop\j\final\Bethany\diabetes.pkl','rb'))
model3 = pickle.load(open(r'C:\Users\hriti\Desktop\j\final\Bethany\corona.pkl','rb'))
@app.route("/")
def about():
    return render_template('index.html')  
@app.route("/disease")     
def ab0():
    return render_template('sample-inner-page.html')
@app.route("/ambulance")    
def a():
    return render_template('track.html')
@app.route("/or")    
def aa():
    return render_template('register.html')
@app.route("/sexed")    
def aba():
    return render_template('sexed.html')    
@app.route("/fungal")    
def ab1():
    return render_template('fungal.html')
@app.route("/diabetes")    
def ab2():
    return render_template('diabetes.html') 
@app.route("/corona")    
def ab3():
    return render_template('corona.html')        
@app.route('/predict',methods=['POST'])
def home():
    data1= request.form['itching']
    data2= request.form['rash']
    data3= request.form['nodal']
    data4= request.form['patches']
    input_data=(data1,data2,data3,data4)
    arr=np.asarray(input_data)
    input_data_reshaped=arr.reshape(1,-1)
    pred= model.predict(input_data_reshaped)
    ans=pred[0]
    return render_template('after.html',data=ans)
@app.route('/predict2',methods=['POST'])
def home2():
    data1= request.form['preg']
    data2= request.form['glucose']
    data3= request.form['bp']
    data4= request.form['skin']
    data5= request.form['insulin']
    data6= request.form['bmi']
    data7= request.form['dpf']
    data8= request.form['age']
    input_data=(data1,data2,data3,data4,data5,data6,data7,data8)
    arr=np.asarray(input_data)
    input_data_reshaped=arr.reshape(1,-1)
    pred= model2.predict(input_data_reshaped)
    ans=pred[0]
    return render_template('after2.html',data=ans)   
@app.route('/predict3',methods=['POST'])
def home3():
    data1= request.form['paragraph_text']
    data2= request.form['fever']
    data3= request.form['sore']
    data4= request.form['sob']
    data5= request.form['headache']
    input_data=(data1,data2,data3,data4,data5)
    arr=np.asarray(input_data)
    input_data_reshaped=arr.reshape(1,-1)
    pred= model3.predict(input_data_reshaped)
    ans=pred[0]
    return render_template('after3.html',data=ans)      
if __name__ == '__main__':
    app.run(port=5000, debug=True)