import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")

def start_pg():
    return render_template("start.html")

@app.route("/pred",methods=['POST'])
def predict():
    
    if request.method == "POST":
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        age = float(request.form["Age"])

        features = np.array([[pregnancies, glucose, skin_thickness,insulin,bmi, age]])

        # Scale the features using the loaded StandardScaler
        scaled_features = scaler.transform(features)
        print(scaled_features)

        # Make a prediction using the scaled features and the loaded model
        prediction = model.predict(scaled_features)
              
        
        print("Scaler Mean (Jupyter Notebook):", scaler.mean_)
        print("Scaler Scale (Standard Deviation) (Jupyter Notebook):", scaler.scale_)
        
        output=prediction[0]
        return render_template("start.html", pred_text=f'Prediction = {output}')
    return render_template("start.html")

if(__name__ == '__main__'):
    app.run(debug=True)

    

# import numpy as np
# from flask import Flask,render_template,request,jsonify
# import pickle

# np.set_printoptions(precision=2, suppress=True)

# app=Flask(__name__)
# model=pickle.load(open("model.pkl","rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

# @app.route("/")
# def start_pg():
#     return render_template("start.html")

# @app.route("/pred", methods=['POST'])
# def predict():
#     if request.method == "POST":
#         pregnancies = float(request.form["Pregnancies"])
#         glucose = float(request.form["Glucose"])
#         skin_thickness = float(request.form["SkinThickness"])
#         insulin = float(request.form["Insulin"])
#         bmi = float(request.form["BMI"])
#         age = float(request.form["Age"])

#         features = np.array([[pregnancies, glucose, bmi, skin_thickness, insulin, age]])
        
#                 # Before scaling
#         print("Input Features (Before Scaling):", features)

#         # Scale the features using the loaded StandardScaler
#         scaled_features = scaler.transform(features)

#         print("Scaled Features:", scaled_features)
        
#         # Debugging print statements
#         print("Scaler Object (backend):", scaler)
#         print("Mean (backend):", scaler.mean_)
#         print("Scale (Standard Deviation) (backend):", scaler.scale_)

#         # Make a prediction using the scaled features and the loaded model
#         prediction = model.predict(scaled_features)

#         output = prediction[0]
#         return render_template("start.html", pred_text=f'Prediction = {output}')


# if(__name__ == '__main__'):
#     app.run(debug=True)
