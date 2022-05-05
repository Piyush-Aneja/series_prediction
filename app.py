from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
from mode import training
from mode import plotting
import os
print("Started..!!!!!".center(40))
# model=pickle.load(open("myseries.pkl","rb"))
app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
@app.route('/')
def home():
    print("home page")
    if os.path.exists("static/graph.png"):
        print("file deleted")
        os.remove("static/graph.png")
    return render_template("index.html")



@app.route("/predict",methods=["POST"])

def predict():
    xx,yy=[],[]
    xfeatures=request.form["train"]
    z=np.array(float(request.form['test']))
   
    # yfeatures=request.form["test"]
    x=xfeatures.split(',')
    for i in range(0,len(x),2):
        xx.append(float(x[i]))
        yy.append(float(x[i+1]))
    xx=np.array(xx)
    yy=np.array(yy)
    # plotting(xx,yy)
    xx=xx.reshape(-1,1)
    yy=yy.reshape(-1,1)
    print(xx)
    print(yy)
    training(xx,yy)
    print("Prediction on: ",z)
    print("Done training..!!!!")
    lr=pickle.load(open("myseries1.pkl","rb"))
    z=z.reshape(1,-1)
    res=lr.predict(z)
    print("Prediction result :",res)
    # print(type(z[0][0]))
    ans=round(res[0][0],3)
    display_arr=[ f"Successor of Term :{z[0][0]}","Your Series :"+xfeatures,f"The next number might be :{ans} ","static\\graph.png"]
    return render_template("index.html",term= display_arr[0],series=display_arr[1],prediction_value=display_arr[2],img_src=display_arr[3])
    # return render_template("index.html",term= f"Successor of Term :{z[0][0]}",series="Your Series :"+xfeatures,prediction_value=f"The next number might be :{ans} "))
    

    
    
        
    # xfeatures=np.array(xfeatures)
    # y=yfeatures.split(',')
    
    return render_template("index.html",prediction_text="Next number would be ")

# @app.route("/predict_api",methods=["POST","GET"])
# def predict_api():
#     # data=request.get_json(force=True)
#     prediction=model.predict([[8]])   
#     output=prediction[0][0] 
#     return {"msg":output}


if __name__ == '__main__':
    app.run(debug=True)
    
    
    