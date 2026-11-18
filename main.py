from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app=FastAPI()
model=joblib.load("model.pkl")
class Flowerdata(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float
@app.get("/")
def Home():
    return {"message":"MODEL IS READY TO PREDICT"}
@app.post("/predict")
def predict(data: Flowerdata):
  input_data= [[
     data.sepal_length,
     data.sepal_width,
     data.petal_length,
     data.petal_width
  ]]
  prediction = model.predict(input_data)

  Flower_Name = ["Setosa", "Versicolor", "Virginica"]
  result= Flower_Name[prediction[0]]
  return {"prediction": result}


@app.get("/BIO")
def bio():
   return {"name":"Shabnam Thakur",
           "age":21,
           "city":"Shimla",
           "email":"tshabnam569@gmail.com",
           "course":"Data analytics",
           "mobile":"7590971335"}

class email(BaseModel):
   email:str
   name:str
   age:int
   mobile:int
   city:str
   course:str
@app.post("/email")
def inputdata(data: email):
   return {"email":data.email,
           "name":data.name,
           "age":data.age,
           "mobile":data.mobile,
           "city":data.city,
           "course":data.course}


