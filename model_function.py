import numpy as np
import pandas as pd
import pickle

#def model():
#retrieve user inputs
input_age = "30"
input_race = "Non-Hispanic White"
input_income = "$75,000+"
input_gender = "Male"
input_stage = "IIB"
input_site = "Thyroid"
input_type = "Adenomas And Adenocarcinomas"
factors = [input_age,input_race,input_income,input_gender,input_stage,input_site,input_type]

#load resources 
prediction_data = pd.read_csv('resources/Blank_Form.csv')
prediction_model = pickle.load(open("resources/model.pkl","rb"))

#model the prediction
prediction_data.Age = int(factors[0])
prediction_data[f'Race_{factors[1]}']=1
prediction_data[f'Median_Household_Income_{factors[2]}']=1
prediction_data[f'Gender_{factors[3]}']=1
prediction_data[f'Cancer_Stage_{factors[4]}']=1
prediction_data[f'Cancer_Site_{factors[5]}']=1
prediction_data[f'Cancer_Type_{factors[6]}']=1

prediction_model.predict(np.reshape(np.array(prediction_data.values.tolist()),(1,1584)))
    
#make the prediction
pred = prediction_model.predict(prediction_data)

#print pred
print(pred)

#return the prediction
#return pred

