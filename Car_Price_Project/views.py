from django.shortcuts import render
import pickle
import numpy as np



def home(request):
    
    return render(request, 'predict.html')

def result(request):
    results = request.GET['car']
    result2 = request.GET['carmodel']
    mileageA = request.GET['mileage']
    conditionA = request.GET['condition']
    yearA = request.GET['year']
    result3 = load_model(results, result2, mileageA, conditionA, yearA)
    price_a = Average_error+result3
    price_a = round(price_a)
    price_b = result3-Average_error
    price_b = round(price_b)
    price_a = format(price_a,',d')
    price_b = format(price_b, ',d')
    return render(request, 'result.html', {'Name': results, 'Carmodule': result2, 'Caryear': yearA, 'Dump': price_a, 'Ramp': price_b})

model = pickle.load(open('Car_Price_Project/Group_estimator.pkl', 'rb'))

RF = model["model0"]  # regressor
Name = model["Name"]  # Name label encoder
Condition = model["Condition"]  # Name label encoder
Standard_Scalar = model["Standard_Scalar"]  # location label encoder
Average_error = model["Average_error"]  # location label encoder


def load_model(word1, word2, mile, cond, yea):
    nam = word1+' '+word2
    mile = int(mile)
    yea = int(yea)
    recall = [[nam, mile, cond, yea]]
    #recall.append(int(mile))
    #recall.append(cond)
    #recall.append(int(yea))
    #recall = [recall]
    recall = np.array(recall)
    recall[:, 0] = Name.transform(recall[:, 0])
    recall[:, 2] = Condition.transform(recall[:, 2])
    recall = Standard_Scalar.transform(recall)
    recall = RF.predict(recall)
    recall = round(recall[0])  # round to the neaarest Naira
    return recall



#    <form action="{%url 'prediction'%}" method="post">
#    </form>
 #<input type="text" name="mileage" id="mileage">