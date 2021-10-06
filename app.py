import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd


app = Flask(__name__)
import os
os.getcwd()
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    fitted = model
    #.fit(disp = -1)
    #print(fitted.summary())
    fc1, se1, conf1 = fitted.forecast(1828, alpha=0.05)  # 95% confidence
    fc_series1 = pd.DataFrame(fc1, index = pd.date_range(start = pd.to_datetime('today').date(), periods = 1828, freq='D'), columns = ['forecast'])
    
    fc_series1 = fc_series1.reset_index()
    fc_series1.columns=['Date','Price']
    fc_series1.set_index('Date',inplace = True)
    fc_series1['Date'] = fc_series1.index.date
    fc_series1.set_index('Date', inplace=True)
    fc_series1['Date'] = fc_series1.index
    fc_series1.set_index('Date', inplace=True)
    fc_series1=fc_series1.reset_index()
    buy_price = fc_series1.iloc[1,-1]
    return_price = fc_series1.iloc[1827,-1]
    profit = ((return_price-buy_price)/buy_price)*100
    print(profit)
    
    return render_template('index.html',forecast1 ="Profit percentage is ${}".format(profit))

   

if __name__ == "__main__":
    app.run(debug=True)
    

