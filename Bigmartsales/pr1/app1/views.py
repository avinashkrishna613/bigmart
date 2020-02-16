from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from .forms import QuestionForm 
from . import templates
from sklearn.ensemble import RandomForestRegressor
def load_data():
    df1=pd.read_csv('/home/kolaparthi/Bigmartsales/train.csv') #loading the train set
    df2=pd.read_csv('/home/kolaparthi/Bigmartsales/test.csv')  #loading the test set
    return (df1, df2)





def removeNull(df1):
    df1['Item_Weight'].fillna(value=df1['Item_Weight'].mean(),inplace=True)
   

    #Filling the Null values in 'Outlet_Size' column with "Unknown"
    df1['Outlet_Size'].fillna(value='Unknown',inplace=True)
   
    return (df1)

def item_identify(cols):
  item_id=cols[0]
  item_type=cols[1]
  
  if item_id[:2] == 'NC':
    return 'Non Consumables'
  elif item_id[:2] == 'DR':
    return 'Drinks'
  else:
    return 'Foods'


def apply_item_identify(df1):
    df1['Item_Type']=df1[['Item_Identifier','Item_Type']].apply(item_identify,axis=1)
   
    return (df1)


def item_fat(cols):
  fat=cols[0]
  typ=cols[1]
  
  if (fat=='Low Fat' or fat=='LF' or fat=='low fat') and (typ=='Foods' or typ=='Drinks'):
    return 'Low Fat'
  elif (fat=='Regular' or fat=='reg') and (typ=='Foods' or typ=='Drinks'):
    return 'Regular'
  else:
    return 'Non Edible'

def apply_item_fat(df1):
    df1['Item_Fat_Content']=df1[['Item_Fat_Content','Item_Type']].apply(item_fat,axis=1)
    
    return (df1)

def fill_Item_visibility(df1):
    
    df1['Item_Visibility'].mask(df1['Item_Visibility']== 0,df1['Item_Visibility'].mean(),inplace=True)
   
    return (df1)

def num_years(col):
  return 2013-int(col)



def apply_num_years(df1):
        df1['Years_of_Operation']=df1['Outlet_Establishment_Year'].apply(num_years)
        

        return (df1)






def creatingDataForModel(df1):


   
    item_fat_content=pd.get_dummies(df1['Item_Fat_Content'])
    item_type=pd.get_dummies(df1['Item_Type'])
    outlet_size=pd.get_dummies(df1['Outlet_Size'])
    outlet_location_type=pd.get_dummies(df1['Outlet_Location_Type'])
    output_type=pd.get_dummies(df1['Outlet_Type'])
    
    
    
    train=df1
    train=pd.concat([train,item_fat_content,item_type,outlet_size,outlet_location_type,output_type],axis=1)
    train.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size', 'Outlet_Location_Type',
           'Outlet_Type'],axis=1,inplace=True)
    # test=df2
    # test=pd.concat([test,item_fat_content_test,item_type_test,outlet_size_test,outlet_location_type_test,output_type_test],axis=1)
    # test.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size', 'Outlet_Location_Type',
    #        'Outlet_Type'],axis=1,inplace=True)
    # x=train.drop(['Item_Outlet_Sales'],axis=1)
    # y=train['Item_Outlet_Sales']
    # x_test = test
    df1 = train
   
    return (df1)

def transform_data(x):
    sc_x=StandardScaler()
    x=sc_x.fit_transform(x)

   
    return (x)



def train_validation_split(x, y):
    x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.1,random_state=42)
    return (x_train,x_val,y_train,y_val)



def buildmodel_regression(x_train ,y_train):
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    return lm

def buildmodel_tree(x_train, y_train):
    rf=RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=76,n_jobs=4)
    rf.fit(x_train,y_train)
    return rf

def add_fat(pred):
    l = [0, 0, 0]
    p = pred.iloc[0, 2]
    
    if p == "Low Fat":
        l [0] = 1
    elif p == "Non Edible":
        l[1] = 1
    else:
        l[2] = 1
    df = pd.DataFrame([l], columns = ['Low Fat', 'Non Edible', 'Regular'])
    
    return df


def add_type(pred):
    l = [0, 0, 0]
    p = pred.iloc[0, 4]
    if p == 'Drinks':
        l[0] = 1
    elif p == 'Foods':
        l[1] = 1
    else:
        l[2] = 1
    df = pd.DataFrame([l], columns = ['Drinks', 'Foods', 'Non Consumables'])
   
    return df


def add_outlet_size(pred):
    l = [0, 0,0,0]
    p = pred.iloc[0, 8]
    if p == 'High':
        l[0] = 1
    elif p=='Medium':
        l[1] = 1
    elif p == 'Small':
        l[2] = 1
    else:
        l[3] = 1
    df = pd.DataFrame([l], columns = ['High', 'Medium', 'Small', 'Unknown'])
    # df = df.append({'High':l[0], 'Medium':l[1], 'Small':l[2], 'Unknown':l[3]})
    return df


def add_outlet_loc(pred):
    l = [0, 0,0 ]
    p = pred.iloc[0, 9]
    if p=='Tier 1':
        l[0] = 1
    elif p=='Tier 2':
        l[1] = 1
    else:
        l[2] = 1
    df = pd.DataFrame([l], columns = ['Tier 1', 'Tier 2', 'Tier 3'])
    # df = df.append({'Tier 1':l[0], 'Tier 2':l[1], 'Tier 3':l[2]})
    return df


def add_output_type(pred):
    l = [0, 0,0, 0]
    
    p = pred.iloc[0, 10]
   
    if p == 'Grocery Store':
        l[0] = 1
    elif p=='Supermarket Type1':
        l[1] = 1
    elif p=='Supermarket Type2':
        l[2] = 1
    else:
        l[3] = 1

    df = pd.DataFrame([l], columns = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    # df = df.append({'Grocery Store':l[0], 'Supermarket Type1':l[1], 'Supermarket Type2':l[2], 'Supermarket Type3':l[3]})
    return df

def calculate_metrics(lm, x_val, y_val):
    predictions=lm.predict(x_val)
    
    print('Mean Absolute Error: ',metrics.mean_absolute_error(y_val,predictions))
    print('Mean Squared Error: ',metrics.mean_squared_error(y_val,predictions))
    print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_val,predictions)))
    print('Explained Variance Score: ',metrics.explained_variance_score(y_val,predictions))


def commonForModels(data):
    pred = pd.DataFrame([data], columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
    
    df1, df2 = load_data()
    df1 = removeNull(df1)
    df2 = removeNull(df2)

    df1 = apply_item_identify(df1)
    df2 = apply_item_identify(df2)
    pred = apply_item_identify(pred)

    
    df1 = apply_item_fat(df1)
    df2 = apply_item_fat(df2)
    pred = apply_item_fat(pred)
    

    df1 = fill_Item_visibility(df1)
    df2 = fill_Item_visibility(df2)
    pred = fill_Item_visibility(pred)

    
    df1 = apply_num_years(df1)
    df2 = apply_num_years(df2)
    pred = apply_num_years(pred)
    
    dff1 = add_fat(pred)
   
    dff2 = add_type(pred)
    dff3 = add_outlet_size(pred)
    dff4 = add_outlet_loc(pred)
    dff5 = add_output_type(pred)

    pred = pd.concat([pred, dff1, dff2, dff3, dff4, dff5], axis = 1)
    
    pred.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type'],axis=1,inplace=True)
   
    data1 = creatingDataForModel(df1)
   
    x = data1.drop(['Item_Outlet_Sales'],axis=1)
    y = data1['Item_Outlet_Sales']
    x_test = creatingDataForModel(df2)
    
    x  = transform_data(x)
    x_test = transform_data(x_test)
    
   
    x_train, x_val, y_train, y_val = train_validation_split(x, y)
    
    return (x_train, y_train, x_val, y_val, pred)




def predict_helper_tree(data):
    x_train,  y_train, x_val, y_val, pred = commonForModels(data)
    
    model = buildmodel_tree(x_train , y_train)
    
    calculate_metrics(model, x_val, y_val)
    return model.predict(pred)


def predict_tree(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            stri = "field"
            listing = []
            for i in range(1, 12):
                stri1 = ""+ stri + str(i)+""
                h = request.POST[stri1]
                print(h)
                listing.append(h)
            
            intent = predict_helper_tree(listing)
            
            return HttpResponse(intent)
    else:
        form = QuestionForm()
    return render(request, 'predict1.html', {'form':form})




def predict_helper_regression(data):
    print("Inside predict helper")
   
    x_train, y_train, x_val, y_val, pred = commonForModels(data)
   
    model = buildmodel_regression(x_train , y_train)
    
    calculate_metrics(model, x_val, y_val)
    
    
    return (model.predict(pred))
    


def predict_regression(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            stri = "field"
            listing = []
            for i in range(1, 12):
                stri1 = ""+ stri + str(i)+""
                h = request.POST[stri1]
                print(h)
                listing.append(h)
            
            intent = predict_helper_regression(listing)
            
            return HttpResponse(intent)
    else:
        form = QuestionForm()
    return render(request, 'predict.html', {'form':form})



def predict_helper_xgboost(data):
    x_train, y_train, x_val, y_val, pred = commonForModels(data)
    model = buildmodel_xgboost(x_train , y_train)
    
    calculate_metrics(model, x_val, y_val)
    print(model.predict(pred))


def predict_xgboost(request):
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            stri = "field"
            listing = []
            for i in range(1, 12):
                stri1 = ""+ stri + str(i)+""
                h = request.POST[stri1]
                print(h)
                listing.append(h)
            
            intent = predict_helper_xgboost(listing)
            
            return HttpResponse(intent)
    else:
        form = QuestionForm()
    return render(request, 'predict3.html', {'form':form})