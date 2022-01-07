import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures



def classifier(age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    ### Get the info from the user:
    
    if ever_married == "Yes":
        ever_married_Yes, ever_married_No = 1, 0
    elif ever_married == "No":
        ever_married_Yes, ever_married_No = 0, 1
    else:
        print("invalid input")
        return 0
        
    if hypertension == "Yes":
        hypertension = 1
    elif hypertension == "No":
        hypertension = 0
    else:
        print("invalid input")
        return 0
        
    if heart_disease == "Yes":
        heart_disease = 1
    elif heart_disease == "No":
        heart_disease = 0
    else:
        print("invalid input")
        return 0

    work_type_Govt, work_type_Never_worked, work_type_Private = 0, 0, 0
    work_type_Self_employed, work_type_children = 0, 0
    
    Residence_type_Rural, Residence_type_Urban = 0, 0
    
    smoking_status_Unknown, smoking_status_formerly_smoked = 0, 0
    smoking_status_never_smoked, smoking_status_smokes = 0, 0
    
    
    if work_type == "Never worked":
        work_type_Never_worked = 1
    elif work_type == "Govt job":
        work_type_Govt = 1
    elif work_type == "Private":
        work_type_Govt = 1
    elif work_type == "Self-employed":
        work_type_Self_employed = 1
    elif work_type == "Children":
        work_type_children = 1
    else:
        print("invalid input")
        return 0
        
    if Residence_type == "Rural":
        Residence_type_Rural = 1
    elif Residence_type == "Urban":
        Residence_type_Urban = 1
    else:
        print("invalid input")
        return 0
        
    if smoking_status == "Unknown":
        smoking_status_Unknown = 1
    elif smoking_status == "formerly smoked":
        smoking_status_formerly_smoked = 1
    elif smoking_status == "never smoked":
        smoking_status_never_smoked = 1
    elif smoking_status == "smokes":
        smoking_status_smokes = 1
    else:
        print("invalid input")
        return 0
        
    list = [[0, 0, 0, age, hypertension, heart_disease, avg_glucose_level, bmi,
             ever_married_No, ever_married_Yes, work_type_Govt, work_type_Never_worked,
             work_type_Private, work_type_Self_employed, work_type_children, Residence_type_Rural,
             Residence_type_Urban, smoking_status_Unknown, smoking_status_formerly_smoked, smoking_status_never_smoked,
             smoking_status_smokes]]
    new_df = pd.DataFrame(list)

    
    ## Now let us get train the models
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")
    data = data.set_index("id") 

    # stores the features in data_f
    # and the targets in data_t
    data_f = data.iloc[:, 0:10]
    data_t = data.iloc[:, 10]
    
    data["bmi"].fillna(data["bmi"].median(), inplace = True)
    data_f["bmi"].fillna(data_f["bmi"].median(), inplace = True)

    gender = pd.get_dummies(data_f["gender"], prefix = "gender")
    married = pd.get_dummies(data_f["ever_married"], prefix = "ever_married")    
    work = pd.get_dummies(data_f["work_type"], prefix = "work_type")    
    res = pd.get_dummies(data_f["Residence_type"], prefix = "Residence_type")    
    smoking = pd.get_dummies(data_f["smoking_status"], prefix = "smoking_status")    
    data_new = data_f.drop(["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], axis = 1)    
    data_fc = pd.concat([data_new, gender, married, work, res, smoking], axis = 1)    
    
    
    rs_scaler = preprocessing.RobustScaler()
    # apply the scaler to the data
    data_fs = rs_scaler.fit_transform(data_fc)
    data_fsd = pd.DataFrame(data_fs, columns = data_fc.columns)
    
    new_df = rs_scaler.transform(new_df)

    new_df = pd.DataFrame(new_df)
    new_df = new_df.iloc[:, 3:]
    
    del data_fsd["gender_Male"]
    del data_fsd["gender_Other"]
    del data_fsd["gender_Female"]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    data_fsd_poly = poly.fit_transform(data_fsd)
    
    new_df = poly.transform(new_df)

    poly_features = pd.DataFrame(data_fsd_poly)
    new_df = pd.DataFrame(new_df)
    
    poly_features['stroke'] = data_t.values
    
    correlations = poly_features.corr()['stroke'][:]
    
    to_be_deleted = []

    for i in range(len(poly_features.columns) - 1):
    # gets the correlation between a single feature
        x = correlations[i]
        if abs(x) < 0.05:
            to_be_deleted.append(i)

    poly_features = poly_features.drop(poly_features.columns[to_be_deleted], axis=1)
    new_df = new_df.drop(new_df.columns[to_be_deleted], axis = 1)
      
    stroke = poly_features[poly_features["stroke"] == 1]
    stroke_over = pd.concat([stroke.sample(100), stroke.sample(100), stroke.sample(100), 
                             stroke.sample(100), stroke.sample(100), stroke.sample(100),
                             stroke.sample(100), stroke.sample(100), stroke.sample(100),
                             stroke.sample(100)])

    non_stroke = poly_features[poly_features["stroke"] == 0]
    non_stroke_under = non_stroke.sample(1000)
    
    artificial = pd.concat([stroke_over, non_stroke_under])
    artificial_f = artificial.iloc[:, :-1]
    artificial_t = artificial.iloc[:, -1]
    
    classifier = KNeighborsClassifier(n_neighbors=2, 
                                      weights="uniform")
    classifier.fit(artificial_f, artificial_t)
    
    Y_prediction = classifier.predict(new_df)
    if Y_prediction[0] == 1:
        return "You are in danger of a stroke"
        
    
    else:
        return "You are safe from the risk of a stroke"


#The user for now will input their data through this line
# Age, hypertension, heart disease, ever married, work type, residence type, avg_glocuse_level, bmi, smoking
# for work type: "Private", "Govt job", "Never worked", "Self-employed", "Children"
# for residence type: "Rural", "Urban"
# for smoking:"never smoked", "formerly smoked", "Uknown", "smokes"
age = input("Please enter you age: ")
hypertension = input("Do you have hypertension(Yes or No)?: ")
heart_disease = input("Do you have any heart diseases(Yes or No)?: ")
ever_married = input("Have you ever been married(Yes or No)?: ")
work_type = input("Work type(Private, Govt job, Never worked, Self-employed, Children): ")
Residence_type = input("Please choose you residence type(Rural, Urban): ")
avg_glucose_level = input("Please enter you average glucose level: ")
bmi = input("Please enter you bmi: ")
smoking_status = input("Please enter your smoking status(never smoked, formerly smoked, smokes, refrain): ")



print(classifier(age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status))
    
    
    
    
        
    
    
    