"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
© Copyright 2022, California, Department of Motor Vehicle, all rights reserved.
The source code and all its associated artifacts belong to the California Department of Motor Vehicle (CA, DMV), and no one has any ownership
and control over this source code and its belongings. Any attempt to copy the source code or repurpose the source code and lead to criminal
prosecution. Don't hesitate to contact DMV for further information on this copyright statement.

Release Notes and Development Platform:
The source code was developed on the Google Cloud platform using Google Cloud Functions serverless computing architecture. The Cloud
Functions gen 2 version automatically deploys the cloud function on Google Cloud Run as a service under the same name as the Cloud
Functions. The initial version of this code was created to quickly demonstrate the role of MLOps in the ELP process and to create an MVP. Later,
this code will be optimized, and Python OOP concepts will be introduced to increase the code reusability and efficiency.
____________________________________________________________________________________________________________
Development Platform                | Developer       | Reviewer   | Release  | Version  | Date
____________________________________|_________________|____________|__________|__________|__________________
Google Cloud Serverless Computing   | DMV Consultant  | Ajay Gupta | Initial  | 1.0      | 09/18/2022

-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import streamlit as vAR_st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


def Regression_Model():

    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Training Data')

    with col4:
        vAR_st.write('')
        vAR_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="train")
        
    if vAR_dataset is not None:
        vAR_df = pd.read_csv(vAR_dataset)
        
        Preview_Data(vAR_df,"Training")
              
        Statistics_Details(vAR_df)
        
        Feature_Selection(vAR_df)
        
        vAR_model = Model_Implementation(vAR_df)
        
        
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Upload Test Data')

        with col4:
            vAR_st.write('')
            vAR_test_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="test")
            
        if vAR_test_dataset is not None:   
            vAR_test_data = pd.read_csv(vAR_test_dataset)
             
            # Preview Test Data
                
            Preview_Data(vAR_test_data,"Test")
            
            
            
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader('Test Model')

            with col4:
                vAR_st.write('')
                vAR_st.write('')
                vAR_test = vAR_st.button("Test Model")
                
        
                
            if vAR_test:
                # Preprocessing
                data_encoded = pd.get_dummies(vAR_test_data, columns=['Vehicle_Type'], drop_first=True)
                
                vAR_model = Train_Model(vAR_df)
                
                col1,col2,col3 = vAR_st.columns([3,15,1])
                
                with col2:
                    
                
                    vAR_test_data["Predicted_Score"] = vAR_model.predict(data_encoded)
                    
                    vAR_st.write(vAR_test_data)
                    
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
                with col4:
                    
                    vAR_result_data = vAR_test_data.to_csv().encode('utf-8')
                    
                    vAR_st.download_button(
    label="Download Model Outcome as CSV",
    data=vAR_result_data,
    file_name='Model Outcome.csv',
    mime='text/csv',
)
                    
                    
                    
                    
                    
                
            
            
        
    
        
        
        
                    
            
                
                
            
def Preview_Data(vAR_df,type):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Preview '+type +' Data')
        

    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_preview_data = vAR_st.button("Preview "+type+" Data")
        
        
    if vAR_preview_data:
        
        if type=="Test":
            col1,col2,col3 = vAR_st.columns([3,10,1.5])
        else:
            col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.dataframe(data=vAR_df,height=310)
            # vAR_st.info(" Note: Risk Score Calculated Based on Feature Weightage")
            
        
            
        
            


def Statistics_Details(vAR_df):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Stats on Training Data')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_data_stats = vAR_st.button("View Training Data Stats")
                
    if vAR_data_stats:
        
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        vAR_st.write('')
        vAR_st.write('')
        
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write(vAR_df.describe(),height=200)
        


def Feature_Selection(vAR_df):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_columns = list(vAR_df.columns)
    
    if "Risk_Score" in vAR_columns:
        vAR_columns.remove("Risk_Score")
    if "Normalized_Risk_Score" in vAR_columns:
        vAR_columns.remove("Normalized_Risk_Score")
    
    
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Feature Selection')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_features = vAR_st.multiselect('',vAR_columns)
        vAR_st.write('')
        vAR_st.write('')
        
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        vAR_st.write('')
        with vAR_st.expander("List selected features"):
            for i in range(0,len(vAR_features)):
                vAR_st.write('Feature',i+1,':',vAR_features[i])
        


def Model_Implementation(vAR_df):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        
        vAR_st.subheader('Model Training')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
       
        vAR_model_train = vAR_st.button("Train the Model")
        vAR_st.write('')
        
        vAR_st.write('')
        
                
    if vAR_model_train:
        col1,col2,col3 = vAR_st.columns([3,10,1.5])
        with col2:
            vAR_st.info("Data Preprocessing Completed!")
            vAR_st.info("Regression Model Successfully Trained")
            
            
        vAR_model = Train_Model(vAR_df)
        
        
        return vAR_model
    
    
    
    
    
    
def Train_Model(vAR_df):
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        # Preprocessing
        data_encoded = pd.get_dummies(vAR_df, columns=['Vehicle_Type'], drop_first=True)
        
        

        # Define features and target
        X = data_encoded.drop('Normalized_Risk_Score', axis=1)
        X = X.drop('Risk_Score', axis=1)
        y = data_encoded['Normalized_Risk_Score']

        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42)
        
        
        # print('X_test - ',X_test.iloc[0])
        


        # Training the Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        
        # y_pred = model.predict([[ 7,  9, 15, 56, 80,  1]])
        
        # vAR_st.write(y_pred)

        
    return model
