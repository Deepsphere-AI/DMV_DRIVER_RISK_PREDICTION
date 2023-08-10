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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def Classification_Model():

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
        
        # Not able plot the Variable distribution
        ####################################################################################################################
        # Error - UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure
        
        # Feature_Distribution(vAR_df, ['Conviction_Points', 'Geo_Risk', 'Years_of_Experience', 'Age'], 2, 2)

        # # Plot count plot for categorical features
        # sns.countplot(x='Vehicle_Type', data=vAR_df)
        # plt.show()

        # sns.countplot(x='Risk_Level', data=vAR_df)
        # plt.show()

        ######################################################################################################################
        
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
                
                
                # One-hot encoding for 'Vehicle_Type'
                one_hot = OneHotEncoder()
                vehicle_type_encoded = one_hot.fit_transform(vAR_test_data[['Vehicle_Type']]).toarray()
                vehicle_type_encoded_df = pd.DataFrame(vehicle_type_encoded, columns=one_hot.categories_[0])

                # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
                data_encoded = pd.concat([vAR_test_data, vehicle_type_encoded_df], axis=1)
                data_encoded.drop(['Vehicle_Type'], axis=1, inplace=True)

               
                # Logistic Regression requires feature scaling, so let's scale our features
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(data_encoded)
                
                
                
                vAR_model = Train_Model(vAR_df)
                
                col1,col2,col3 = vAR_st.columns([3,15,1])
                
                with col2:
                    
                
                    # Predict probabilities on the test data
                    y_pred_proba_log_reg = vAR_model.predict_proba(X_test_scaled)
                    
                    print('y_pred_proba_log_reg - ',y_pred_proba_log_reg)
                    
                    print('y_pred_proba_log_reg type- ',type(y_pred_proba_log_reg))

                    # Convert to DataFrame for better visualization
                    y_pred_proba_log_reg_df = pd.DataFrame(y_pred_proba_log_reg, columns=['High','Low','Medium'])
                    
                    print('y_pred_proba_log_reg_df - ',y_pred_proba_log_reg_df)
                    

                    
                    vAR_test_data["High"] = y_pred_proba_log_reg_df["High"]
                    
                    vAR_test_data["Medium"] = y_pred_proba_log_reg_df["Medium"]
                    
                    vAR_test_data["Low"] = y_pred_proba_log_reg_df["Low"]
                    
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
        


def Feature_Distribution(vAR_df,features,rows,cols):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Variable Distribution')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_plot = vAR_st.button("Plot Variable Distribution")
                
    if vAR_plot:

        # Define a function to plot histograms for numerical features
        fig=plt.figure(figsize=(20,20))
        for i, feature in enumerate(features):
            ax=fig.add_subplot(rows,cols,i+1)
            vAR_df[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
            ax.set_title(feature+" Distribution",color='DarkRed')
            
        fig.tight_layout()  
        plt.show()




def Feature_Selection(vAR_df):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_columns = list(vAR_df.columns)
    
    if "Risk_Level" in vAR_columns:
        vAR_columns.remove("Risk_Level")
    
            
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
            vAR_st.info("Classification Model Successfully Trained")
            
            
        vAR_model = Train_Model(vAR_df)
        
        
        return vAR_model
    
    
    
    
    
    
def Train_Model(vAR_df):
    
    

    # One-hot encoding for 'Vehicle_Type'
    one_hot = OneHotEncoder()
    vehicle_type_encoded = one_hot.fit_transform(vAR_df[['Vehicle_Type']]).toarray()
    vehicle_type_encoded_df = pd.DataFrame(vehicle_type_encoded, columns=one_hot.categories_[0])

    # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
    data_encoded = pd.concat([vAR_df, vehicle_type_encoded_df], axis=1)
    data_encoded.drop(['Vehicle_Type'], axis=1, inplace=True)

    # Label encoding for 'Risk_Level'
    label_enc = LabelEncoder()
    data_encoded['Risk_Level'] = label_enc.fit_transform(data_encoded['Risk_Level'])

    # Split the data into features (X) and target (y)
    X = data_encoded.drop(['Risk_Level'], axis=1)
    y = data_encoded['Risk_Level']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42)

    X_train.head(), y_train.head()
    
    

    # Logistic Regression requires feature scaling, so let's scale our features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a Logistic Regression object
    log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)

    # Train the model
    log_reg.fit(X_train_scaled, y_train)
    
    return log_reg

    

