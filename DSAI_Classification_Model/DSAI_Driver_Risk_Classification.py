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
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import shap
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lime import lime_tabular
import streamlit.components.v1 as components
import base64
import sweetviz
from DSAI_Data_Read_Bigquery.DSAI_Read_Training_Data import ReadTrainingData

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



def DriverRiskClassification():
    
    # if "vAR_model" not in vAR_st.session_state: 
    #     vAR_st.session_state = {}
    
    # try:

    vAR_df = None
    vAR_test = None

    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_result_data = pd.DataFrame()
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Training Data')

    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        # vAR_dataset = vAR_st.button("Read Bigquery Table")
        vAR_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="reg")
        
    if vAR_dataset is not None:
        
        
        # vAR_df = ReadTrainingData()
        vAR_df = pd.read_csv(vAR_dataset)
        print('len - ',len(vAR_df))
        
        
        if "training_data" not in vAR_st.session_state:
            # vAR_df.fillna(0,inplace=True)
            # print('info - ',vAR_df.info())
            
            # vAR_df["dtDiss"].fillna("1800-01-01",inplace=True)
            # vAR_df["Sec3"].fillna("XX",inplace=True)
            # vAR_df["Sec4"].fillna("XX",inplace=True)
            # vAR_df["Sec5"].fillna("XX",inplace=True)
            # vAR_df["Sec6"].fillna("XX",inplace=True)
            
            # vAR_df["Sec7"].fillna(0,inplace=True)
            # vAR_df["Sec8"].fillna(0,inplace=True)
            # vAR_df["DISM_CORR_IND"].fillna("XX",inplace=True)
            # vAR_df["FTA_DESTR_IND"].fillna(0,inplace=True)
            # vAR_df.FTA_DESTR_IND = vAR_df.FTA_DESTR_IND.astype(float)
            # vAR_df["FTP_IND"].fillna(0,inplace=True)
            # vAR_df.FTP_IND = vAR_df.FTP_IND.astype(float)
            # vAR_df['LABELS'] = vAR_df['LABEL'].replace({'LOW-SEVERITY': 0, 'INJURY CRASH': 1, 'FATAL CRASH': 2})
            vAR_st.session_state["training_data"] = vAR_df
        
        
    if vAR_st.session_state["training_data"] is not None:
        
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Preview Data')
            

        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_preview_data = vAR_st.button("Preview Data")
            
        
        
        if vAR_preview_data:
            col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
            with col2:
                
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.dataframe(data=vAR_st.session_state["training_data"])
                
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        
        vAR_st.write('')
        vAR_st.write('')
        
        with col4:
            vAR_st.write('')
            vAR_st.write('')
            vAR_eda = vAR_st.button("Exploratory Data Analysis")
            
        
        if vAR_eda:
            col1,col2,col3 = vAR_st.columns([0.4,10,1])
            with col2:
                vAR_analysis = sweetviz.analyze(vAR_st.session_state["training_data"],pairwise_analysis="on")
                # vAR_analysis.show_html('DataAnalysis.html')   
                # vAR_analysis.show_notebook()
                
                # raw_html = html_object._repr_html_()

                # components.v1.html(vAR_analysis.show_notebook()._repr_html_())
                # vAR_analysis.show_html(filepath=r'C:\Users\ds_007\Desktop\DMV_Driver_Risk_Prediction\DataAnalysis.html', open_browser=False, layout='vertical', scale=1.0)
                
                vAR_analysis.show_html(filepath='/tmp/DataAnalysis.html', open_browser=False, layout='vertical', scale=1.0)
                
                # with open(r'C:\Users\ds_007\Desktop\DMV_Driver_Risk_Prediction\DataAnalysis.html', 'r') as f:
                with open('/tmp/DataAnalysis.html', 'r') as f:
                    raw_html = f.read().encode("utf-8")
                    # raw_html = base64.b64encode(raw_html).decode()
                src = f"data:text/html;base64,{raw_html}"
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            with col4:
                vAR_st.download_button(label="Download EDA Report",
                            data=raw_html,
                            file_name="DriverRiskAnalysis.html",mime="text/html")
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                # components.iframe(src=src, width=1100, height=2500, scrolling=True)
        if vAR_st.session_state["training_data"] is not None:
            
            Feature_Selection(vAR_st.session_state["training_data"])
            
            col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.info('**Note : We took only potential features based on EDA.**')
            
            
        # Model Training
        if vAR_st.session_state["training_data"] is not None:
            
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
                if "vAR_model" not in vAR_st.session_state and "X_train" not in vAR_st.session_state and "X_train_cols" not in vAR_st.session_state:
                    vAR_st.session_state['vAR_model'],vAR_st.session_state['X_train'],vAR_st.session_state['X_train_cols'] = Train_Model(vAR_st.session_state["training_data"])

            # Model Testing
            
            if vAR_st.session_state['vAR_model'] is not None:
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
                    
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                with col2:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.subheader('Preview Data')
                    

                with col4:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_preview_data = vAR_st.button("Preview Data",key="test_preview")
                    
                
                
                if vAR_preview_data:
                    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
                    with col2:
                        
                        vAR_st.write('')
                        vAR_st.write('')
                        vAR_st.dataframe(data=vAR_test_data)
                
                
                
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
                vAR_df_columns = vAR_test_data.columns
            
                vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 
                
                vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
                
                vAR_model = vAR_st.session_state['vAR_model']
                
                data_encoded = pd.get_dummies(vAR_test_data, columns=vAR_categorical_column)
                
                                
                # Later this can be removed
                cols = ['SEC1_A12500D', 'SEC1_A14600A', 'SEC1_A21455', 'SEC1_A21710', 'SEC1_A22348B', 'SEC1_A22406A', 'SEC1_A23109C', 'SEC1_A23152', 'SEC1_A23153A', 'SEC1_A23153B', 'SEC1_A27156B', 'SEC1_A27315E', 'SEC1_A4000A', 'SEC1_A4000A1', 'SEC1_A5200A', 'SEC1_C11550A']
                
                for col in cols:
                    data_encoded[col] = [False]*len(data_encoded)
                
                data_encoded = data_encoded[vAR_st.session_state['X_train_cols']]
                
                print('vAR_numeric_columns test- ',vAR_numeric_columns)

                print('vAR_categorical_column test- ',vAR_categorical_column)
                
                print('data_encoded cols test- ',data_encoded.columns)
                
                # Logistic Regression requires feature scaling, so let's scale our features
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(data_encoded)
                            
                col1,col2,col3 = vAR_st.columns([3,15,1])
                
                with col2:
                    
                
                    # Predict probabilities on the test data
                    y_pred_proba_log_reg = vAR_model.predict_proba(X_test_scaled)
                    
                    print('y_pred_proba_log_reg - ',y_pred_proba_log_reg)
                    
                    print('y_pred_proba_log_reg type- ',type(y_pred_proba_log_reg))

                    # Convert to DataFrame for better visualization
                    y_pred_proba_log_reg_df = pd.DataFrame(y_pred_proba_log_reg, columns=['INJURY CRASH','LOW-SEVERITY'])
                    
                    print('y_pred_proba_log_reg_df - ',y_pred_proba_log_reg_df)
                    

                    
                    vAR_test_data["INJURY CRASH"] = y_pred_proba_log_reg_df["INJURY CRASH"]
                    
                    vAR_test_data["LOW-SEVERITY"] = y_pred_proba_log_reg_df["LOW-SEVERITY"]
                    
                    
                    if "vAR_test_data" not in vAR_st.session_state:
                        vAR_test_data['LABEL'] = vAR_test_data[['INJURY CRASH', 'LOW-SEVERITY']].where(lambda x: x > 0.5).idxmax(axis=1)
                        vAR_st.session_state["vAR_test_data"] = vAR_test_data
                    
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write(vAR_test_data)
                    
                    if "vAR_tested_log" not in  vAR_st.session_state:
                        vAR_st.session_state["vAR_tested_log"] = True
            if vAR_st.session_state["vAR_tested_log"]:
                    
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
                with col4:
                    if vAR_st.session_state["vAR_test_data"] is not None:
                        vAR_st.markdown(create_download_button(vAR_test_data), unsafe_allow_html=True)
                        
                        vAR_st.write('')
                        vAR_st.write('')
                        
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
                with col2:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.subheader('Model Outcome')

                with col4:
                    vAR_st.write('')
                    vAR_model_outcome_graph = vAR_st.button("Model Outcome Summary")
                    
                if vAR_model_outcome_graph:
                    
                    ModelOutcomeSummary(vAR_st.session_state["vAR_test_data"])
                    
                        
                        
                
            
    
    
    # except BaseException as e:
    #     print('Error - ',str(e))
                    
                    
                    
                
                
            
            
        
    
            
                
            
def Preview_Data(vAR_df):
    
    print('len in preview - ',len(vAR_df))
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Preview Data')
        

    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_preview_data = vAR_st.button("Preview Data")
        
        
    if vAR_preview_data:
        
        
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.dataframe(data=vAR_df,height=310)
            # vAR_st.info(" Note: Risk Score Calculated Based on Feature Weightage")
            
        

def Feature_Selection(vAR_df):
    
    vAR_columns =["All"]
    vAR_potential_features = ["DRIVER_AGE","SEC1","COURT","DISM_CORR_IND","RES_COUNTY","YEARS_OF_EXP",
                              "DACTYPE","CRASH_TIME","NO_OF_INJURIES","NO_OF_FATALS","SOBRIETY","PHYS_COND","CITED"]
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_columns.extend(vAR_potential_features)
    
    with col2:
        
        vAR_st.write('')
        vAR_st.subheader('Feature Selection')
        
    with col4:
        
        vAR_features = vAR_st.multiselect(' ',vAR_columns,default="All")
        vAR_st.write('')
        
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        vAR_st.write('')
        with vAR_st.expander("List selected features"):  
            if 'All' in vAR_features:
                vAR_st.write('Features:',vAR_columns[1:])
            else:
                for i in range(0,len(vAR_features)):
                    vAR_st.write('Feature',i+1,':',vAR_features[i])
                    
                    
                    
       
        
            
def Train_Model(vAR_df):
    
    vAR_df = vAR_df[["DRIVER_AGE","SEC1","COURT","DISM_CORR_IND","RES_COUNTY","YEARS_OF_EXP",
                              "DACTYPE","CRASH_TIME","NO_OF_INJURIES","NO_OF_FATALS","SOBRIETY","PHYS_COND","CITED","LABEL"]].copy()
    
    vAR_train_df = vAR_df.drop(vAR_df.columns[-1],axis=1)
    
    vAR_df_columns = vAR_train_df.columns
        
    vAR_numeric_columns = vAR_train_df._get_numeric_data().columns 
    
    vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
    
    data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
    
    print('vAR_numeric_columns - ',vAR_numeric_columns)
    
    print('vAR_categorical_column - ',vAR_categorical_column)
    
    print('data_encoded cols - ',data_encoded.columns)
    
    vAR_data_encoded_cols = data_encoded.columns

    # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
    # data_encoded.drop(vAR_categorical_column, axis=1, inplace=True)

    # Label encoding for 'Crash_Level'
    label_enc = LabelEncoder()
    data_encoded['Crash_Level'] = label_enc.fit_transform(data_encoded[data_encoded.columns[-1]])

    # Split the data into features (X) and target (y)
    X = data_encoded.drop(data_encoded.columns[-1],axis=1)
    y = vAR_df.iloc[: , -1:]
    
    
    
    print('X cols - ',X.columns)
    print('y cols - ',y.columns)
    
    print('data_encoded Crash Level - ',data_encoded['Crash_Level'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42)

    
    

    # Logistic Regression requires feature scaling, so let's scale our features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    with col2:
        vAR_st.info("Data Preprocessing Completed!")
        vAR_st.info("Classification Model Successfully Trained")

    # Create a Logistic Regression object
    log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)

    # Train the model
    log_reg.fit(X_train_scaled, y_train)
    
    print('LABELS - ',log_reg.classes_)
    
    return log_reg,X_train,vAR_data_encoded_cols



def dataframe_to_base64(df):
    """Convert dataframe to base64-encoded csv string for downloading."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

def create_download_button(df, filename="data.csv"):
    """Generate a link to download the dataframe as a csv file."""
    b64_csv = dataframe_to_base64(df)
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="{filename}" style="display: block; margin: 1em 0; padding: 13px 20px 16px 12px; background-color: rgb(47 236 106); text-align: center; border: none; border-radius: 6px;color: black; text-decoration: none;">Download Model Outcome as CSV</a>'
    return href


def ModelOutcomeSummary(vAR_test_data):
    frequencies = vAR_test_data['LABEL'].value_counts()
    
    fig, ax = plt.subplots()
    bars = ax.bar(frequencies.index, frequencies.values, color=['blue', 'grey', 'red'])  # adjust colors as needed
    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency')
    ax.set_title('Model Outcome Summary - Frequency of Crash Levels')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + 0.5,  # Adjust this value for the position of the count
                '%d' % int(height),
                ha='center', va='bottom')
    
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.pyplot(fig)

    
    