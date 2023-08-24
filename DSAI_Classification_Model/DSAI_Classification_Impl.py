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
import shap

from lime.lime_text import LimeTextExplainer
from lime import lime_tabular
import streamlit.components.v1 as components
import base64


def Classification_Model():
    
    # if "vAR_model" not in vAR_st.session_state: 
    #     vAR_st.session_state = {}

    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Training Data')

    with col4:
        vAR_st.write('')
        vAR_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="classify")
        
    if vAR_dataset is not None:
        vAR_df = pd.read_csv(vAR_dataset)
        vAR_test=False
        vAR_tested = False
        
        Preview_Data(vAR_df,"Training")
              
        Statistics_Details(vAR_df)
        
        Feature_Selection(vAR_df)
        
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
            if "vAR_model" not in vAR_st.session_state and "X_train" not in vAR_st.session_state:
                vAR_st.session_state['vAR_model'],vAR_st.session_state['X_train'] = Train_Model(vAR_df)
            
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
            
            vAR_model = vAR_st.session_state['vAR_model']
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
                
                if "vAR_test_data" not in vAR_st.session_state:
                    vAR_st.session_state["vAR_test_data"] = vAR_test_data
                
                vAR_st.write(vAR_test_data)
                
                if "vAR_tested_log" not in  vAR_st.session_state:
                    vAR_st.session_state["vAR_tested_log"] = True
                print('vAR_test_data cols - ',vAR_st.session_state["vAR_test_data"].columns)
        print('vaR_tested clas- ',vAR_st.session_state["vAR_tested_log"])
        if vAR_st.session_state["vAR_tested_log"]:
                
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
            with col4:
                if vAR_st.session_state["vAR_test_data"] is not None:
                    vAR_st.markdown(create_download_button(vAR_test_data), unsafe_allow_html=True)
                
            vAR_st.write('')
            vAR_st.write('')
        
        if vAR_st.session_state["vAR_tested_log"]:
                    
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
            with col4:
                vAR_outcome_analysis = vAR_st.button("Model Outcome Analysis")
                    
                
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                    
            if vAR_outcome_analysis:
                
                
                df = vAR_st.session_state["vAR_test_data"].drop(["Vehicle_Type"],axis=1)
                
                print('outcome analysis test data cols1 - ',df.columns)
                
                print('outcome analysis test data cols2 - ',vAR_st.session_state["vAR_test_data"].columns)
                
            
                col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
                with col2:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.info("Correlation Between Features")
                    plot_correlation_matrix(df)
                    
                    
                    
                with col4:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.info("Distribution of Predicted Level(High)")
                    plot_score_distribution(df)
                    
                col1,col2,col3 = vAR_st.columns([1,15,1])
                
                with col2:
                    
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.markdown("<div style='text-align: center; color: black;'>Risk Level Variation with Features in ScatterPlot</div>", unsafe_allow_html=True)
                    vAR_st.write('')
                    vAR_st.write('')
                    plot_scatter_matrix(vAR_st.session_state["vAR_test_data"])
                
                
                
                
                    
                    
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader("Select Driver ID For XAI")
                
                
                
            with col4:
                vAR_st.write('')
                vAR_st.write('')
                
                vAR_idx_values = ['Select Driver Id'] 
                vAR_idx_values.extend(["DL10"+str(item) for item in vAR_test_data.index])               
                vAR_test_id = vAR_st.selectbox(' ',vAR_idx_values)
                
                
            
            if vAR_test_id!='Select Driver Id': 
                # One-hot encoding for 'Vehicle_Type'
                one_hot = OneHotEncoder()
                vehicle_type_encoded = one_hot.fit_transform(vAR_test_data[['Vehicle_Type']]).toarray()
                vehicle_type_encoded_df = pd.DataFrame(vehicle_type_encoded, columns=one_hot.categories_[0])

                # Add the one-hot encoded variables to the dataset and remove the original 'Vehicle_Type' column
                data_encoded = pd.concat([vAR_test_data, vehicle_type_encoded_df], axis=1)
                data_encoded.drop(['Vehicle_Type'], axis=1, inplace=True)
                
                vAR_model,X_train = vAR_st.session_state["vAR_model"],vAR_st.session_state["X_train"]
                features = X_train.columns
                print('features - ',features)
                col1,col2,col3 = vAR_st.columns([1,15,1])
                
                with col2:
                    
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.markdown("<div style='text-align: center; color: black;font-weight:bold;'>Explainable AI with LIME(Local  Interpretable Model-agnostic Explanations) Technique</div>", unsafe_allow_html=True)
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write('')
                    ExplainableAI(X_train,features,vAR_model,data_encoded,vAR_test_id)
                    # SHAPExplainableAI(X_train,vAR_model,data_encoded,vAR_test_id)
                    
                    
                    
                    
                
            
            
        
    
        
        
        
                    
            
                
                
            
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
    
    vAR_columns =["All"]
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_columns.extend(list(vAR_df.columns))
    
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
        vAR_features = vAR_st.multiselect(' ',vAR_columns,default="All")
        vAR_st.write('')
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
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    with col2:
        vAR_st.info("Data Preprocessing Completed!")
        vAR_st.info("Regression Model Successfully Trained")

    # Create a Logistic Regression object
    log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)

    # Train the model
    log_reg.fit(X_train_scaled, y_train)
    
    return log_reg,X_train



# Model Outcome Analysis Functions
    
def plot_correlation_matrix(data):
    print('df len - ',len(data))
    correlation_matrix = data.corr()
    plt.figure(figsize=(8, 8))
    # plt.title("Correlation Between Features")
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    vAR_st.pyplot(plt)
    
    
def plot_score_distribution(data):
    print('data cols inplot - ',data.columns)
    sns.displot(data,x="High")
    # plt.title(f'Distribution of Predicted_Score')
    vAR_st.pyplot(plt)
    
    
def plot_scatter_matrix(df):
    sns.pairplot(df,hue='Vehicle_Type',kind='scatter')
    vAR_st.pyplot(plt)

def ExplainableAI(X_train,features,vAR_model,vAR_test_data,test_id):
    
    print('train type - ',type(X_train))
    print('X_train - ',X_train.columns)
    print('vAR_test_data - ',type(vAR_test_data))
    print('dtypes - ',vAR_test_data.dtypes)
    print('test cols - ',vAR_test_data.columns)
    explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, 
                                              feature_names=features, 
                                              class_names=['Low', 'Medium', 'High'], 
                                              mode='classification')
    
    exp = explainer_lime.explain_instance(
    vAR_test_data.iloc[int(test_id[4:])], vAR_model.predict_proba,labels=[0,1,2],top_labels=1)
    
    # Extract LIME explanations and visualize
    features, weights = zip(*exp.as_list())
    chart_data = pd.DataFrame({'Feature': features, 'Weight': weights})
    print(chart_data.head())
    
    
    
    vAR_st.dataframe(chart_data)
    
    vAR_st.write('')
    
    html = exp.as_html()
    components.html(html, height=1000,width=1000)
    
    
    
    

    
def SHAPExplainableAI(X_train,vAR_model,X_test_data,test_id):
    
    # SHAP
    explainer = shap.LinearExplainer(vAR_model, X_train)
    shap_values = explainer.shap_values(X_test_data)
    
    # Force plot for selected instance
    shap.initjs()  # Required for visualizations
    plt.figure(figsize=(20, 3))
    shap.force_plot(explainer.expected_value, shap_values[test_id], X_test_data.iloc[test_id])
    vAR_st.pyplot(plt.gcf())  # Display the current figure

    # Summary plot for global explanations
    vAR_st.write("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test_data, show=False)  # Set `show=False` so it doesn't display immediately
    vAR_st.pyplot(plt.gcf())  # Display the current figure
    
    
    
    
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