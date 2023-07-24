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







import os
import streamlit as vAR_st
import json
import pandas as pd




from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value






        



def DriverRiskPrediction():
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_conviction = vAR_st.number_input("Conviction Points",min_value=0,max_value=20,help="Think of this like the number of times a person has been caught doing something wrong while driving. Like running a red light, or not stopping at a stop sign. The more conviction points, the riskier the driver.")
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_experience = vAR_st.number_input("Years of Experience",min_value=0,max_value=50,help="This tells us how long the person has been driving. Usually, people who have been driving longer are less risky, because they have more experience on the road.")
        
        vAR_st.write('')
        vAR_st.write('')
        
        vAR_age = vAR_experience = vAR_st.number_input("Age",min_value=18,max_value=70)
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_geo_risk = vAR_st.number_input("Geo Risk",min_value=0,max_value=10,help="This is a number that tells us how risky the area where the driver usually drives is. Maybe it's a busy city with lots of traffic, or a quiet countryside road. The higher the Geo Risk number, the more risky the area, and potentially the driver too.")
        
        vAR_st.write('')
        vAR_st.write('')
        
        vAR_vehicle = vAR_st.selectbox("Select Vehicle Type",("Car","MotorCycle","Heavy Duty Vehicle"))
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        
        vAR_submit = vAR_st.button("Predict Driver Risk Score")
        
    if vAR_submit:
        vAR_payload = {"Conviction_Points":str(vAR_conviction),
        "Geo_Risk":str(vAR_geo_risk),
        "Years_of_Experience":str(vAR_experience),
        "Vehicle_Type":vAR_vehicle,
        "Age":str(vAR_age)}
        vAR_risk_score = RiskScorePredictionAPI(os.environ["PROJECT_ID"],os.environ["REGRESSION_ENDPOINT_ID"],vAR_payload)
        
        col1,col2,col3 = vAR_st.columns([4,15,4])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            
            vAR_st.subheader('Driver Risk Score (Predicted by Vertex AI AutoML)')
            vAR_st.write('')
            vAR_st.write('')
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            
            vAR_st.write(vAR_risk_score)
            
        
        








def RiskScorePredictionAPI(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-west1",
    api_endpoint: str = "us-west1-aiplatform.googleapis.com",
):
    vAR_result = {}
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response - ",response)
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_regression_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    
    print('predictions - ',predictions)
    
    for prediction in predictions:
        print(" prediction:", dict(prediction))
        vAR_result = dict(prediction)
        
    return vAR_result
        