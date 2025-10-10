# predictor/views.py
import os
import joblib
from django.conf import settings
import json
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load the model when the server starts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'house_price_model.joblib')
COLUMNS_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'training_columns.json')

# MODEL_PATH = 'houseprice_project/predictor/ml_models/house_price_model.joblib'
model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH, 'r') as f:
    training_columns = json.load(f)

@api_view(['POST'])
def predict_price(request):
    """
    API endpoint to predict house prices.
    """
    try:
        # 1. Get data from the request
        data = request.data
        
        input_df = pd.DataFrame([data])

        # Apply log transforms (use the same columns as in training)
        cols_to_log = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_basement', 'yr_built']
        for col in cols_to_log:
            input_df[col] = np.log(input_df[col].astype(float) + 1)
            
        # Feature Engineering (must match training)
        input_df['living_quality'] = input_df['view'] + (input_df['grade'] / 2) + input_df['condition']
        
        # Drop the original columns
        input_df = input_df.drop(columns=['view', 'condition', 'grade'])

        ## reordering the columns to match training columns
        input_df = input_df[training_columns]
        # 3. Make prediction
        prediction = model.predict(input_df)
        
        # The output of predict is a numpy array, so we get the first element
        predicted_price = prediction[0]

        
        # 4. Return the response
        return Response({'predicted_price': predicted_price})

    except Exception as e:
        return Response({'error': str(e)}, status=400)