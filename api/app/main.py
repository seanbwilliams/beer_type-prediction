from typing import List
from fastapi import FastAPI, Query
from pydantic import BaseModel
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from enum import Enum

# Setup a variable to introduce and describe the
projectDescription = """
The **Beer Type Prediction Project** uses a trained Machine Learning Model to accurately predict a type of beer based on the following set of review criteria:

* **Brewery Name** (_The name of the brewery producing the beer_).
* **Aroma Review Score** (_The review score given to the beer regarding the aroma of the beer (on a scale from 0 to 5)_).
* **Appearance Review Score** (_The review score given to the beer regarding the appearance of the beer (on a scale from 0 to 5)_).
* **Palate Review Score** (_The review score given to the beer regarding the palate of the beer (on a scale from 0 to 5)_).
* **Taste Review Score** (_The review score given to the beer regarding the taste of the beer (on a scale from 0 to 5)_).
* **Beer ABV** (_The alcohol by volume measurement of the beer_).

### API Endpoints
This API provides the following list of endpoints that provide API status reports and interactions with the Machine Learning model:

**Status**
* **/health** - _Check the Operational status of the API_.

**Predictions**
* **/beer/type** - _Predict the type of beer based on a single set of inputs_.
* **/beers/type** - _Predict the type of beer for one or more sets of inputs_.

Please refer to the **Parameters** and **Responses** section of each endpoint for details of the inputs and response(s) of the API

### Using the API
* Expand the endpoint you would like to interact with
* Click the "Try it Out" button (found on the right hand side in the **Parameters** section of each endpoint)
* Fill in all of the Required API Parameters with the **Parameters** section of each endpoint
* Click the "Execute" button (found at the bottom of the **Parameters** section of each endpoint)

### API Results
* The API response will be found in the Response Body of the **Reponses** section of each endpoint

### GitHub Repo Link
The source for this project can be found on GitHub at: [https://github.com/seanbwilliams/beer_type_prediction](https://github.com/seanbwilliams/beer_type_prediction)
"""

# Create a list containing API section tags and associated descriptions
tags_metadata = [
    {
        "name": "status",
        "description": "Provides API status operations."
    },
    {
        "name": "predictions",
        "description": "Provides interactions with the Machine Learning model"
    }
]

# Create a Class to enumerate review ratings
class ReviewRating(str, Enum):
    zero = "0"
    zerofive = "0.5"
    one = "1"
    onefive = "1.5"
    two = "2"
    twofive = "2.5"
    three = "3"
    threefive = "3.5"
    four = "4"
    fourfive = "4.5"
    five = "5"

# Create a Class to Model a single beer type response
class SingleResponse(BaseModel):
    beer_type: str

# Create a Class to Model a multiple beer type responses
class BeerModel(BaseModel):
    brewery_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    beer_abv: float
    beer_type: float    

# Create a Class to Model a multiple beer type responses
class MultiResponse(BaseModel):
    beers: List[BeerModel]

# Initialise the API with some description attributes
app = FastAPI(title="Beer Type Prediction",
              description=projectDescription,
              version="0.0.1",
              docs_url="/",
              redoc_url=None,
              openapi_tags=tags_metadata)

# Load Preprocessing Pipeline
preproc = load('../models/preproc_beer_type_prediction.joblib')

# Load Target Encoder
targetEncoder = load('../models/target_encoder.joblib')

# Load Pre-trained XGBoost Classifier
clf_xgb = load('../models/xgb_beer_type_prediction.joblib')

# Function to convert datatype of passed in objects
def convert(input):
    converters = [int, float]
    for converter in converters:
        try:
            return converter(input)
        except (TypeError, ValueError):
            pass
    return input

# Function to format features
def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float, single_multi: str = 'SINGLE'):
    if single_multi == 'SINGLE':
        return {
            'brewery_name': [brewery_name],
            'review_aroma': [review_aroma],
            'review_appearance': [review_appearance],
            'review_palate': [review_palate],
            'review_taste': [review_taste],
            'beer_abv': [beer_abv]
        }
    else:
        return {
            'brewery_name': brewery_name,
            'review_aroma': review_aroma,
            'review_appearance': review_appearance,
            'review_palate': review_palate,
            'review_taste': review_taste,
            'beer_abv': beer_abv
        }

# Define the Health endpoint
@app.get('/health', status_code=200, tags=["status"], summary="Check the Operational status of the API")
def healthcheck():
    return 'The Beer Type Prediction Project is ready to go'

# Define the Single Beer Review endpoint
@app.get("/beer/type/", tags=["predictions"], response_model=SingleResponse, summary="Predict the type of beer based on a single set of inputs")
def predict(brewery_name: str, review_aroma: ReviewRating, review_appearance: ReviewRating, review_palate: ReviewRating, review_taste: ReviewRating, beer_abv: float):
    """
    This API provides a beer type prediction for a single beer review. The API accepts 6 individual parameters for each of the review criteria listed below.
    
    brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv
    """    
    # Format features
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    # Create a Pandas Dataframe of Observations
    df_rawobs = pd.DataFrame(features)
    # Preprocess dataframe data
    procobs = preproc.transform(df_rawobs)
    # Make Predictions on data
    pred = clf_xgb.predict(procobs)
    # Get the Label corresponding to the prediction
    predenc = targetEncoder.inverse_transform(pred)
    # Create a response from the predictions
    responseDict = {"beer_type": beer_type for beer_type in predenc}
    # Return the response
    return JSONResponse(responseDict)

# Define the Multiple Beer Reviews endpoint
@app.get("/beers/type/", tags=["predictions"], response_model=MultiResponse, summary="Predict the type of beer for one or more sets of inputs")
def predict(beers: List[str] = Query(..., description="A comma separated string of values representing a single beer review. This endpoint accepts multiple beer reviews")):
    """
    This API provides beer type predictions for one or more beer reviews. Each review is provided to the API in the form of a comma seperated string of values that take the following form:
    
    {brewery_name},{review_aroma},{review_appearance},{review_palate},{review_taste},{beer_abv}

    e.g. Caldera Brewing Company,2.5,2,1.5,5,5.8

    Click on the "Add string item" button (found in the **Parameters** section of the endpoint) to supply one or more beer reviews to the API or the "-" (minus) button to remove a beer review
    """    
    # Create a list to store passed in observations
    featuresList = []
    # Iterate over passed in parameters, split, convert to appropriate datatypes and append to List
    for beer in beers:
        # Convert the CSV String to a List with correct datatypes
        beerList = [convert(x) for x in beer.split(",")]
        # Create a dictionary of features
        features = format_features(beerList[0], beerList[1], beerList[2], beerList[3], beerList[4], beerList[5], 'MULTI')
        # Append the dictionary to the features list
        featuresList.append(features)
    # Create a Pandas Dataframe of Observations
    df_rawobs = pd.DataFrame(featuresList)
    # Preprocess dataframe data
    procobs = preproc.transform(df_rawobs)
    # Make Predictions on data
    preds = clf_xgb.predict(procobs)
    # Get the Labels corresponding to the predictions
    predsenc = targetEncoder.inverse_transform(preds)
    # Create a response from the predictions
    for idx, predenc in enumerate(predsenc):
        featuresList[idx].update({"beer_type": predenc})
    # Return the response
    return JSONResponse({"beers": featuresList})