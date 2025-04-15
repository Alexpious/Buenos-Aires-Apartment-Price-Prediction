
import os
import pandas as pd

from glob import glob
from category_encoders import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def wrangle(filepath):
    
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighbourhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    #Drop high null value column
    df.drop(columns = ['floor' , 'expenses'], inplace = True)
    
    #High and low cardinality
    df.drop(columns = ['operation' , 'property_type' , 'currency' , 'properati_url'], inplace = True)
    
    #Drop Leakage
    df.drop(columns = ['price','price_aprox_local_currency','price_per_m2', 'price_usd_per_m2'], inplace = True)
    
    #Drop column with multicollinearity
    df.drop(columns = ['surface_total_in_m2', 'rooms'], inplace = True)
    

    
    return df

static_dir = os.path.join(BASE_DIR, 'static')
files = glob(os.path.join(static_dir, 'buenos-aires-real-estate-*.csv'))

frames = []
for file in files:
    frames.append(wrangle(file))

df = pd.concat(frames)


y_train = df["price_aprox_usd"]
X_train = df.drop(columns=["price_aprox_usd"])


model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)

def test_model(surface_covered_in_m2, lat, lon, neighbourhood):
    
    prediction = {
        "surface_covered_in_m2": surface_covered_in_m2,
        "lat": lat,
        "lon": lon,
        "neighbourhood": neighbourhood
    }
    df_prediction = pd.DataFrame(prediction, index=[0])
    predicted = model.predict(df_prediction)[0]
    return round(predicted, 2)