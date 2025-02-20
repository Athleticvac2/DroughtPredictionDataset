import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import when,col


drought_thresholds = {
    'Rainfall_mm': 70,
    'Soil_Moisture_Percent': 20,
    'Temperature_Celsius': 25
}
def calculate_drought_status(df):
    return df.withColumn(
        'Drought_Status',
        when(
            (col('Rainfall_mm') < drought_thresholds['Rainfall_mm']) &
            (col('Soil_Moisture_Percent') < drought_thresholds['Soil_Moisture_Percent']) &
            (col('Temperature_Celsius') < drought_thresholds['Temperature_Celsius']),
            1
        ).otherwise(0)
    )
class DroughtStatusCalculator(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, outputCol=None):
        super(DroughtStatusCalculator, self).__init__()
        # Initialize parameters if needed

    def _transform(self, df):
        return calculate_drought_status(df)

# Create a Spark session
spark = SparkSession.builder.appName("DroughtPredictionApp").getOrCreate()


model_path = "/user/athleticvac2/prediction_model/Drought_Prediction_Model"
loaded_model = PipelineModel.load(model_path)

st.title("Drought Prediction App")

st.header("Input data")
rainfall = st.number_input("Rainfall(mm)", min_value=0)
soil_moisture = st.number_input("Soil Moisture(%)", min_value=0)
temperature = st.number_input("Temperature(Celsius)", min_value=0)
crop = st.selectbox("Crop",["Maize","Coffee","Sweet Potatoes","Tea","Wheat","Sugarcane","Cowpeas","Millet","Pyrethrum","Cashewnuts","Mangoes","Cassava","Sorghum","Mangoes","Coconut","Green Grams","Potatoes"])
region = st.selectbox("Region", ["Central Highlands","Coast","Western Kenya","Eastern", "Rift Valley"])


input_data = [(0, crop, region, rainfall, soil_moisture, temperature)]
input_columns = ['id', 'Crop', 'Region', 'Rainfall_mm', 'Soil_Moisture_Percent', 'Temperature_Celsius']
input_df =  spark.createDataFrame(input_data, input_columns)

predictions = loaded_model.transform(input_df)
predicted_drought_status = predictions.select(F.col("prediction")).first()[0]

if predicted_drought_status == 1:
	st.write("Prediction: Drought is likely to occur.")
else:
	st.write("Prediction: Drought is not likely to occur.")


