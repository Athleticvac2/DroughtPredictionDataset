import calendar
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.connect.functions import date_format
from pyspark.sql.functions import when, col, to_date, date_format

spark = (SparkSession.builder.appName("DroughtPredictionApp")
        .config("spark.mongodb.input.uri","mongodb://localhost:27017/pyspark_DB.prediction")
        .config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
        .getOrCreate())
spark.conf.set('spark.sql.shuffle.partitions', '1')

class DroughtStatusCalculator(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, outputCol=None):
        super(DroughtStatusCalculator, self).__init__()

    def _transform(self, df):
        return calculate_drought_status(df)

model_path = "/user/athleticvac2/new_model_02"
loaded_model = PipelineModel.load(model_path)

new_df = spark.read.format("mongo").load()
print(new_df.printSchema())

drought_thresholds = {
    'Rainfall_mm': 30,
    'Soil_Moisture_Percent': 20,
    'Temperature_Celsius': 25
}

def calculate_drought_status(df):
    return df.withColumn(
        'Drought_Status',
        when(
            (col('Rainfall_mm') < drought_thresholds['Rainfall_mm']) &
            (col('Soil_Moisture_Percent') < drought_thresholds['Soil_Moisture_Percent']) &
            (col('Temperature_Celsius') > drought_thresholds['Temperature_Celsius']),
            1
        ).otherwise(0)
    )

new_df = calculate_drought_status(new_df)

# Ensure Date column is parsed as date type
new_df = new_df.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))

def main_page():
    st.title("🏜️ Drought Prediction For Kenya 🇰🇪")
    st.header("👥 Farmers may present their input data for the model to predict whether their conditions meet drought")
    rainfall = st.number_input("🌧️ Rainfall(mm)", min_value=0.0)
    soil_moisture = st.number_input("💧 Soil Moisture(%)", min_value=0.0)
    temperature = st.number_input("❄️ Temperature(Celsius)", min_value=0.0)
    crop = st.selectbox("🌽 Crop", ["Maize", "Coffee", "Sweet Potatoes", "Tea", "Wheat", "Sugarcane", "Cowpeas", "Millet", "Pyrethrum", "Cashewnuts", "Mangoes", "Cassava", "Sorghum", "Mangoes", "Coconut", "Green Grams", "Potatoes"])
    region = st.selectbox("🌍 Region", ["Central Highlands", "Coast", "Western Kenya", "Eastern", "Rift Valley"])

    input_data = [(0, crop, region, rainfall, soil_moisture, temperature)]
    input_columns = ['id', 'Crop', 'Region', 'Rainfall_mm', 'Soil_Moisture_Percent', 'Temperature_Celsius']
    input_df = spark.createDataFrame(input_data, input_columns)

    predictions = loaded_model.transform(input_df)
    predicted_drought_status = predictions.select(F.col("Drought_Status")).first()[0]

    logging.info(f"Input Data: {input_data}")
    logging.info(f"Predicted Drought Status: {predicted_drought_status}")

    if predicted_drought_status == 1:
        st.write(f"Prediction: Drought is likely to occur. You may not plant {crop}")
    else:
        st.write(f"Prediction: Drought is not likely to occur. You may plant {crop}")


def recommendations_page():
    st.title("Planting Recommendations with Charts")

    def get_planting_suitability(region):
        # Get average drought status for all crops in the selected region
        filtered_df = new_df.filter(new_df['Region'] == region)

        if filtered_df.count() == 0:
            return None

        # Extract month from the Date column and group by Crop and Month
        monthly_suitability = filtered_df.withColumn('Month', date_format(col('Date'), 'M')).groupBy('Crop',
                                                                                                     'Month').agg(
            F.mean('Drought_Status').alias('Average_Drought_Status'))

        return monthly_suitability

    def create_planting_chart(monthly_suitability):
        if monthly_suitability is None or monthly_suitability.count() == 0:
            st.write("No data available for the selected region.")
            return

        # Convert PySpark DataFrame to Pandas DataFrame
        monthly_suitability_pd = monthly_suitability.toPandas()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Month', y='Average_Drought_Status', hue='Crop', data=monthly_suitability_pd, ax=ax)
        ax.set_title('Planting Suitability Over Months')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Drought Status (0=Suitable, 1=Drought)')
        st.pyplot(fig)

    region = st.selectbox("🌍 Region", ["Central Highlands", "Coast", "Western Kenya", "Eastern", "Rift Valley"])

    if st.button("Get Planting Suitability"):
        monthly_suitability = get_planting_suitability(region)
        create_planting_chart(monthly_suitability)

        # Assess which crops are suitable for planting
        suitable_crops = monthly_suitability.groupBy('Crop').agg(
            F.mean('Average_Drought_Status').alias('Avg_Drought_Status')).collect()

        st.write("### Planting Recommendations:")

        # Create a dictionary to hold crops and their suitable months
        crop_recommendations = {}

        for row in suitable_crops:
            crop_name = row['Crop']
            avg_drought_status = row['Avg_Drought_Status']
            if avg_drought_status == 0:
                st.write(f"✅ You may plant **{crop_name}** as the drought status is suitable.")
                # Find the months with drought status of zero for this crop
                zero_months = monthly_suitability.filter((monthly_suitability['Crop'] == crop_name) &
                                                         (monthly_suitability['Average_Drought_Status'] == 0)).select(
                    'Month').distinct().collect()
                crop_recommendations[crop_name] = [row['Month'] for row in zero_months]
            else:
                st.write(f"❌ You may not plant **{crop_name}** as the drought status indicates potential drought.")

        # Provide specific month recommendations for crops that can be planted
        st.write("### Specific Month Recommendations:")
        for crop, months in crop_recommendations.items():
            if months:
                month_names = [calendar.month_name[int(month)] for month in months]  # Convert month numbers to names
                st.write(f"🌱 You can plant **{crop}** in the following months: {', '.join(month_names)}.")



page = st.sidebar.selectbox("Select a page", ["Main Page", "Planting Recommendations"])

if page == "Main Page":
    main_page()
elif page == "Planting Recommendations":
    recommendations_page()