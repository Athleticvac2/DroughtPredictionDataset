import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Practical soil types and their characteristics
soil_types = {
	'Sandy Loam': {
		'sand_percent': 60,
		'silt_percent': 30,
		'clay_percent': 10,
		'water_retention': 0.6,
		'drainage_rate': 0.8,
		'nutrient_retention': 0.65,
		'ph_range': '5.5-6.8',
		'organic_matter': '2-3'
	},
	'Clay Loam': {
		'sand_percent': 34,
		'silt_percent': 34,
		'clay_percent': 32,
		'water_retention': 0.8,
		'drainage_rate': 0.5,
		'nutrient_retention': 0.85,
		'ph_range': '6.0-7.0',
		'organic_matter': '3-4'
	},
	'Heavy Clay': {
		'sand_percent': 20,
		'silt_percent': 20,
		'clay_percent': 60,
		'water_retention': 0.9,
		'drainage_rate': 0.3,
		'nutrient_retention': 0.9,
		'ph_range': '6.5-7.5',
		'organic_matter': '3-5'
	},
	'Silt Loam': {
		'sand_percent': 15,
		'silt_percent': 70,
		'clay_percent': 15,
		'water_retention': 0.75,
		'drainage_rate': 0.6,
		'nutrient_retention': 0.75,
		'ph_range': '6.0-7.0',
		'organic_matter': '2.5-3.5'
	},
	'Loam': {
		'sand_percent': 40,
		'silt_percent': 40,
		'clay_percent': 20,
		'water_retention': 0.7,
		'drainage_rate': 0.7,
		'nutrient_retention': 0.8,
		'ph_range': '6.0-7.0',
		'organic_matter': '3-4'
	}
}

# Kenya's agricultural regions with common soil types
regions = {
	'Central Highlands': {
		'counties': ['Nyeri', 'Kirinyaga', 'Murang\'a', 'Kiambu'],
		'common_soils': ['Clay Loam', 'Loam'],
		'crops': ['Tea', 'Coffee', 'Maize', 'Beans', 'Potatoes'],
		'altitude': '1500-2500m',
		'avg_rainfall': '1200-2000mm',
		'temp_range': '10-24'
	},
	'Western Kenya': {
		'counties': ['Kakamega', 'Bungoma', 'Busia', 'Vihiga'],
		'common_soils': ['Clay Loam', 'Heavy Clay'],
		'crops': ['Maize', 'Sugarcane', 'Beans', 'Sweet Potatoes'],
		'altitude': '1000-1800m',
		'avg_rainfall': '1100-2000mm',
		'temp_range': '14-32'
	},
	'Rift Valley': {
		'counties': ['Uasin Gishu', 'Trans Nzoia', 'Nakuru', 'Nandi'],
		'common_soils': ['Loam', 'Sandy Loam'],
		'crops': ['Wheat', 'Maize', 'Potatoes'],
		'altitude': '1500-2300m',
		'avg_rainfall': '900-1800mm',
		'temp_range': '12-28'
	},
	'Coast': {
		'counties': ['Kilifi', 'Kwale', 'Lamu', 'Mombasa'],
		'common_soils': ['Sandy Loam', 'Silt Loam'],
		'crops': ['Coconuts', 'Cashewnuts', 'Cassava', 'Mangoes'],
		'altitude': '0-500m',
		'avg_rainfall': '500-1200mm',
		'temp_range': '22-35'
	},
	'Eastern': {
		'counties': ['Machakos', 'Makueni', 'Kitui', 'Embu'],
		'common_soils': ['Sandy Loam', 'Clay Loam'],
		'crops': ['Sorghum', 'Millet', 'Green Grams', 'Cowpeas'],
		'altitude': '500-1200m',
		'avg_rainfall': '500-1000mm',
		'temp_range': '18-33'
	}
}

# Generate data
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

data = []
for region_name, region_data in regions.items():
	for county in region_data['counties']:
		for soil_type in region_data['common_soils']:
			soil_props = soil_types[soil_type]
			for crop in region_data['crops']:
				for date in dates:
					month = date.month
					is_rainy_season = month in [3, 4, 5, 10, 11, 12]

					# Weather data
					min_rain, max_rain = map(int, region_data['avg_rainfall'].replace('mm', '').split('-'))
					min_temp, max_temp = map(int, region_data['temp_range'].split('-'))

					if is_rainy_season:
						rainfall = np.random.uniform(min_rain / 12 * 1.5, max_rain / 12 * 1.5)
						humidity = np.random.uniform(70, 85)
					else:
						rainfall = np.random.uniform(0, min_rain / 12)
						humidity = np.random.uniform(50, 65)

					temperature = np.random.uniform(min_temp, max_temp)

					# Soil moisture calculation based on soil properties
					soil_moisture = min(100, (rainfall * soil_props['water_retention'] +
					                          humidity * 0.3) * (1 - soil_props['drainage_rate'] * 0.2))

					# Calculate pH within range
					min_ph, max_ph = map(float, soil_props['ph_range'].split('-'))
					ph_value = round(np.random.uniform(min_ph, max_ph), 1)

					# Calculate organic matter within range
					min_om, max_om = map(float, soil_props['organic_matter'].split('-'))
					organic_matter = round(np.random.uniform(min_om, max_om), 1)

					# Crop health index (0-100)
					crop_health = min(100, max(0,
					                           soil_moisture * 0.3 +
					                           soil_props['nutrient_retention'] * 30 +
					                           (1 - abs(ph_value - 6.5) / 2) * 20 +  # Optimal pH around 6.5
					                           organic_matter * 5
					                           ))

					data.append({
						'Date': date.strftime('%Y-%m-%d'),
						'Region': region_name,
						'County': county,
						'Soil_Type': soil_type,
						'Sand_Percent': soil_props['sand_percent'],
						'Silt_Percent': soil_props['silt_percent'],
						'Clay_Percent': soil_props['clay_percent'],
						'Crop': crop,
						'Rainfall_mm': round(rainfall, 1),
						'Temperature_Celsius': round(temperature, 1),
						'Humidity_Percent': round(humidity, 1),
						'Soil_Moisture_Percent': round(soil_moisture, 1),
						'pH_Value': ph_value,
						'Organic_Matter_Percent': organic_matter,
						'Water_Retention': soil_props['water_retention'],
						'Drainage_Rate': soil_props['drainage_rate'],
						'Nutrient_Retention': soil_props['nutrient_retention'],
						'Crop_Health_Index': round(crop_health, 1),
						'Altitude_Range': region_data['altitude']
					})

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('/home/athleticvac2/PycharmProjects/DroughtPredictionDataset/data/kenya_agricultural_soil_data.csv', index=False)

print(f"Generated {len(df)} records")
print("\nSample of the data:")
print(df.head())
print("\nDataset statistics:")
print(df.describe())