#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
#     
# # FIT5196 Assessment 3
# #### Student Name: QI WANG
# #### Student ID: 33354081
# 
# Date: 14/10/2023
# 
# 
# Environment: Python 3
# 
# Libraries used:<br>
# **pandas**: For data manipulation, analysis, and data structures.<br>
# **datetime**: For handling and manipulating date and time data.<br>
# **fitz**: For working with PDF and other document formats.<br>
# **geopandas**: Provides tools for working with spatial data, extending the capabilities of pandas.<br>
# **shapely**: For manipulation and analysis of planar geometric objects.<br>
# **re**: For regular expression operations, to search, match, and manipulate strings.<br>
# **math**: For basic mathematical and trigonometric calculations.<br>
# **sklearn.preprocessing**: Specifically, StandardScaler for standardizing features by removing the mean and scaling to unit variance.<br>
# **scipy.stats**: Provides a wide variety of statistical functions, with boxcox for performing Box-Cox power transformations.<br>
# **matplotlib**: For plotting and visualization of data.<br>
# **seaborn**: Based on matplotlib, provides a higher-level interface for creating attractive graphics.<br>
# **numpy**: For numerical operations, handling arrays, and mathematical functions.<br>
# 
# </div>

# <div class="alert alert-block alert-danger">
#     
# ## Table of Contents
# 
# </div>    
# 
# [1. Introduction](#Intro) <br>
# [2. Importing Libraries](#libs) <br>
# [3. Task1](#task1) <br>
# [4. Task2](#task2) <br>
# [5. Summary](#summary) <br>
# [6. References](#Ref) <br>

# -------------------------------------

# <div class="alert alert-block alert-warning">
# 
# ## 1.  Introduction  <a class="anchor" name="Intro"></a>
#     
# </div>

# In today's data-driven era, data integration and parsing have become crucial. Various data sources and formats present challenges, but they also offer opportunities, enabling us to extract meaningful insights. In the process of data-driven decision-making, the quality and manner in which data is formatted play a pivotal role in the accuracy and reliability of the results. This project comprises two key tasks, aiming to handle, integrate, and optimize data, laying the groundwork for further analysis and modeling.
# 
# Data integration is the cornerstone of data analysis. In Task 1, we delve deeply into the challenges of integrating data from multiple sources into a single dataset. The form and distribution of data profoundly influence the choice of analytical methods and the ultimate results. Task 2 is centered around data reshaping, with a particular focus on exploring various normalization and transformation methods, such as standardization, min-max normalization, log transformation, power transformation, and Box-Cox transformation.
# 
# 

# -------------------------------------

# <div class="alert alert-block alert-warning">
#     
# ## 2.  Importing Libraries  <a class="anchor" name="libs"></a>
#  </div>

# The packages to be used in this assessment are imported in the following. They are used to fulfill the following tasks:
# 
# 
# * **pandas**: For data manipulation, analysis, and data structures.
# * **datetime**: For handling and manipulating date and time data.
# * **fitz**: For working with PDF and other document formats.
# * **BeautifulSoup (from bs4)**: Used for web scraping purposes to pull the data out of HTML and XML documents. It creates a parse tree from page source code that can be used to extract data in a hierarchical and more readable manner.
# * **geopandas**: Provides tools for working with spatial data, extending the capabilities of pandas.
# * **shapely**: For manipulation and analysis of planar geometric objects.
# * **re**: For regular expression operations, to search, match, and manipulate strings.
# * **math**: For basic mathematical and trigonometric calculations.
# * **sklearn.preprocessing**: Specifically, `StandardScaler` for standardizing features by removing the mean and scaling to unit variance.
# * **scipy.stats**: Provides a wide variety of statistical functions, with `boxcox` for performing Box-Cox power transformations.
# * **matplotlib**: For plotting and visualization of data.
# * **seaborn**: Based on matplotlib, provides a higher-level interface for creating attractive graphics.
# * **numpy**: For numerical operations, handling arrays, and mathematical functions.
# 
# 
# 

# ----

# In[ ]:


get_ipython().system('pip install geopandas')


# In[ ]:


import pandas as pd
import datetime as datetime
from bs4 import BeautifulSoup
import fitz
import geopandas as gpd
from shapely.geometry import Point
import re
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# -------------------------------------

# <div class="alert alert-block alert-warning">
# 
# ## 3.  Task1: Data Integration  <a class="anchor" name="load"></a>
# 
# </div>

# #### Load Data

# In[ ]:


df_jason = pd.read_json("33354081.json")
df_jason.head()


# I fixed two errors in original xml file, and then I can load the xml file successfully.

# In[ ]:


# read and parse XML file 
with open("33354081.xml", "r") as file:
    soup = BeautifulSoup(file, "xml")

# retrive all records
records = []
for item in soup.find_all("property"):  
    record = {}
    for child in item.children:
        if child.name:
            record[child.name] = child.text
    records.append(record)

# create DataFrame
df_xml = pd.DataFrame(records)
df_xml


# Concate two dataframes

# In[ ]:


concat_df = pd.concat([df_xml, df_jason], ignore_index = True)
concat_df


# Determine the suburb each property is located in, and add the suburb name to the `sb_df` dataframe.

# In[ ]:


sb_df = concat_df.copy()

# Step 1: Create a new GeoDataFrame with a geometry column
geometry = [Point(xy) for xy in zip(sb_df["lng"], sb_df["lat"])]
sb_gdf = gpd.GeoDataFrame(sb_df, geometry=geometry)

# Load the Shapefile as a GeoDataFrame
gdf = gpd.read_file("VIC_LOCALITY_POLYGON_shp.shp")

# Step 2: Ensure the CRS match
sb_gdf.crs = gdf.crs

# Step 3: Perform a spatial join to determine the suburb for each point
result_gdf = gpd.sjoin(sb_gdf, gdf, predicate="within", how="left")

# suburb name in the column vic_loca_2
sb_df["suburb"] = result_gdf["VIC_LOCA_2"]  


# In[ ]:


# retrieve lga info from pdf
pdf_document = fitz.open("Lga_to_suburb.pdf")

pdf_text = ""

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pdf_text += page.get_text()

pdf_document.close()
print(pdf_text)


# create dataframe with column lga and suburb infomation

# In[ ]:


lines = pdf_text.strip().split("\n")

data = []

for line in lines:
    # split lga and suburb 
    lga, suburbs_str = line.split(":")
    # use regex to retrive suburb
    suburbs = re.findall(r"'(.*?)'", suburbs_str)
    for suburb in suburbs:
        data.append({"suburb": suburb,"lga": lga})

df = pd.DataFrame(data)
df


# Reference sample out put ,lga and Suburb are upcase format

# In[ ]:


# Convert lga and Suburb to upcase
df['lga'] = df['lga'].str.upper()
df['suburb'] = df['suburb'].str.upper()
df


# -----

# Combining two dataframes based on the suburb they are associated with

# In[ ]:


merged_df = pd.merge(sb_df,df,on = "suburb", how = "left")
merged_df


# -------------------------------------

# Caculate distance from the closest train station to the property

# In[ ]:


# create stops information dataframe
stops = pd.read_csv("stops.txt", sep = ",")
stops


# In[ ]:


# define haversine function
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6378  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

# define closest_station function
def closest_station(lat, lng):
    closest_distance = float('inf')  # Initialize the closest distance to infinity
    closest_station_info = None  # Initialize the nearest site information to None
    for index, row in stops.iterrows():  
        distance = haversine(lng, lat, row['stop_lon'], row['stop_lat'])  
        if distance < closest_distance:  
            closest_distance = distance
            closest_station_info = row
    return pd.Series({'closest_train_station_id': int(closest_station_info['stop_id']),
                      'distance_to_closest_train_station': round(closest_distance,2)})

# Converting lat and lng columns to numeric data types
merged_df['lat'] = pd.to_numeric(merged_df['lat'], errors='coerce')
merged_df['lng'] = pd.to_numeric(merged_df['lng'], errors='coerce')

# Use the apply method to apply the closest_station function
result = merged_df.apply(lambda row: closest_station(row['lat'], row['lng']), axis=1)

df_final = pd.concat([merged_df, result], axis=1)

df_final['closest_train_station_id'] = df_final['closest_train_station_id'].astype(int)

df_final


# I reffer [here](https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points) to caculate distance

# -----

# Detect if the closest station has direct train to melbourne central station

# In[ ]:


# check central station stop id
mc = stops[stops["stop_name"] == "Melbourne Central Railway Station"]
mc


# In[ ]:


calender = pd.read_csv("calendar.txt")
calender


# Through my observation, only T0 has train service from monday to friday.

# In[ ]:


stop_times = pd.read_csv("stop_times.txt")
stop_times


# By observation, there are also T0 tags in the trip_id column. Therefore, I only select the trip id that contain the T0 tag to filter the trips I need.

# In[ ]:


# Filtering the stop_times DataFrame to retain only those rows where the 'trip_id' string contains 'T0'
weekday_stop_times = stop_times[stop_times['trip_id'].str.contains('T0')]

# Identifying the unique trip IDs that stop at the Melbourne Central Station (stop_id 19842) on weekdays.
direct_trip_ids = weekday_stop_times[weekday_stop_times['stop_id'] == 19842]['trip_id'].unique().tolist()

# Filtering the stop_times DataFrame to retain only the rows corresponding to the identified weekday trips
direct_all = stop_times[stop_times['trip_id'].isin(direct_trip_ids)]

def compute_direct_journey_flag(row):
    closest_station_id = row['closest_train_station_id']
    closest_station_records = direct_all[direct_all['stop_id'] == closest_station_id]
    melbourne_central_records = direct_all[direct_all['stop_id'] == 19842]
    
    if melbourne_central_records.empty or closest_station_records.empty:
        return 0  # Return 0 if no records found
    
    melbourne_sequence = melbourne_central_records['stop_sequence'].iloc[0]
    
    # Check if the closest station comes before Melbourne Central Station in the trip sequence
    has_direct_journey = any(
        (record['stop_sequence'] < melbourne_sequence) and 
        ('07:00:00' <= record['departure_time'] <= '09:00:00') 
        for idx, record in closest_station_records.iterrows()
    )
    
    return 1 if has_direct_journey else 0

df_final['direct_journey_flag'] = df_final.apply(compute_direct_journey_flag, axis=1)


# ----

# Calculate average travel time (in minutes) for a direct ride from the nearest train station to "Melbourne Central" station between 7 a.m. and 9 a.m., Monday through Friday.

# In[ ]:


# Functions to fix time formatting
def fix_time_format(t):
    hour, minute, second = map(int, t.split(':'))
    hour = hour % 24  # Convert 24 hours to 0 hours
    return f'{hour:02d}:{minute:02d}:{second:02d}'  # Formatted time is HH:mm:ss
direct_all['arrival_time'] = direct_all['arrival_time'].apply(fix_time_format)
direct_all['departure_time'] = direct_all['departure_time'].apply(fix_time_format)


# I fixed the time format first, because in time column has some error format.

# In[ ]:


# Pre-filter the direct_all dataframe based on the unique closest_train_station_id values in df_final
unique_stations = df_final['closest_train_station_id'].unique()
direct_all_filtered = direct_all[direct_all['stop_id'].isin(unique_stations) | (direct_all['stop_id'] == 19842)]

# Reset the index on direct_all_filtered
direct_all_filtered.reset_index(inplace=True)

def travel_time(row):
    closest_station_id = row['closest_train_station_id']
    if closest_station_id == 19842:  
        return 0

    # Extract the necessary records from the dataframe
    closest_station_records = direct_all_filtered[direct_all_filtered['stop_id'] == closest_station_id]
    melbourne_records = direct_all_filtered[direct_all_filtered['stop_id'] == 19842]
    
    # Merge the two dataframes to easily compare stop_sequence
    merged = pd.merge(closest_station_records, melbourne_records, on='trip_id', suffixes=('_closest', '_melbourne'))
    merged = merged[merged['stop_sequence_closest'] < merged['stop_sequence_melbourne']]
    
    # Filter out records based on time criteria
    valid_times = merged[
        (merged['departure_time_closest'] >= '07:00:00') & 
        (merged['departure_time_closest'] <= '09:00:00')
    ]
    
    # Calculate travel times
    valid_times['departure_time'] = pd.to_datetime(valid_times['departure_time_closest'])
    valid_times['arrival_time'] = pd.to_datetime(valid_times['arrival_time_melbourne'])
    valid_times['travel_time'] = (valid_times['arrival_time'] - valid_times['departure_time']).dt.seconds / 60
    
    # Calculate and return average travel time
    if not valid_times.empty:
        return round(valid_times['travel_time'].mean())
    else:
        return "no direct trip is available"

# Apply the function on the chunk
df_final['travel_min_to_MC'] = df_final.apply(travel_time, axis=1)

# Display a sample of the results
df_final


# ----

# Reorder the column according to the sample output 

# In[ ]:


# Desired column order
desired_order = [
    "property_id", "lat", "lng", "addr_street", "suburb", "lga", 
    "closest_train_station_id", "distance_to_closest_train_station", 
    "travel_min_to_MC", "direct_journey_flag"
]

# Reorder columns
reordered = df_final[desired_order]

# Display the first few rows of the reordered dataframe
reordered.head()


# In[ ]:


reordered.to_csv("33354081_A3_output.csv", index=False)


# -----

# <div class="alert alert-block alert-warning">
# 
# ## 4.  Task2: Datareshaping  <a class="anchor" name="load"></a>
# 
# </div>

# ## Data Loading and Preliminary Exploration

# In[ ]:


# Load the suburb_info.xlsx data
suburb_data = pd.read_excel("suburb_info.xlsx")
suburb_data.head()


# ----

# ## Data cleaning

# I noticed that the data in the columns aus_born_perc, median_income and median_house_price seem to contain some non-numeric characters such as % and $. For further data exploration and processing, I cleaned up these columns

# In[ ]:


# Remove non-numeric characters and convert to appropriate data types
suburb_data['aus_born_perc'] = suburb_data['aus_born_perc'].str.rstrip('%').astype('float') / 100.0
suburb_data['median_income'] = suburb_data['median_income'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(int)
suburb_data['median_house_price'] = suburb_data['median_house_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(int)

# Descriptive statistics of the dataset
desc_stats = suburb_data.describe()

# Check for missing values
missing_values = suburb_data.isnull().sum()

desc_stats, missing_values


# ---

# ## Data exploration

# In order to better understand the distribution of the data and the relationships between the features, I performed the following operations:Plot histograms for number_of_houses, number_of_units, population, aus_born_perc, median_income, and median_house_price

# In[ ]:


# List of columns to visualize
columns_to_visualize = ['number_of_houses', 'number_of_units', 'population', 'aus_born_perc', 'median_income', 'median_house_price']

# Plot histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize, 1):
    plt.subplot(2, 3, i)
    plt.hist(suburb_data[col], bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Through observation, I found that the distributions of number_of_houses, number_of_units and population are all right skewed, indicating that there are some suburbs that have significantly higher numbers of houses, units and population than most other suburbs.<br>
# The distribution of aus_born_perc is relatively even, but there are some concentrations of higher and lower percentages.<br>
# The distribution of median_income is also right-skewed, but most data points are concentrated in the middle range.<br>
# The distribution of median_house_price is right-skewed, suggesting that most suburbs are concentrated in a lower range, but there are some that are significantly higher than others.

# Next, to further explore the relationship between the target variable median_house_price and the other attributes, I will plot scatter plots for each attribute.

# In[ ]:


# Plot scatter plots for each attribute against median_house_price
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data[col], suburb_data['median_house_price'], alpha=0.7)
    plt.title(f'{col} vs. median_house_price')
    plt.xlabel(col)
    plt.ylabel('Median House Price')

plt.tight_layout()
plt.show()


# Accoding to my  observation, I have the following findings.<br>
# `number_of_houses`: There's no evident linear relationship with the median house price.<br>
# `number_of_units`: There's also no clear linear relationship with the median house price.<br>
# `population`: No noticeable linear relationship exists with the median house price.<br>
# `aus_born_perc`: There seems to be a slight linear relationship with the median house price, meaning in suburbs with a higher percentage of Australian-born residents, the house prices might be relatively higher.<br>
# `median_income`: There's a clear positive linear relationship with the median house price, implying that in suburbs with a higher median income, house prices are typically higher as well.<br>
# Overall, a distinct linear relationship exists between `median_income` and `median_house_price`, while the relationships between other attributes and the median house price are not very pronounced

# ------

# ## Data normalization/transformation

# Based on my observation, scaling or transformation is essential due to the presence of right-skewed features such as `number_of_houses` and `median_income`. Additionally, the wide variation in feature scales can bias certain algorithms. Transformations can enhance the linearity of relationships, improving model performance, while scaling ensures a consistent and interpretable impact of each feature on the model's outcome.
# 

# In[ ]:


# Apply standardisation
standard_scaler = StandardScaler()
suburb_data_standardised = suburb_data.copy()
suburb_data_standardised[columns_to_visualize] = standard_scaler.fit_transform(suburb_data[columns_to_visualize])

# Plot scatter plots for each attribute against median_house_price after standardisation
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data_standardised[col], suburb_data_standardised['median_house_price'], alpha=0.7)
    plt.title(f'Standardised {col} vs. median_house_price')
    plt.xlabel(f'Standardised {col}')
    plt.ylabel('Standardised Median House Price')

plt.tight_layout()
plt.show()


# The primary purpose of standardization is to scale all features to the same scale, ensuring that their weights in the model can be fairly compared. Although the scale of the data has changed (all features now have a mean of approximately 0 and a standard deviation of 1), the relationship between the features and the target variable remains unchanged. Compared to the original data, the linear relationship has neither strengthened nor weakened.

# In[ ]:


# Apply min-max normalization
min_max_scaler = MinMaxScaler()
suburb_data_minmax = suburb_data.copy()
suburb_data_minmax[columns_to_visualize] = min_max_scaler.fit_transform(suburb_data[columns_to_visualize])

# Plot scatter plots for each attribute against median_house_price after min-max normalization
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data_minmax[col], suburb_data_minmax['median_house_price'], alpha=0.7)
    plt.title(f'{col} vs. median_house_price')
    plt.xlabel(f'Min-Max Normalised {col}')
    plt.ylabel('Min-Max Normalised Median House Price')

plt.tight_layout()
plt.show()


# Through observation, it can be noted that the min-max normalization has scaled the data, bringing it into the [0,1] range. The scale of the data has been altered, but the relationship between the features and the target variable remains consistent.

# In[ ]:


# Apply log transformation
suburb_data_log = suburb_data.copy()
suburb_data_log[columns_to_visualize] = np.log(suburb_data[columns_to_visualize])

# Plot scatter plots for each attribute against median_house_price after log transformation
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data_log[col], suburb_data_log['median_house_price'], alpha=0.7)
    plt.title(f'Log Transformed {col} vs. median_house_price')
    plt.xlabel(f'Log Transformed {col}')
    plt.ylabel('Log Transformed Median House Price')

plt.tight_layout()
plt.show()


# From the graphs, it is evident that the logarithmic transformation has altered the distribution and scale of the data, especially for features that had a right-skewed original distribution, such as number_of_houses, number_of_units, and population. For these features, the data appears more concentrated after the transformation, and the relationship with the median house price seems more linear.

# In[ ]:


# Apply power transformation (square)
suburb_data_power = suburb_data.copy()
suburb_data_power[columns_to_visualize] = np.power(suburb_data[columns_to_visualize], 2)

# Plot scatter plots for each attribute against median_house_price after power transformation
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data_power[col], suburb_data_power['median_house_price'], alpha=0.7)
    plt.title(f'Power Transformed {col} vs. median_house_price')
    plt.xlabel(f'Power Transformed {col}')
    plt.ylabel('Power Transformed Median House Price')

plt.tight_layout()
plt.show()


# From the graphs, it can be observed that the power transformation (square transformation in this context) has had a significant impact on the distribution and scale of the data. For certain features, like number_of_units, the distribution of the data seems more concentrated after the transformation. However, the relationship with the target variable is still not very apparent in some cases.

# In[ ]:


# Apply Box-Cox transformation
suburb_data_boxcox = suburb_data.copy()
for col in columns_to_visualize:
    suburb_data_boxcox[col], _ = boxcox(suburb_data[col])

# Plot scatter plots for each attribute against median_house_price after Box-Cox transformation
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize[:-1], 1):  # Exclude median_house_price for scatter plot
    plt.subplot(2, 3, i)
    plt.scatter(suburb_data_boxcox[col], suburb_data_boxcox['median_house_price'], alpha=0.7)
    plt.title(f'Box-Cox Transformed {col} vs. median_house_price')
    plt.xlabel(f'Box-Cox Transformed {col}')
    plt.ylabel('Box-Cox Transformed Median House Price')

plt.tight_layout()
plt.show()


# The Box-Cox transformation brings the data closer to a normal distribution, potentially improving the performance of the model. From the graphs, it's evident that the data distribution is more concentrated after the transformation, and in some instances, the relationship with the median house price has become more pronounced.

# -----

# ## Summary

# **Standardization and Min-Max Normalization:** These methods primarily scale the data without altering its shape or its relationship with the target variable.<br>
# **Log Transformation:** For right-skewed data, this transformation helps in reducing the influence of larger values, making the data more concentrated.<br>
# **Power Transformation:** In my case, squaring had a noticeable impact on the data distribution, but the relationship with the target variable didn't improve significantly.<br>
# **Box-Cox Transformation:** This transformation brought the data closer to a normal distribution, potentially aiding in the performance of the model.<br>
# From the observations, the log transformation and Box-Cox transformation seem promising as they enhanced the data distribution and strengthened the relationship with the target variable.

# -------------------------------------

# <div class="alert alert-block alert-warning">
# 
# ## 5. Summary <a class="anchor" name="summary"></a>
# 
# </div>

# In today's data-centric era, integrating and parsing data from diverse sources have become imperative. This not only poses challenges but also opens avenues for extracting invaluable insights. This project is segmented into two pivotal tasks, focusing on refining, amalgamating, and optimizing the data to set a robust foundation for subsequent analyses and modeling.
# 
# Task 1 delves deep into the intricacies of integrating data from multiple sources into a cohesive dataset. Initially, with the aid of various libraries such as geopandas and shapely, data from different formats like JSON and XML were seamlessly imported and merged into a unified dataframe. This was followed by a series of data cleaning and transformations to ensure consistency and accuracy of the data.
# 
# Task 2 hones in on the aspect of data reshaping. A spectrum of normalization and transformation techniques, including standardization, min-max normalization, log transformation, power transformation, and Box-Cox transformation were explored. The application of these methods ensures that the data's form and distribution are conducive to the choice of analytical methods and yield optimal results.
# 
# Through the aforementioned steps, the data's integrity was not only ensured but also primed for further analysis and modeling.
# 
# 
# 
# 
# 
# 
# 
# 

# 
# 
# 
# 
# 
# 
# 
# 
# 

# -------------------------------------

# <div class="alert alert-block alert-warning">
# 
# ## 6. References <a class="anchor" name="Ref"></a>
# 
# </div>

# ## --------------------------------------------------------------------------------------------------------------------------

# In[ ]:




