import pandas as pd
import boto3
from io import StringIO
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import folium_static
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
access_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_id = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION')

# Function to apply KMeans clustering
def apply_kmeans(df, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    return df, kmeans.cluster_centers_

# Function to visualize clusters on a Folium map
def visualize_clusters(df, centers):
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    map = folium.Map(location=map_center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(map)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.Icon(icon='record', color='red'),
        ).add_to(marker_cluster)

    for center in centers:
        folium.Marker(
            location=center,
            icon=folium.Icon(icon='star', color='blue'),
            popup='Patrol Center'
        ).add_to(map)

    return map

# Main function to display the patrolling map
def patrolling_main():
    data = load_data()
    if data.empty:
        st.error("Failed to load data")
        return

    st.title('Patrolling Map')
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    df = data.drop_duplicates(subset=['Latitude', 'Longitude', 'CrimeHead_Name'])
    
    clusters = st.slider('Select number of clusters', min_value=3, max_value=20, value=10, step=1)
    df, centers = apply_kmeans(df, clusters)
    
    if len(df) > 10000:  # Limiting the number of markers to avoid overloading the browser
        df_sample = df.sample(n=10000)
    else:
        df_sample = df
    
    crime_map = visualize_clusters(df_sample, centers)
    folium_static(crime_map)

@st.cache_data(show_spinner=True)
def load_data():
    bucket_name = 'new-trail01'
    file_key = 'FIR_Details_Data.csv'
    
    # Read files from S3
    s3_client = boto3.client('s3', region_name=region_name, aws_access_key_id=access_id, aws_secret_access_key=secret_id)
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    if status == 200:
        csv_content = response["Body"].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

if __name__ == '__main__':
    patrolling_main()
