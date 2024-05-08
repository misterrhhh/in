import pandas as pd
from sklearn.cluster import KMeans
import folium

# Load your dataset
df = pd.read_csv('output.csv', encoding='latin1')

# Assuming 'INCENDIO' is a column in your DataFrame and you want to filter where its value is 1
filtered_df = df[df['INCENDIO'] == 1]

# Select the latitude and longitude columns from the filtered DataFrame
coordinates = filtered_df[['LAT', 'LON']].values

k = 2

# Apply K-means clustering
kmeans = KMeans(k, random_state=42).fit(coordinates)

# Get the cluster labels
labels = kmeans.labels_

# Create a map centered around the average latitude and longitude of the filtered data
m = folium.Map(location=[coordinates[:, 0].mean(), coordinates[:, 1].mean()], zoom_start=10)

# Define colors for each cluster
colors = ['red', 'blue']

# Plot each cluster on the map
for i in range(k):
    cluster_coordinates = coordinates[labels == i]
    folium.CircleMarker(
        location=[cluster_coordinates[:, 0].mean(), cluster_coordinates[:, 1].mean()],
        radius=15,
        color=colors[i],
        fill=True,
        fill_color=colors[i],
        popup=f'Cluster {i+1}',
    ).add_to(m)

# Save the map as an HTML file
m.save('map.html')

print("done")
