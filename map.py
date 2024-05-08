import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Step 1: Read the CSV file
df = pd.read_csv('output.csv', encoding='latin1')

min_lon = -8.15716448432848
max_lon = -7.392083959998975
filtered_df = df[(df['LON'] >= min_lon) & (df['LON'] <= max_lon)]

# Check if filtered_df is empty
if filtered_df.empty:
    print("No data found within the specified longitude range.")
else:
    # Step 2: Calculate the mean latitude and longitude
    mean_lat = filtered_df['LAT'].mean()
    mean_lon = filtered_df['LON'].mean()

    # Step 3: Calculate the range of latitude and longitude to cover around the mean
    range_lat = max(filtered_df['LAT']) - min(filtered_df['LAT'])
    range_lon = max(filtered_df['LON']) - min(filtered_df['LON'])

    # Step 4: Adjust the bounds of the map based on the mean and range
    llcrnrlat = max(mean_lat - range_lat / 2, -90)  # Ensure the lower bound is not less than -90
    urcrnrlat = min(mean_lat + range_lat / 2, 90)    # Ensure the upper bound is not greater than 90
    llcrnrlon = max(mean_lon - range_lon / 2, -180)  # Ensure the lower bound is not less than -180
    urcrnrlon = min(mean_lon + range_lon / 2, 180)   # Ensure the upper bound is not greater than 180

    # Step 5: Create a scatter plot with a map in the background
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # Create a Basemap instance with adjusted bounds
    m = Basemap(ax=ax, projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='l')

    # Draw coastlines, countries, and rivers
    m.drawcoastlines()
    m.drawcountries()
    m.drawrivers(linewidth=0.5)

    # Scatter plot of latitude and longitude
    m.scatter(filtered_df['LON'], filtered_df['LAT'], latlon=True, s=20, alpha=0.7, color='blue')

    # Add a title
    plt.title('Zoomed Map Based on Mean Latitude and Longitude')

    # Show the plot
    plt.show()
