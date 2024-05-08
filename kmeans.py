import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('output.csv', encoding='latin1')


print(df)
# Define preprocessing steps
numeric_features = ['AREAPOV', 'AREAMATO', 'AREAAGRIC', 'AREATOTAL', 'REACENDIMENTOS', 'FALSOALARME', 'FOGACHO', 'INCENDIO', 'AGRICOLA', 'DURACAO', 'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'PRECEPITACAO', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'DSR', 'THC', 'ALTITUDEMEDIA', 'DECLIVEMEDIO', 'DENDIDADERV', 'COSN5VARIEDADE']
categorical_features = ['DISTRITO', 'TIPO', 'FALSOALARME', 'FOGACHO', 'FREGUESIA', 'FONTEALERTA', 'TIPOCAUSA', 'INCENDIO', 'AGRICOLA', 'FREGUESIA', 'FONTEALERTA', 'TIPOCAUSA']

# Create a pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
                           ('kmeans', KMeans(n_clusters=3, random_state=42))])

# Fit the pipeline to the data
pipeline.fit(df)

# Predict the clusters
clusters = pipeline.predict(df)

# Add the cluster labels to the DataFrame
df['Cluster'] = clusters

# Step 4: Visualize the results
plt.scatter(df['TEMPERATURA'], df['HUMIDADERELATIVA'], c=df['Cluster'], cmap='viridis')
plt.title('K-means Clustering of Weather Parameters Impacting Wildfires')
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.show()

