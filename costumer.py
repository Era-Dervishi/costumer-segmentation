import os
from pandas import DataFrame
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data from CSV file
data = pd.read_csv('C:/Users/User/Desktop/Mall_Customers.csv')

# Display the first few rows and columns of the dataframe to verify column names
print("First few rows of the dataframe:")
print(data.head())

print("Column names in the dataframe:")
print(data.columns)

# Check for missing values
print(data.isnull().sum())

# Drop the 'CustomerID' column since it's irrelevant for clustering
data = data.drop(columns=['CustomerID'])
print(data.head())

data: DataFrame = data.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'})

# Visualization of count difference in gender distribution:
labels = data['Gender'].unique()
values = data['Gender'].value_counts(ascending=True)
plt.bar(x=labels, height=values, width=0.4, align='center', color=['#42a7f5', '#d400ad'])
plt.title('Count difference in Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('No. of Customers')
plt.ylim(0,130)
plt.axhline(y=data['Gender'].value_counts()[0], color='#d400ad', linestyle='--', label=f'Female ({data.Gender.value_counts()[0]})')
plt.axhline(y=data['Gender'].value_counts()[1], color='#42a7f5', linestyle='--', label=f'Male ({data.Gender.value_counts()[1]})')
plt.legend()
plt.show()

# Age distribution
fig, ax = plt.subplots(figsize=(20,8))
sns.set(font_scale=1.5)
ax = sns.countplot(x=data['Age'], palette='spring')
ax.axhline(y=data['Age'].value_counts().max(), linestyle='--', color='#c90404', label=f'Max Age Count ({data.Age.value_counts().max()})')
ax.axhline(y=data['Age'].value_counts().mean(), linestyle='--', color='#eb50db', label=f'Average Age Count ({data.Age.value_counts().mean():.1f})')
ax.axhline(y=data['Age'].value_counts().min(), linestyle='--', color='#046ebf', label=f'Min Age Count ({data.Age.value_counts().min()})')
ax.legend(loc='right')
ax.set_ylabel('No. of Customers')
plt.title('Age Distribution', fontsize=20)
plt.show()

# Annual income distribution
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.histplot(data['Annual_Income'], bins=15, ax=ax, color='orange')
ax.set_xlabel('Annual Income (in Thousand USD)')
plt.title('Annual Income count Distribution of Customers', fontsize=20)
plt.show()

# Visualizing Annual Income per Age on a Scatterplot.
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Annual_Income'], x=data['Age'], color='#f73434', s=70, edgecolor='black', linewidth=0.3)
ax.set_ylabel('Annual Income (in Thousand USD)')
plt.title('Annual Income per Age', fontsize=20)
plt.show()

# Visualizing statistical difference of Annual Income between Male and Female Customers.
fig, ax = plt.subplots(figsize=(10,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(x=data['Gender'], y=data["Annual_Income"], hue=data['Gender'], palette='seismic')
ax.set_ylabel('Annual Income (in Thousand USD)')
plt.title('Annual Income Distribution by Gender', fontsize=20)
plt.show()

# Visualizing annual Income per Age by Gender on a scatterplot.
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Annual_Income'], x=data['Age'], hue=data['Gender'], palette='seismic', s=70, edgecolor='black', linewidth=0.3)
ax.set_ylabel('Annual Income (in Thousand USD)')
ax.legend(loc='upper right')
plt.title('Annual Income per Age by Gender', fontsize=20)
plt.show()

# Spending score distribution.
fig, ax = plt.subplots(figsize=(5,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(y=data['Spending_Score'], color="#f73434")
ax.axhline(y=data['Spending_Score'].max(), linestyle='--', color='#c90404', label=f'Max Spending ({data.Spending_Score.max()})')
ax.axhline(y=data['Spending_Score'].describe()[6], linestyle='--', color='#f74343', label=f'75% Spending ({data.Spending_Score.describe()[6]:.2f})')
ax.axhline(y=data['Spending_Score'].median(), linestyle='--', color='#eb50db', label=f'Median Spending ({data.Spending_Score.median():.2f})')
ax.axhline(y=data['Spending_Score'].describe()[4], linestyle='--', color='#eb50db', label=f'25% Spending ({data.Spending_Score.describe()[4]:.2f})')
ax.axhline(y=data['Spending_Score'].min(), linestyle='--', color='#046ebf', label=f'Min Spending ({data.Spending_Score.min()})')
ax.legend(fontsize='xx-small', loc='upper right')
ax.set_ylabel('Spending Score')
plt.title('Spending Score', fontsize=20)
plt.show()

# Visualizing Spending Scores per Age on a scatterplot.
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Spending_Score'], x=data['Age'], s=70, color='#f73434', edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')
plt.title('Spending Scores per Age', fontsize=20)
plt.show()

# Visualizing statistical difference of Spending Score between Male and Female Customers.
fig, ax = plt.subplots(figsize=(10,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(x=data['Gender'], y=data["Spending_Score"], hue=data['Gender'], palette='seismic')
ax.set_ylabel('Spending Score')
plt.title('Spending Score Distribution by Gender', fontsize=20)
plt.show()

# Visualizing Spending Score per Age by Gender on a scatterplot.
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Spending_Score'], x=data['Age'], hue=data['Gender'], palette='seismic', s=70, edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')
ax.legend(loc='upper right')
plt.title('Spending Score per Age by Gender', fontsize=20)
plt.show()

clustering_data = data.iloc[:, [2, 3]]
print(clustering_data.head())

# Visualizing the data which we are going to use for the clustering.
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=clustering_data['Spending_Score'], x=clustering_data['Annual_Income'], s=70, color='#f73434', edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')
ax.set_xlabel('Annual Income (in Thousand USD)')
plt.title('Spending Score per Annual Income', fontsize=20)
plt.show()

# Visualize the Elbow Method
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(clustering_data)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(15,7))
plt.plot(range(1, 30), wcss, linewidth=2, color="red", marker="8")
plt.axvline(x=5, ls='--')
plt.ylabel('WCSS')
plt.xlabel('No. of Clusters (k)')
plt.title('The Elbow Method', fontsize=20)
plt.show()

# Apply KMeans with the optimal number of clusters
kms = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42)
kms.fit(clustering_data)

clusters = clustering_data.copy()
clusters['Cluster_Prediction'] = kms.predict(clustering_data)
print(clusters.head())

# Visualize the clusters
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
sns.scatterplot(x=clusters['Annual_Income'], y=clusters['Spending_Score'], hue=clusters['Cluster_Prediction'], palette='viridis', s=100, edgecolor='black', linewidth=0.3)
plt.scatter(kms.cluster_centers_[:, 0], kms.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.legend()
ax.set_ylabel('Spending Scores')
ax.set_xlabel('Annual Income (in Thousand USD)')
plt.title('Clusters of Customers', fontsize=20)
plt.show()

#Seprate clusters
colors = sns.color_palette('viridis', 5)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,20))
ax[0,0].scatter(x=clusters[clusters['Cluster_Prediction'] == 4]['Annual_Income'],
            y=clusters[clusters['Cluster_Prediction'] == 4]['Spending_Score'],
            s=40,edgecolor='black', linewidth=0.3, c=[colors[4]], label='Cluster 1')
ax[0,0].scatter(x=kms.cluster_centers_[4,0], y=kms.cluster_centers_[4,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[0,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 1')

ax[0,1].scatter(x=clusters[clusters['Cluster_Prediction'] == 0]['Annual_Income'],
            y=clusters[clusters['Cluster_Prediction'] == 0]['Spending_Score'],
            s=40,edgecolor='black', linewidth=0.3, c=[colors[0]], label='Cluster 2')
ax[0,1].scatter(x=kms.cluster_centers_[0,0], y=kms.cluster_centers_[0,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[0,1].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 2')

ax[1,0].scatter(x=clusters[clusters['Cluster_Prediction'] == 2]['Annual_Income'],
            y=clusters[clusters['Cluster_Prediction'] == 2]['Spending_Score'],
            s=40,edgecolor='black', linewidth=0.2, c=[colors[2]], label='Cluster 3')
ax[1,0].scatter(x=kms.cluster_centers_[2,0], y=kms.cluster_centers_[2,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[1,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 3')

ax[1,1].scatter(x=clusters[clusters['Cluster_Prediction'] == 1]['Annual_Income'],
            y=clusters[clusters['Cluster_Prediction'] == 1]['Spending_Score'],
            s=40,edgecolor='black', linewidth=0.3, c=[colors[1]], label='Cluster 4')
ax[1,1].scatter(x=kms.cluster_centers_[1,0], y=kms.cluster_centers_[1,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[1,1].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 4')

ax[2,0].scatter(x=clusters[clusters['Cluster_Prediction'] == 3]['Annual_Income'],
            y=clusters[clusters['Cluster_Prediction'] == 3]['Spending_Score'],
            s=40,edgecolor='black', linewidth=0.3, c=[colors[3]], label='Cluster 5')
ax[2,0].scatter(x=kms.cluster_centers_[3,0], y=kms.cluster_centers_[3,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3, label='Centroids')
ax[2,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 5')

fig.delaxes(ax[2,1])
fig.legend(loc='right')
fig.suptitle('Individual Clusters')
plt.show()
