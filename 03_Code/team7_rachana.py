#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib import gridspec
import seaborn as sns 
import altair as alt
from scipy.stats import randint
import os
import plotly.graph_objects as go

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler, StandardScaler, label_binarize
from sklearn.preprocessing import OneHotEncoder

#Featureimportance
# need to install if you don't have 
# pip3 install yellowbrick
from yellowbrick.model_selection import FeatureImportances

#Correlation
from scipy.stats import spearmanr


# Validation libraries
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

#models
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#%%
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(current_dir, "Mental Health Dataset.csv")

# Read the CSV file
health_data = pd.read_csv(csv_file_path)

print("\nReady to continue.")
# %%
print(health_data.info())
health_data.head()
# %%
## Removing null values
print(health_data.isnull().sum())
print(health_data.dropna(inplace=True))
health_data.info()

#%%[markdown]
# Cleaning Grwoing Stress column

# Assuming health_data is your DataFrame
# Step 1: Remove 'Maybe' values from Growing_Stress
health_data = health_data[health_data['Growing_Stress'] != 'Maybe']

print(health_data['Growing_Stress'].unique())

def map_growing_stress_column(df, column_name):
    """
    Maps the 'Growing_Stress' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Growing_Stress' column
health_data = map_growing_stress_column(health_data, 'Growing_Stress')


# %%
# Clean Timestamp - Convert time and date to year

def process_timestamp_to_year(df, column_name):
    """
    Converts a Timestamp column in the DataFrame to numeric years.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the timestamp column.
        column_name (str): The name of the timestamp column.

    Returns:
        pd.DataFrame: The DataFrame with the processed column as numeric years.
    """
    # Convert the column to datetime
    df[column_name] = pd.to_datetime(df[column_name], format='%m/%d/%Y %H:%M', errors='coerce')
    
    # Extract the year
    df[column_name] = df[column_name].dt.year
    
    # Convert the years to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    return df

# Example usage
health_data = process_timestamp_to_year(health_data, 'Timestamp')

# Preview the result
print(health_data[['Timestamp']].head())
print(health_data['Timestamp'].info())

#%%
print(health_data['Timestamp'].unique())

# %%
# Cleaning Gender variable - Now Male -> 0 and Female -> 1

import pandas as pd

def process_gender_column(df, column_name):
    """
    Processes a Gender column in a DataFrame:
    - Maps "Female" to 1 and "Male" to 0.
    - Handles unexpected values by setting them to NaN or a default value.
    - Prints unique values and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the Gender column.
        column_name (str): The name of the Gender column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the Gender column processed.
    """
    # Map Gender to numeric values, setting unexpected values to NaN
    df[column_name] = df[column_name].map({'Female': 1, 'Male': 0})
    
    # Print unique values and basic column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the Gender column
health_data = process_gender_column(health_data, 'Gender')

# %%
# Cleaning the self_employed data
print(health_data['self_employed'].unique())
def process_self_employed_column(df, column_name):
    """
    Processes a self_employed column in a DataFrame:
    - Maps "Yes" to 1 and "No" to 0.
    - Handles unexpected values by setting them to NaN.
    - Prints unique values, first few rows, and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the self_employed column.
        column_name (str): The name of the self_employed column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the self_employed column processed.
    """
    # Map 'Yes' to 1 and 'No' to 0, handle unexpected values by setting to NaN
    df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})
    
    # Print unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the self_employed column
health_data = process_self_employed_column(health_data, 'self_employed')


# %%
# Cleaning the family_history column
health_data['family_history'].unique()

def process_family_history_column(df, column_name):
    """
    Processes a family_history column in a DataFrame:
    - Maps "Yes" to 1 and "No" to 0.
    - Handles unexpected values by setting them to NaN.
    - Prints unique values, first few rows, and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the family_history column.
        column_name (str): The name of the family_history column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the family_history column processed.
    """
    # Map 'Yes' to 1 and 'No' to 0, handle unexpected values by setting to NaN
    df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})
    
    # Print unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the family_history column
health_data = process_family_history_column(health_data, 'family_history')

# %%
# Clean the Days_Indoors
health_data['Days_Indoors'].unique()


def process_days_indoors_column(df, column_name):
    """
    Processes the 'Days_Indoors' column in a DataFrame by mapping each category 
    to a corresponding numerical value representing the approximate number of days.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'Go out Every day': 365,
        'More than 2 months': 60,
        '31-60 days': 45,
        '15-30 days': 22.5,
        '1-14 days': 7.5
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)

    # Handle missing or unmapped values by setting a default (e.g., NaN or 0)
    df[column_name].fillna(-1, inplace=True)  # Replace NaN with -1 for clarity
    
    # Print the unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Example usage of the function
health_data = process_days_indoors_column(health_data, 'Days_Indoors')


# %%
# Clean Treatment column
health_data['treatment'].unique()
def process_treatment_column(df, column_name):
    """
    Maps the 'treatment' column in a DataFrame to 0 and 1.
    'Yes' -> 1, 'No' -> 0.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'Yes': 1,
        'No': 0
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)

    # Handle missing or unmapped values if necessary (e.g., replace NaN with a default value)
    df[column_name].fillna(-1, inplace=True)  # Replace NaN with -1 for clarity
    
    # Print unique values, first few rows, and column description
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df
# Apply the function to the treatment column
health_data = process_treatment_column(health_data, 'treatment')


# %%
# Clean changes_habits column
health_data['Changes_Habits'].unique()
def process_changes_habits_column(df, column_name):
    """
    Maps the 'Changes_Habits' column in a DataFrame to numerical values.
    'No' -> 0, 'Yes' -> 1, 'Maybe' -> 2.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1,
        'Maybe': 2
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)


    # Print unique values, first few rows, and column description
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df
# Apply the function to the Changes_Habits column
health_data = process_changes_habits_column(health_data, 'Changes_Habits')

# %%
# Clean the Mood_Swings
health_data['Mood_Swings'].unique()

def map_mood_swings_column(df, column_name):
    """
    Maps the 'Mood_Swings' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Mood_Swings' column
health_data = map_mood_swings_column(health_data, 'Mood_Swings')

# %%
# Clean the Mental_Health_History
health_data['Mental_Health_History'].unique()

def map_mental_health_history_column(df, column_name):
    """
    Maps the 'Mental_Health_History' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'Yes': 1,
        'No': 0,
        'Maybe': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Mental_Health_History' column
health_data = map_mental_health_history_column(health_data, 'Mental_Health_History')


# %%
# Clean Coping_Struggles column

health_data['Coping_Struggles'].unique()

def map_coping_struggles_column(df, column_name):
    """
    Maps the 'Coping_Struggles' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Coping_Struggles' column
health_data = map_coping_struggles_column(health_data, 'Coping_Struggles')

# %%
# Clean Work_Interest column

health_data['Work_Interest'].unique()

def map_work_interest_column(df, column_name):
    """
    Maps the 'Work_Interest' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1,
        'Maybe': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Work_Interest' column
health_data = map_work_interest_column(health_data, 'Work_Interest')
# %%
# Clean Social_Weakness
health_data['Social_Weakness'].unique()

def map_social_weakness_column(df, column_name):
    """
    Maps the 'Social_Weakness' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Social_Weakness' column
health_data = map_social_weakness_column(health_data, 'Social_Weakness')

# %%
print(health_data['mental_health_interview'].unique())

def map_mental_health_interview(df, column_name):
    """
    Maps the 'Social_Weakness' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Social_Weakness' column
health_data = map_social_weakness_column(health_data, 'mental_health_interview')

# %%
# Clean care_options column
health_data['care_options'].unique()

def map_care_options_column(df, column_name):
    """
    Maps the 'care_options' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Not sure': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'care_options' column
health_data = map_care_options_column(health_data, 'care_options')


# %%
#Verifying the cleaned dataset
health_data.describe()
print(health_data.info())
# %%[markdown]
# Visualisation for Proportion of each feature in dataset

sns.set_style("whitegrid")

# Columns to visualize
cols_to_visualize = ['Gender', 'Occupation', 'self_employed', 'family_history', 'treatment', 
                     'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 
                     'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 
                     'mental_health_interview', 'care_options']

# Collect value counts for each column
counts = [health_data[col].value_counts() for col in cols_to_visualize]

# Define a refined darker palette
color_palette = [
    '#E07A5F',  # Terra Cotta
    '#81B29A',  # Sage Green
    '#F2CC8F',  # Sandy Gold
    '#3D405B',  # Charcoal
    '#F4A261'   # Peach Orange
]

# Create a 4x4 grid using GridSpec
fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(4, 4, figure=fig)

# Generate pie charts
for i, (col, count) in enumerate(zip(cols_to_visualize, counts)):
    ax = fig.add_subplot(gs[i])  # Create a subplot in the ith grid cell
    ax.pie(
        count,
        labels=count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=color_palette[:len(count)]  # Use palette based on the number of unique values
    )
    ax.set_title(col, fontsize=14, fontweight='bold', color='darkslategray')
    ax.grid(False)

# Adjust layout
plt.tight_layout()
plt.show()


# %%

bright_palette = ["#FFB74D", "#64B5F6", "#81C784", "#E57373", "#FFD54F", "#4DD0E1"]

# Set the Seaborn style to 'whitegrid' for a clean look
sns.set_style("whitegrid")

# Create the countplot
plt.figure(figsize=(20, 10))
ax = sns.countplot(x='Country', data=health_data, palette=bright_palette)  # Brighter custom palette

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right", fontsize=12)

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fontsize=14, padding=8, color='black', weight='bold')

# Add a clean, modern title and axis labels
ax.set_title('Distribution of Countries', fontsize=22, fontweight='bold', color='teal')
ax.set_xlabel('Country', fontsize=18, color='darkslategray')
ax.set_ylabel('Count', fontsize=18, color='darkslategray')

# Adjust the gridlines to be subtle and light
ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Improve layout to prevent clipping of labels
plt.tight_layout()
plt.show()


# %% [markdown]
# Growing stress vs all variables now

# Define the color palette
color_palette = {
    0: 'rgba(144, 238, 144, 0.6)',  # Light Green for Growing Stress 0 with transparency
    1: 'rgba(255, 0, 0, 0.6)'        # Red for Growing Stress 1 with transparency
}

# Label mapping dictionary
label_mappings = {
    'Growing_Stress': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Gender': {0: 'Male', 1: 'Female'},
    'self_employed': {0: 'No', 1: 'Yes'},
    'family_history': {0: 'No', 1: 'Yes'},
    'treatment': {0: 'No', 1: 'Yes'},
    'Days_Indoors' : {7.5: '1-14 days' , 22.5: '15-30 days', 45:'31-60 days', 60: 'More than 2 months', 365: 'Go out Every day'},
    'Changes_Habits': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Mood_Swings': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Mental_Health_History': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Coping_Struggles': {0: 'No', 1: 'Yes'},
    'Work_Interest': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Social_Weakness': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'mental_health_interview': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'care_options': {0: 'No', 1: 'Yes', 2: 'Not sure'}
}



# Initialize a list to hold figures
figures = []

# Loop through each feature in health_data
# Loop through each feature in health_data
for feature in health_data.columns:
    if feature not in ['Growing_Stress', 'Country']:  # Skip the Growing_Stress column
        # Calculate counts of Growing Stress for each category
        counts = health_data.groupby([feature, 'Growing_Stress']).size().unstack(fill_value=0)
        
        # Calculate percentages
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Create a stacked bar chart
        fig = go.Figure()
        
        # Add traces for each Growing Stress value
        for stress_value in percentages.columns:
            fig.add_trace(go.Bar(
                x=percentages.index.map(lambda x: label_mappings[feature].get(x, x) if feature in label_mappings else x),  # Apply label mapping only if feature is in label_mappings
                y=percentages[stress_value],
                name=f'Growing Stress: {("No" if stress_value == 0 else "Yes")}',  # Change legend labels
                marker_color=color_palette[stress_value],
                text=[f"{val:.1f}%" for val in percentages[stress_value]],  # Percentage text
                textposition='inside'  # Center the text inside the bars
            ))
        
        # Update layout for each figure
        fig.update_layout(
            title=f'Growing Stress Distribution by {feature}',
            xaxis_title=feature,
            yaxis_title='Percentage',
            barmode='stack',
            legend_title='Growing Stress',
            template='plotly_white',
            height=400,  # Adjust height for better visualization
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        # Append the figure to the list
        figures.append(fig)

# Show all figures
for fig in figures:
    fig.show()


#%%[markdown] - Haeyeon
# Insights from ditributions

# Growing Stress vs Timestamp
#The distribution differences across the years 2014, 2015, and 2016 are not significant. In all three years, the proportion of respondents answering "Yes" is slightly higher.

# Growing Stress vs Gender:
# Within the same gender, for female, the proportion of those who answered "Yes" to growing stress is higher than that of men.

# Growing Stress vs Occupation:
# Among occupations, the highest proportion of "Yes" responses regarding growing stress is seen in the business, followed by the student occupation.

# Growing Stress vs Self-employed

# Growing Stress vs Family history

# Growing stress vs Treatment

# Growing Stress vs Days_Indoors

# Growing Stress vs Changes Habits

# Growing Stress vs Mental Health History

# Growing Stress vs Mood Swings

# Growing Stress vs Coping Struggles

# Growing Stress vs Work Interest

# Growing Stress vs Social Weakness

# Growing Stress vs Mental Health Interview

# Growing Stress vs Care options

#%%[markdown]
# One - Hot Encoding the Occupation column
final_df = health_data.drop(columns=['Country'])
final_df['Occupation'] = final_df['Occupation'].astype(str)

# One-Hot Encoding using Pandas
categorical_columns = final_df.select_dtypes(include=['object']).columns.tolist()
occupation_encoded = pd.get_dummies(final_df['Occupation'], prefix='Occupation', drop_first=True)
encoded_final_df = pd.concat([final_df.drop('Occupation', axis=1), occupation_encoded], axis=1)
encoded_final_df.head()

# %%[markdown]
# Correlation Matrix

# Calculate Spearman correlation matrix
correlation_matrix, _ = spearmanr(encoded_final_df, axis=0)

# Convert the result to a DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_matrix, index=encoded_final_df.columns, columns=encoded_final_df.columns)

# Set up the matplotlib figure
plt.figure(figsize=(14, 10))

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_df, dtype=bool))

# Create a heatmap with the mask
sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, 
            square=True, mask=mask, linewidths=.5)

# Set titles and labels
plt.title('Spearman Correlation Matrix of Encoded Final DataFrame (Lower Half)', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()



# %%[markdown]
# It seems that the treatment and Family History have high correlation,
# Except those, all the other variables have do not seem to have a linear relationship with Growing Stress.
# Only Mental Health History has a slight correlation with Growing Stress

# %%[markdown]
# Statistical tests
from scipy.stats import chi2_contingency, chi2

cols= ['Timestamp','Gender','Country','Occupation','self_employed','family_history', 'treatment', 'Days_Indoors',
                    'Changes_Habits', 'Mental_Health_History', 'Mood_Swings','Coping_Struggles', 'Work_Interest', 'Social_Weakness','mental_health_interview','care_options']
def calculate_chi_square(column1,column2 = 'Growing_Stress'):
    print(f"Correlation between **{column1}** and **{column2}**\n")
    crosstab = pd.crosstab(health_data[column1],health_data[column2])
    stat,p,dof,expected = chi2_contingency(crosstab, correction = True)
    print(f'P_value = {p}, degrees of freedom =  {dof}')
    prob = 0.95
    critical = chi2.ppf(prob,dof)
    print(f'probability = %.3f, critical = %.3f, stat = %.3f' %(prob,critical,stat) )
    if stat >= critical:
        print('dependent(reject(Ho))')
    else:
        print('independent(accept(Ho))')
    alpha = 1.0 - prob
    print(f'significance = %.3f, p-value = %.3f' %(alpha,p))
    if p <= alpha:
        print('dependent(reject(Ho))')
    else:
        print('independent(accept(Ho))')
    print('\n-----------------------------------\n')
print('** Chi_square Correlation between Dichotomous features with Target:Growing stress**\n')
for col in cols:
    calculate_chi_square(col)


#%%[markdown]
# As the Timestamp, Country, self-employed are independednt of Growing Stress and also have a highly unbalanced data
# it is better to remove those columns for our Modelling.

#%%[markdown]
# Modelling

encoded_final_df = encoded_final_df.drop(columns=['Timestamp', 'self_employed'])
data = encoded_final_df.drop('Growing_Stress', axis=1)
f_data = encoded_final_df['Growing_Stress']
X = data
y = f_data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)
methodDict = {}

#%%
logit = LogisticRegression()  # instantiate
logit.fit( Xtrain, Ytrain )
print('Logit model accuracy (with the test set):', logit.score(Xtest, Ytest))
print('Logit model accuracy (with the train set):', logit.score(Xtrain, Ytrain))


# %%[markdown]
# Evaluation of the model
def evalClassModel(model, y_test, y_pred_class, plot=False):
    #Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    # examine the class distribution of the testing set (using a Pandas Series method)
    print('Null accuracy:\n', y_test.value_counts())
    
    # calculate the percentage of ones
    print('Percentage of ones:', y_test.mean())
    
    # calculate the percentage of zeros
    print('Percentage of zeros:',1 - y_test.mean())
    
    #Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])
    
    #Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    #Metrics computed from a confusion matrix
    #Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))
    
    #False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)
    
    #Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))
    
    
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))
    
    # calculate cross-validated AUC
    # Change cross valisation to k -fold cross vaidation
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
    
    ##########################################
    #Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    print('First 10 predicted responses:\n', model.predict(Xtest)[0:10])

    # print the first 10 predicted probabilities of class membership
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(Xtest)[0:10])

    # print the first 10 predicted probabilities for class 1
    model.predict_proba(Xtest)[0:10, 1]
    
    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(Xtest)[:, 1]
    
    if plot == True:
        # histogram of predicted probabilities
        plt.rcParams['font.size'] = 12
        plt.hist(y_pred_prob, bins=8)
        
        # x-axis limit from 0 to 1
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')
    
    
    # predict treatment if the predicted probability is greater than 0.3
    # it will return 1 for all values above 0.3 and 0 otherwise
    # results are 2D so we slice out the first column
    y_pred_prob = y_pred_prob.reshape(-1,1) 
    y_pred_class = binarize(y_pred_prob, threshold=0.3)[0]
    
    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])
    
    ##########################################
    #ROC Curves and Area Under the Curve (AUC)
    ##########################################
    
    #AUC is the percentage of the ROC plot that is underneath the curve
    #Higher value = better classifier
    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)
    
    

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()
        
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
    
    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(threshold):
        #Sensitivity: When the actual value is positive, how often is the prediction correct?
        #Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)
    
    
    
    return accuracy
#
#%%[markdown]
# Logistic regression

def logisticRegression():
    # train a logistic regression model on the training set
    logreg = LogisticRegression()
    logreg.fit(Xtrain, Ytrain)
    
    # make class predictions for the testing set
    y_pred_class = logreg.predict(Xtest)
    
    accuracy_score = evalClassModel(logreg, Ytest, y_pred_class, True)
    
    #Data for final graph
    methodDict['Log. Regression'] = accuracy_score * 100
    
logisticRegression()

#%%
model = LogisticRegression(max_iter=1000)

# Create the feature importances visualizer
visualizer = FeatureImportances(model)

# Fit the visualizer and the model
visualizer.fit(Xtrain, Ytrain)

# Show the visualization
visualizer.show()




# %%[markdown]
# %%[markdown]
# Desciion Tree Before Regularization

target = encoded_final_df['Growing_Stress']

# Drop the target variable from the features
features = encoded_final_df.drop(['Growing_Stress'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Initialize the Decision Tree Classifier
clf = DTC(criterion='entropy', random_state=0)

# Fit the model
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f'Accuracy: {accuracy}')

# Calculate and print residual deviance
resid_dev = log_loss(y_test, clf.predict_proba(X_test))
print(f'Residual Deviance: {resid_dev}')

# Plot the decision tree
ax = subplots(figsize=(12, 12))[1]
plot_tree(clf, feature_names=features.columns, ax=ax)

# %%[markdown]


# Section 2: Cross-Validation and Pruning
from sklearn.model_selection import KFold

# Calculate cost complexity pruning path
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)

# Initialize KFold for cross-validation
kfold = KFold(n_splits=5, random_state=1, shuffle=True)


# %%[markdown]
# Section 3: Grid Search for Optimal Alpha


# Grid search for optimal alpha
grid = GridSearchCV(clf, {'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)

# Best score from grid search
print(f'Best Grid Search Score: {grid.best_score_}')

# Visualize the best estimator
best_clf = grid.best_estimator_
ax = subplots(figsize=(12, 12))[1]
plot_tree(best_clf, feature_names=features.columns, ax=ax)


# %%[markdown]

# Section 4: Evaluating the Best Estimator
# Evaluate the best estimator
best_accuracy = accuracy_score(y_test, best_clf.predict(X_test))
print(f'Best Estimator Accuracy: {best_accuracy}')

# Generate confusion matrix for best estimator
best_confusion = confusion_matrix(y_test, best_clf.predict(X_test))
print("Best Estimator Confusion Matrix:")
print(best_confusion)

# %%[markdown]
# Though this looks like a pretty good model, it sems to be overfitting,
# better todo some regularisation by decreasing the depth of the tree

# %%[markdown]
# Descision Tree After Regularisation
# Step 1 : Fix depth of the tree and min_samples_split and leaf

# Define the target and features
target = encoded_final_df['Growing_Stress']
features = encoded_final_df.drop(['Growing_Stress'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# While going through all the depth values, this seems to give the best model
clf = DTC(criterion='entropy', 
          max_depth=8,              
          min_samples_split = 20,     
          min_samples_leaf = 10,       
          random_state=0)

# %%[markdown]
# Step 2 : Evaluation of the model using train and test logloss using 5-fold cross validation

# Track log loss for training and testing
train_log_losses = []
test_log_losses = []
max_log_loss_diff = float('-inf')  # Initialize to negative infinity
best_params_log_loss_diff = {}

# KFold for cross-validation
kfold = KFold(n_splits=5, random_state=1, shuffle=True)

# Perform K-Fold Cross-Validation to compute log loss
for train_index, test_index in kfold.split(features):
    X_fold_train, X_fold_test = features.iloc[train_index], features.iloc[test_index]
    y_fold_train, y_fold_test = target.iloc[train_index], target.iloc[test_index]
    
    clf.fit(X_fold_train, y_fold_train)
    
    # Log loss for training data
    train_pred_proba = clf.predict_proba(X_fold_train)
    train_log_loss = log_loss(y_fold_train, train_pred_proba)
    train_log_losses.append(train_log_loss)
    
    # Log loss for testing data
    test_pred_proba = clf.predict_proba(X_fold_test)
    test_log_loss = log_loss(y_fold_test, test_pred_proba)
    test_log_losses.append(test_log_loss)

    # Calculate the difference between training and testing log loss
    log_loss_diff = train_log_loss - test_log_loss
    if log_loss_diff > max_log_loss_diff:
        max_log_loss_diff = log_loss_diff
        best_params_log_loss_diff['train_log_loss'] = train_log_loss
        best_params_log_loss_diff['test_log_loss'] = test_log_loss
        best_params_log_loss_diff['params'] = (clf.max_depth, clf.min_samples_split, clf.min_samples_leaf)

# Average log losses
average_train_log_loss = np.mean(train_log_losses)
average_test_log_loss = np.mean(test_log_losses)

print(f'Average Training Log Loss: {average_train_log_loss}')
print(f'Average Testing Log Loss: {average_test_log_loss}')
print(f'Best Parameters for Max Log Loss Difference: {best_params_log_loss_diff}')

# Plotting training and testing log loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_log_losses) + 1), train_log_losses, marker='o', label='Training Log Loss')
plt.plot(range(1, len(test_log_losses) + 1), test_log_losses, marker='x', label='Testing Log Loss')
plt.title('Log Loss Comparison: Training vs Testing')
plt.xlabel('Fold Number')
plt.ylabel('Log Loss')
plt.legend()
plt.grid()
plt.show()

# %%[markdown]
# From this it could be observed that the model is not overfitting
# as the training and testing log loss are almost equal.

# %%[markdown]

# Step 3 : Grid Search for Optimal Alpha using best parameters from previous log loss
# Calculate cost complexity pruning path
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)

# Perform grid search for the optimal alpha
grid = GridSearchCV(
    DTC(criterion='entropy', random_state=0, max_depth=best_params_log_loss_diff['params'][0], 
        min_samples_split=best_params_log_loss_diff['params'][1], min_samples_leaf=best_params_log_loss_diff['params'][2]),  
    {'ccp_alpha': ccp_path.ccp_alphas},        
    refit=True, 
    cv=kfold, 
    scoring='accuracy'
)
grid.fit(X_train, y_train)

# Get the best estimator and evaluate it
best_clf = grid.best_estimator_
best_clf
# %%[markdown]

# Evaluate the model using optimal alpha by finding accuracy through k-fold cross validation
accuracies = []
for train_index, test_index in kfold.split(X_train):
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    best_clf.fit(X_fold_train, y_fold_train)
    accuracy = accuracy_score(y_fold_test, best_clf.predict(X_fold_test))
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)

# Print accuracies and mean accuracy
print(f'Accuracies from each fold: {accuracies}')
print(f'Mean Accuracy: {mean_accuracy}')

# Calculate ROC AUC for the best estimator
best_y_proba = best_clf.predict_proba(X_test)[:, 1]
best_roc_auc = roc_auc_score(y_test, best_y_proba)
print(f'Best Estimator ROC AUC Score: {best_roc_auc}')


# Visualize the best estimator
plt.figure(figsize=(12, 12))
plot_tree(best_clf, feature_names=features.columns, filled=True, class_names=True)
plt.show()

# %%[markdown]
# Step 4 :  Generating Confusion matrix and ROC plot
# Generate confusion matrix for best estimator

best_confusion = confusion_matrix(y_test, best_clf.predict(X_test))
print("Best Estimator Confusion Matrix:")
print(best_confusion)



# Calculate ROC AUC for the best estimator
best_y_proba = best_clf.predict_proba(X_test)[:, 1]
best_roc_auc = roc_auc_score(y_test, best_y_proba)
print(f'Best Estimator ROC AUC Score: {best_roc_auc}')

#%%
# Calculate FPR and TPR for ROC curve
fpr, tpr, _ = roc_curve(y_test, best_y_proba)

# Plotting ROC AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {best_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()
# %%[markdown]

# Extract feature importances from the best estimator
feature_importances = best_clf.feature_importances_

# Create a DataFrame to hold the feature names and their importance
importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(importance_df)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Decision Tree Classifier')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.grid()
plt.show()




#%%



























































#%%[markdown]
# EDA specifically for the Student occupation
# Filter data for 'Student' occupation
student_data = health_data[health_data['Occupation'] == 'Student']

# Check the first few rows of the filtered data
student_data.head()

#%%[markdown]
# Distribution of Days_Indoors for Students
# Plot distribution of 'Days_Indoors' for students
sns.histplot(student_data['Days_Indoors'], kde=True, color='skyblue')
plt.title('Distribution of Days Indoors for Students')
plt.xlabel('Days Indoors')
plt.ylabel('Frequency')
plt.show()

#%%[markdown]
# Treatment vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='treatment', data=student_data, palette='Set2')
plt.title('Treatment vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Changes in Habits vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Changes_Habits', data=student_data, palette='Set2')
plt.title('Changes in Habits vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')

# Move the legend (hue label) to the top-left
plt.legend(title='Changes in Habits', loc='upper left', bbox_to_anchor=(0, 1))
plt.show()


#%%[markdown]
# Coping Struggles vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Coping_Struggles', data=student_data, palette='Set2')
plt.title('Coping Struggles vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Social Weakness vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Social_Weakness', data=student_data, palette='Set2')
plt.title('Social Weakness vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Mood Swings vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Mood_Swings', data=student_data, palette='Set2')
plt.title('Mood Swings vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Gender vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Gender', data=student_data, palette='Set2')
plt.title('Gender vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Days Indoors vs Growing Stress for students
sns.boxplot(x='Growing_Stress', y='Days_Indoors', data=student_data, palette='Set2')
plt.title('Days Indoors vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Days Indoors')
plt.show()

#%%[markdown]
# Correlation heatmap for variables correlated with Growing Stress for students
student_corr = student_data[['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 
                             'Mood_Swings', 'Social_Weakness', 'treatment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(student_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Growing Stress and Related Variables for Students')
plt.show()


# %%
# Gender, occupation, family_history,treatment, Days_indoors, changes_habits, mental_health_history,
# mood_swings, coping_struggles, work_interest, social weakness, mental_health_interview, care_options
# have a significant p-value suggesting that the growing stress levels for these groups is different for
# different categories. We are also removing the gender variable as it has unequal distribution


#%%[markdown]
# Before the unknown factor to create the normal model without the unknown factor

#%%[markdown]
# Random Forest -haeyeon

# Create a copy of the health_data to avoid modifying the original
health_data_copy = health_data.copy()

# Filter data to only include students
health_data_students = health_data_copy[health_data_copy['Occupation'] == 'Student']

# Preprocessing the dataset
# Drop 'Timestamp' and other irrelevant columns for prediction
X = health_data_students.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1)  # Drop target and irrelevant columns
y = health_data_students['Growing_Stress']  # Target column

# Convert categorical columns to numeric (if any)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Important for Random Forest and other models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get predicted probabilities for each class
y_pred_proba = rf.predict_proba(X_test_scaled)

# One-vs-Rest approach: Binarize the target labels for multiclass classification
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Binarize the target labels (0, 1, 2)

# Compute AUC for each class (One-vs-Rest)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')

# Plot the ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(3):  # 3 classes: 0, 1, and 2
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i]):.2f})')

# Plot diagonal line for random classifier
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Set plot labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve (One-vs-Rest) for Students')
plt.legend(loc='lower right')
plt.show()

# Print the Macro AUC score
print(f"Macro AUC-ROC score for Students: {roc_auc:.2f}")
#%%[markdown]
# A Macro AUC-ROC score of 1.00, might be because of Overfitting, Data Imbalance(Growing_Stress (with values 0, 1, 2)
# We tried additioanl validation

#%%[markdown]
# Perform cross-validation (e.g., 5-fold)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc_ovr')
print(f'Cross-validated AUC-ROC scores: {cv_scores}')
print(f'Mean AUC-ROC score from cross-validation: {cv_scores.mean():.2f}')

# Check Class Distribution
print(y.value_counts())

# Confusion Matrix
# Generate confusion matrix
cm = confusion_matrix(y_test, rf.predict(X_test_scaled))
print(cm)

#%%[markdown]
# The AUC scores across folds range from 0.45 to 0.75, with a mean AUC of 0.62
# Class 2: 22,915 samples, Class 1: 21,424 samples, Class 0: 16,348 samples <- little imblanced, but not extreme
# Class 0 (Stress Level 0): The model has made no misclassifications, indicating that it is very confident when predicting the "no stress" class.
# Class 1 (Stress Level 1): The model is performing fairly well, with a few misclassifications (12).
# Class 2 (Stress Level 2): The model is performing quite well here too, with a very small number of misclassifications (17)



















#%%[markdown]
# We have a the "unknown factor" quantifies the portion of the variance in Growing_Stress that is unexplained by the selected features.
# (i) Calculate the weighted probability of all features, ensuring the total feature weights sum to 1 and the 
# conditional probability if found for each feature.
# (ii) If the total weights are less than 1, it indicates growing stress due to personal reasons. 
# (iii) This suggests that an unknown factor is impacting the growing stress. And create a new feature unknown factor.
# (iv) This will help us have a more certainity in Growing factor by utilising the unknown factor to bring the 
# Growing stress from 3 classes to 2 classes
# 
#%%[markdown]
#caluclate the conditional probability
cols = health_data[['Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Growing_Stress', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']]

def calculate_conditional_pro(feature1, feature2='Growing_Stress'):
    column = pd.crosstab(index=health_data[feature1], columns=health_data[feature2], normalize='columns')
    print(f"Probability of {feature1} given {feature2}:\n")
    print(f'Column conditional probability:\n{column}')
    print('\n------------------------------------------\n')
    return column

for con in cols:
    data = calculate_conditional_pro(con)
#%%[markdown]
# Create DataFrame
df = pd.DataFrame(health_data)

# Selected features
selected_features = ['Timestamp','Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Growing_Stress', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']

# Calculate the frequency of each feature value given each target value
conditional_probs = {feature: {} for feature in selected_features}

for feature in selected_features:
    for target in df['Growing_Stress'].unique():
        conditional_probs[feature][target] = {}
        for value in df[feature].unique():
            prob = df[(df[feature] == value) & (df['Growing_Stress'] == target)].shape[0] / df[df['Growing_Stress'] == target].shape[0]
            conditional_probs[feature][target][value] = prob

# Calculate the average conditional probability for each feature for 'Yes' and 'No'
avg_cond_probs = {}
total_avg_cond_prob = {}
for target in df['Growing_Stress'].unique():
    avg_cond_probs[target] = {}
    total_avg_cond_prob[target] = 0
    for feature in selected_features:
        avg_cond_probs[target][feature] = np.mean(list(conditional_probs[feature][target].values()))
        total_avg_cond_prob[target] += avg_cond_probs[target][feature]
weights = {}
for target in df['Growing_Stress'].unique():
    weights[target] = {}
    for feature in selected_features:
        weights[target][feature] = avg_cond_probs[target][feature] / total_avg_cond_prob[target]

# Create temporary columns for each feature probability
for target in df['Growing_Stress'].unique():
    for feature in selected_features:
        df[f'{feature}_prob_{target}'] = df[feature].map(conditional_probs[feature][target])

#%%
# Calculate the combined new feature as a weighted sum of the conditional probabilities for both 'Yes' and 'No'
df['New_Feature'] = sum(df[f'{feature}_prob_1'] * weights[1][feature] + df[f'{feature}_prob_0'] * weights[0][feature] for feature in selected_features)
df['unknown_factor'] = 1 - df['New_Feature']
pd.set_option('display.max_columns', None)
df.head()
#%%
# Keep only necessary columns
final_df = df[['Occupation',  'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options', 'unknown_factor','Growing_Stress']]

final_df.head()

#%%
# Group by Growing_Stress and calculate the mean of unknown_factor
mean_unknown_factor = final_df.groupby('Growing_Stress')['unknown_factor'].mean()
print(mean_unknown_factor)

print(final_df['unknown_factor'].median())
threshold = final_df['unknown_factor'].median()

print(threshold)



# %%[markdown] - how to explain unknown factor by utilizing some visualization ---- start from haeyeon

# 1. Distribution of Conditional Probabilities per Feature
# Visualize how the conditional probability of each feature varies across the different values of Growing_Stress
# Display the distribution of the conditional probabilities for each feature, so you can see how the features behave for each category of Growing_Stress (e.g., Yes, No).

def plot_conditional_probability(feature, conditional_probs):
    """
    Plot the distribution of conditional probabilities for a specific feature across Growing_Stress classes.
    """
    plt.figure(figsize=(10, 6))
    for target in conditional_probs[feature]:
        values = list(conditional_probs[feature][target].values())
        sns.histplot(values, kde=True, label=f'Growing_Stress {target}', color='blue' if target == 1 else 'orange')
    
    plt.title(f"Conditional Probability Distribution for {feature} by Growing_Stress")
    plt.xlabel('Conditional Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Example: Visualizing conditional probabilities for the 'Occupation' feature
plot_conditional_probability('Occupation', conditional_probs)


# %%[markdown]
# 2. Feature Weights Visualization
# how each feature contributes to explaining the variance in Growing_Stress by showing the weights for each feature.
# Show the weights calculated for each feature, indicating how much each feature contributes to the overall model

def plot_feature_weights(weights):
    """
    Plot feature weights, which represent how much each feature contributes to the model.
    """
    # Flatten the feature weights for plotting
    feature_names = list(weights[1].keys())
    feature_weights = [weights[1][feature] + weights[0][feature] for feature in feature_names]  # Sum of weights for both Yes and No

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=feature_weights, palette='viridis')
    plt.title('Feature Weights for Growing_Stress')
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Visualize feature weights for Growing_Stress
plot_feature_weights(weights)
print(final_df.columns)
# %%[markdown]
# 3. Unknown Factor Distribution
# visualize its distribution to see how much it contributes to each data point.
# Display the distribution of the unknown factor and compare it with the classification of Growing_Stress to see how the factor influences the classification

def plot_unknown_factor_distribution(df):
    """
    Plot the distribution of the 'unknown_factor' and how it relates to 'Growing_Stress' classifications.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['unknown_factor'], kde=True, color='green')
    
    plt.title('Distribution of Unknown Factor')
    plt.xlabel('Unknown Factor')
    plt.ylabel('Frequency')
    
    # Show how unknown_factor is related to Growing_Stress
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Growing_Stress', y='unknown_factor', data=df, palette='Set2')
    plt.title('Unknown Factor by Growing_Stress Class')
    plt.xlabel('Growing_Stress')
    plt.ylabel('Unknown Factor')
    plt.show()

# Visualize the unknown factor distribution and how it relates to Growing_Stress
plot_unknown_factor_distribution(final_df)

# %%[markdown]
# 4. Impact of Threshold on Classification
# visualize how the unknown_factor affects the binary classification of Growing_Stress
# plot a histogram of the unknown_factor values and use a threshold to split the data into 0 (No Stress) and 1 (Stress)
# Display how the threshold (set to the median) splits the data into two categories, allowing you to visualize how the unknown_factor affects classification.

def plot_threshold_impact(df, threshold):
    """
    Plot the impact of the threshold on Growing_Stress classification.
    """
    plt.figure(figsize=(10, 6))

    # Plot the histogram of unknown_factor
    sns.histplot(df['unknown_factor'], kde=True, color='purple', label='Unknown Factor', bins=30)

    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    
    # Display
    plt.title('Impact of Threshold on Growing_Stress Classification')
    plt.xlabel('Unknown Factor')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot the impact of the threshold on the classification of Growing_Stress
plot_threshold_impact(final_df, threshold)

#%% [markdown]
# 5. Final Classification Overview
# After the transformation of Growing_Stress based on the unknown factor, we can visualize the final class distribution of Growing_Stress (either 0 or 1), comparing it to the original categories
# Show the final distribution of Growing_Stress after applying the threshold to classify it into two categories (1 for stress, 0 for no stress)

def plot_final_classification(final_df):
    """
    Visualize the final classification of Growing_Stress (0 or 1).
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Growing_Stress', data=final_df, palette='pastel')
    plt.title('Final Classification of Growing_Stress')
    plt.xlabel('Growing_Stress')
    plt.ylabel('Count')
    plt.show()

# Plot the final classification of Growing_Stress
plot_final_classification(final_df)

#%% [markdown] -- haeyeon end of unknown factor visualization

#%%
# Set threshold based on the mean or a slightly higher value
  # or use another positive threshold based on analysis

# Convert Growing_Stress from 2 to 0 and 1 based on unknown_factor
final_df['Growing_Stress'] = final_df.apply(
    lambda row: 1 if row['unknown_factor'] > threshold else 0 
    if row['Growing_Stress'] == 2 else row['Growing_Stress'], axis=1
)
# Check the updated values
print(final_df['Growing_Stress'].value_counts())

#%%
final_df = final_df.drop('unknown_factor',axis = 1)

#%%[markdown]
# Random Forest
def randomForest():
    # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators = 20)

    featuresSize = Xtrain.shape[1]
    param_dist = {"max_depth": [3, None],
              "max_features": randint(1, featuresSize),
              "min_samples_split": randint(2, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
    
    # Building and fitting my_forest
    forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)
    my_forest = forest.fit(Xtrain, Ytrain)
    
    # make class predictions for the testing set
    y_pred_class = my_forest.predict(Xtest)
    
    accuracy_score = evalClassModel(my_forest, Ytest, y_pred_class, True)

    #Data for final graph
    methodDict['Random Forest'] = accuracy_score * 100

    feature_names = Xtrain.columns if hasattr(Xtrain, 'columns') else [f"Feature {i}" for i in range(Xtrain.shape[1])]
    plot_feature_importance(my_forest, feature_names)
randomForest()





# %%[makrdown]
# Feature importance and selection




# %%[markdown]
# As seen from the summary Gender plays a very important role and contributes the most to the Growing stress followed by
# Mental health interview and family history positive coefficent meaning the ones who have given the interview 
# and have a familty hisroty of mental health could have more growing stress





#%%SVM prep (Yonathan)

'''Need to encode occupation since SVM takes numerical variables'''

final_df_encoded = pd.get_dummies(final_df, columns=['Occupation'], drop_first=True)
#First line returns us more variables (breaks variable down) with boolean values (True/False)
#Need to convert into integer
final_df_encoded[final_df_encoded.select_dtypes(include='bool').columns] = final_df_encoded.select_dtypes(include='bool').astype(int)

#Check if it worked
print(final_df_encoded.dtypes)
#Yes it worked as of running the line (12/10). Now move forward to model building

X_clean = X.dropna()
y_clean = y[X_clean.index]


#%%
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


'''There were some missing values for X so I will remove them in the previous cell'''
X_clean = final_df_encoded.drop('Growing_Stress', axis=1).dropna()
y_clean = final_df_encoded['Growing_Stress'][X_clean.index]

#%%
#from sklearn.model_selection import train_test_split
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

#scaler = StandardScaler()
#Xtrain_scaled = scaler.fit_transform(Xtrain)
#Xtest_scaled = scaler.transform(Xtest)

#svm_model = SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
#svm_model.fit(Xtrain_scaled, Ytrain)

# Predictions
#y_pred_class = svm_model.predict(Xtest_scaled)
#y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]

# %%[markdown]
# Is SVM model ideal for the dataset? SVM is more ideal for smaller datasets
# Should I take a percentage of the data and test again to see if computation time lowers?

'''New code to take random sample(10%) of the data'''
X_clean_sampled = X_clean.sample(frac=0.1, random_state=42)
y_clean_sampled = y_clean[X_clean_sampled.index]


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean_sampled, y_clean_sampled, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(Xtrain_scaled, Ytrain)


# Predictions
y_pred_class = svm_model.predict(Xtest_scaled)
y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]

#%%[markdown]
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
accuracy = accuracy_score(Ytest, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")


print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(Ytest, y_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

auc_score = roc_auc_score(Ytest, y_pred_prob)
print(f"ROC-AUC: {auc_score:.2f}")

fpr, tpr, _ = roc_curve(Ytest, y_pred_prob)
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#Model has an accuracy of 0.68 and an AUC of 0.71. Let's try to see what parameters we can optimize to improve predictive power.

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(Xtrain_scaled, Ytrain)

print("Best Parameters:", grid_search.best_params_)

#Analysis and conclusion for SVM:
#The target variable was Growing Stress similar to other binary models in this document. 
#We've evaluated the performance of the model through metrics like accuracy (0.68) and AUC(0.71)
#SVM has presented us with computational challanges likely due to the number of features, one-hot encoding, and data size
#To mitigate that, we tried to tune our parameters (TBD) and did our SVM on a sample of the dataset(10%)
#The difficulty and output leaves us wondering if SVM is the ideal model to move forward with
# %%[]
