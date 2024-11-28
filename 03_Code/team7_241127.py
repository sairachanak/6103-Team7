#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional
import altair as alt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

import os


#%%
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# health_data = pd.read_csv("Mental_Health_Dataset.csv")

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
# %%
print(health_data.head)
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
#Verifying the cleaned dataset
health_data.describe()
print(health_data.info())
# %%[markdown]
# Visualisation for Proportion of each feature in dataset


sns.set_style("whitegrid")

# Columns to visualize
cols_to_visualize = ['Timestamp', 'Gender', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                     'Coping_Struggles', 'Work_Interest', 'Social_Weakness']

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

# Create subplots dynamically based on the number of columns
n_cols = 4
n_rows = (len(cols_to_visualize) + n_cols - 1) // n_cols  # Calculate required rows dynamically
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

# Generate pie charts
for i, (col, count) in enumerate(zip(cols_to_visualize, counts)):
    axs[i].pie(
        count,
        labels=count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=color_palette[:len(count)]  # Use palette based on the number of unique values
    )
    axs[i].set_title(col, fontsize=14, fontweight='bold', color='darkslategray')
    axs[i].grid(False)

# Remove any unused subplots
for j in range(len(cols_to_visualize), len(axs)):
    fig.delaxes(axs[j])

# Adjust layout
plt.tight_layout()
plt.show()



# %%


bright_palette = ["#FFB74D", "#64B5F6", "#81C784", "#E57373", "#FFD54F", "#4DD0E1"]

# Set the Seaborn style to 'whitegrid' for a clean look
sns.set(style="whitegrid")

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

# %%[markdown]
# Growing stress vs all variables now

#%%[markdown]
#Growing stress by gender
df = pd.crosstab(health_data.Growing_Stress,health_data.Gender).plot(kind = 'bar',color = ['m','c'])
df.grid(False)
df.set_title('Growing stress by Gender ')

#%%[markdown]
# Most of the males seems to have a 'Maybe' category in Growing stress and most of the
# female have 'Yes' category in Growing category 
#%%[markdown]

# Group the data by 'Growing_Stress' and 'Days_Indoors' and count the occurrences
Icount = health_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

# Create the bar plot using Altair
import altair as alt

chart = alt.Chart(Icount).mark_bar().encode(
    x='Count:Q',
    y='Days_Indoors:N',
    color='Growing_Stress:N',
    tooltip=['Growing_Stress', 'Count']
).properties(
    title='Distribution of Growing Stress by Days Indoors',
    width=600,
    height=400
).configure_mark(
    opacity=0.7  # Subtle opacity for the bars
).configure_title(
    fontSize=18, 
    font='Arial', 
    anchor='middle', 
    color='gray'
)

# Show the chart
chart.show()
# %%[markdown]
# Growing stress by Days spent in indoors

count_data = health_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

custom_palette = ['#FF6F81',
                  '#6DAED9',
                  '#D17000']

# Create the barplot
plt.figure(figsize=(15, 8))
ax = sns.barplot(
    data=count_data,
    x='Days_Indoors',
    y='Count',
    hue='Growing_Stress',
    palette=custom_palette
)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add bar labels for clarity
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)

# Set titles and labels
ax.set_title('Distribution of Growing Stress by Days Indoors', fontsize=18, color='darkslategray', weight='bold')
ax.set_xlabel('Days Indoors', fontsize=14, color='gray')
ax.set_ylabel('Count', fontsize=14, color='gray')
ax.legend(title='Growing Stress', fontsize=12, title_fontsize=12)

# Remove gridlines for a cleaner look
sns.despine()
plt.tight_layout()

# Show the plot
plt.show()

#%%[markdown]
# Growing stress vs occupation

count_data = health_data.groupby(['Growing_Stress', 'Occupation']).size().reset_index(name='Count')

# Set Seaborn style
sns.set_style("whitegrid")

# Create custom color palette
custom_palette = {'No': '#FF6F81', 'Yes': '#6DAED9', 'Maybe': '#D17000'}

# Create the bar plot
plt.figure(figsize=(15, 8))
ax = sns.barplot(
    data=count_data,
    x='Count',  # Count for the x-axis
    y='Occupation',  # Occupation for the y-axis
    hue='Growing_Stress',  # Hue for 'Growing_Stress'
    palette=custom_palette
)

# Add labels to the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)

# Customize plot title and labels
ax.set_title('Distribution of Growing Stress by Occupations', fontsize=18, fontweight='bold', color='darkslategray')
ax.set_xlabel('Count', fontsize=14, color='gray')
ax.set_ylabel('Occupation', fontsize=14, color='gray')
ax.legend(title='Growing Stress', fontsize=12, title_fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
#%%[markdown]
# Corporate employees report the highest levels of growing stress, while housewives and students experience the 
# most uncertainty, often categorized as 'maybe' stress
# %%[markdown]
# Growing stress vs work interest

color_map = {'Yes': '#FF6F81', 'No': '#6DAED9', 'Maybe': '#D17000'}

# Create the plot
plt.figure(figsize=(8, 8))
ax1 = sns.countplot(x='Work_Interest', hue='Growing_Stress', data=health_data, palette=color_map)

# Add title and remove grid
plt.title('Growing Stress by Work Interest')
plt.grid(False)

# Add labels to bars
for container in ax1.containers:
    ax1.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

#%%[markdown]
# The people with maybe work interest seem to have the highest growing stress followed by 
# people with 'yes' work interest 
# %%[markdown]
# The growing stress vs chnages_habits

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Changes_Habits', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Changes_Habits')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%[markdown]
# The changes_habits 'maybe' category has the highest growing stress population who doesnot change habits.

# %%[markdown]
# Growing stress vs Self_employed
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='self_employed', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Cself employed')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%[markdown]
# The data reveals that non-self-employed individuals exhibit higher levels of growing stress compared to no growing stress. 
# Conversely, self-employed individuals display a balanced distribution, with equal proportions experiencing growing stress and no growing stress.

#%%[markdown]
# Growing stress vs Family history
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='family_history', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by family history')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]
# The data reveals that individuals with family history exhibit higher levels of growing stress compared to no growing stress. 
# and individuals with no family history also display high levels of growing stress to no growing stress.
#%%[markdown]
# Growing stress vs Treatment

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='treatment', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by treatment')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%[markdown]
# The data reveals that individuals who had treatment exhibit higher levels of growing stress compared to no growing stress and the 
# individuals with no treatment also display a higher levels of growing stress to no growing stress.

#%%[markdown]

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Mood_Swings', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by treatment')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%[markdown]
# The people with medium stress levels have highest growing stress compared to people with low and high mood 

#%%[markdown]
# Growing stress vs coping struggles
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Coping_Struggles', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Coping struggles')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%[markdown]
# Surprisingly the people who donot struggle with coping exhbit more growing stress levels

#%%[markdown]

# Growing stress vs Care options
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='care_options', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by care options')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]
# The people with care options exhibit growing stress levels compared to others 

#%%[markdown]
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Mental_Health_History', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Mental health history')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]

# The people with mental health history have growing stress compared to other people.

#%%[markdown]
# Growing stress vs Timestamp
health_data['Timestamp'].unique()
#%%

# Group by 'Timestamp' and 'Growing_Stress', then count occurrences
grouped_data = health_data.groupby(['Timestamp', 'Growing_Stress']).size().reset_index(name='Count')

# Map the Growing_Stress categories to readable names (for the legend)
category_map = {'No':0, 'Yes':1, 'Maybe':2}
grouped_data['Growing_Stress_Label'] = grouped_data['Growing_Stress'].map(category_map)

# Create the lineplot
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='Timestamp',
    y='Count',
    hue='Growing_Stress_Label',  # Hue based on the mapped Growing_Stress categories
    data=grouped_data,
    palette={0: '#6DAED9', 1: '#FF6F81', 2: '#D17000'}
)

# Customize the plot
plt.title('Growing Stress Over Time by Category', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count of Growing Stress Cases', fontsize=12)
plt.legend(title='Growing Stress', fontsize=10, title_fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()



#%%[markdown]
# Growing stress vs Social_weakness

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Social_Weakness', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Social_Weakness')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
#%%[markdown]
# The people who have social weakness have growing stress compared to others

# %%[markdown]
# Additional Visualization(EDA) added by haeyeon-start

#%%[markdown]
# Treatment vs Gender
 
plt.figure(figsize=(6, 4))  # create countplot
sns.countplot(x='Gender', hue='treatment', data=health_data, palette="viridis")
plt.title("Treatment vs Gender")
plt.show()

#%%[markdown]
# Treatment vs Family History

plt.figure(figsize=(6, 4))   # create countplot
sns.countplot(x='family_history', hue='treatment', data=health_data, palette="viridis")
plt.title("Treatment vs Family History")
plt.show()

#%%[markdown]
# Treatment vs Self-Employed

plt.figure(figsize=(6, 4))   # create countplot
sns.countplot(x='self_employed', hue='treatment', data=health_data, palette="viridis")
plt.title("Treatment vs Self-Employed")
plt.show()

#%%[markdown]
# Treatment vs Mental Health History

# create countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='Mental_Health_History', hue='treatment', data=health_data, palette="viridis")
plt.title("Treatment vs Mental Health History")
plt.show()

#%%[markdown]
# Treatment vs Days Indoors

health_data['treatment_cat'] = health_data['treatment'].astype('category') # Change 'treatment' as categorical

# Create a cross-tabulation to see proportions
crosstab = pd.crosstab(health_data['Days_Indoors'], health_data['treatment_cat'])

# Plot stacked bar chart
crosstab.div(crosstab.sum(axis=1), axis=0).plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette("Set2"))
plt.title("Proportions of Treatment for Each Days Indoors Category")
plt.xlabel("Days Indoors")
plt.ylabel("Proportion")
plt.legend(title="Treatment", loc='upper left')
plt.show()

#%%[markdown]
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
# %%
# Gender, occupation, family_history,treatment, Days_indoors, changes_habits, mental_health_history,
# mood_swings, coping_struggles, work_interest, social weakness, mental_health_interview, care_options
# have a significant p-value suggesting that the growing stress levels for these groups is different for
# different categories

#%%[markdown]
# We have a the "unknown factor" quantifies the portion of the variance in Growing_Stress that is unexplained by the selected features.
# (i) Calculate the weighted probability of all features, ensuring the total feature weights sum to 1.
# (ii) If the total weights are less than 1, it indicates growing stress due to personal reasons. 
# (iii) This suggests that an unknown factor is impacting the growing stress. And create a new feature unknown factor.
# 
#%%[markdown]
#caluclate the conditional probability
cols = health_data[['Gender', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
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

# Calculate the combined new feature as a weighted sum of the conditional probabilities for both 'Yes' and 'No'
df['New_Feature'] = sum(df[f'{feature}_prob_Yes'] * weights['Yes'][feature] + df[f'{feature}_prob_No'] * weights['No'][feature] for feature in selected_features)
df['unknown_factor'] = 1 - df['New_Feature']
# Keep only necessary columns
final_df = df[['Gender','Occupation',  'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options', 'unknown_factor','Growing_Stress']]

final_df.head()

# %%
from sklearn.model_selection import train_test_split

data = final_df.drop('Growing_Stress',axis = 1)
f_data = final_df['Growing_Stress']
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data,f_data,test_size = 0.2,random_state = 42)
Xtrain.head()
# %%

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
#         ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier(random_state=42))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
# %%
from sklearn.model_selection import train_test_split

data = final_df.drop('Growing_Stress',axis = 1)
f_data = final_df['Growing_Stress']
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data,f_data,test_size = 0.2,random_state = 42)
Xtrain.head()
# %%

# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

#%%
# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%
# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                                                           
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%
# Importing necessary libraries
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data with dense output
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Set sparse_output=False for dense output
])

# Bundle preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the HistGradientBoostingClassifier model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%[markdown]
# Model 2 - haeyeon

# SMART Question 
# What factors most significantly contribute to individuals seeking mental health treatment?

# %% # Clean and preprocess the data
# Convert categorical variables to numeric using Label Encoding or One-Hot Encoding
label_cols = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 
              'treatment', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 
              'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles', 
              'Work_Interest', 'Social_Weakness', 'mental_health_interview', 
              'care_options', 'treatment_cat']

# Initialize LabelEncoder
le = LabelEncoder()

for col in label_cols:
    health_data[col] = le.fit_transform(health_data[col])

# Drop the Timestamp column as it is irrelevant to the model
X = health_data.drop(columns=['treatment','treatment_cat', 'Timestamp'])
y = health_data['treatment']

# Handle missing data by imputing missing values
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for models like SVM and KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%[markdown]
# Train and evaluate the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

# %%[markdown]
# KNN Accuracy: 66.08% , The model correctly predicted the target class about 66% of the time
# %%
# Train and evaluate the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, log_reg_pred))

# %%[markdown]
# Logistic Regression Accuracy: 70.19%, The model correctly predicted the target class around 70% of the time, which is an improvement over the KNN model (66.08%).

# %%[markdown]
# Try Model Improvement - KNN
# This part of the code takes about 9 minutes to execute, so it has been commented out.
# After running code below, the best KNN Parameters: {'n_neighbors': 11, 'metric': 'euclidean'} 

from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid for KNN
# param_dist_knn = {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan']}

# Create RandomizedSearchCV objects
# random_search_knn = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_dist_knn, n_iter=10, cv=5, random_state=42)

# Fit RandomizedSearchCV
# random_search_knn.fit(X_train, y_train)

# Print best parameters
# print("Best KNN Parameters:", random_search_knn.best_params_)

#%%[markdown]
# Best KNN Parameters: {'n_neighbors': 11, 'metric': 'euclidean'}

# Try KNN Model with n_neighbors=11

# Use the best parameters found from GridSearchCV
knn_best11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')

# Train the KNN model on the training data
knn_best11.fit(X_train, y_train)

# Make predictions on the test data
knn_pred11 = knn_best11.predict(X_test)

# Evaluate the performance
print("KNN Accuracy:", accuracy_score(y_test, knn_pred11))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred11))


#%%[markdown]
# The updated KNN model has accuracy of 68.36% which is little better than previous model using n_neighbors=5(66.08%)
# %%
# Try Model Improvement - Logistic Regression

# Define the parameter grid for Logistic Regression
param_dist_logreg = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}

# Create RandomizedSearchCV objects
random_search_logreg = RandomizedSearchCV(LogisticRegression(), param_distributions=param_dist_logreg, n_iter=10, cv=5, random_state=42)

# Fit RandomizedSearchCV
random_search_logreg.fit(X_train, y_train)

# Print best parameters
print("Best Logistic Regression Parameters:", random_search_logreg.best_params_)

#%%[markdown]
# The best Logistic Regression Parameters: {'solver': 'liblinear', 'C': 1}

# Try Logistic Regression model with best parameters

log_reg_best2 = LogisticRegression(solver='liblinear', C=1, random_state=42)
log_reg_best2.fit(X_train, y_train)
log_reg_pred_best = log_reg_best2.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy with Best Parameters:", accuracy_score(y_test, log_reg_pred_best))
print("Logistic Regression Classification Report with Best Parameters:\n", classification_report(y_test, log_reg_pred_best))
# %%[markdown]
# after fine-tuning the parameters for Logistic Regression, the performance might not have significantly improved

# %%