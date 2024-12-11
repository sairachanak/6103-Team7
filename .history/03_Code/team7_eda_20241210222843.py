# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import altair as alt

# %%
# Load the dataset
import os


#%%
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# health_data = pd.read_csv("Mental_Health_Dataset.csv")

# Construct the full path to the CSV file
csv_file_path = os.path.join(current_dir, "Mental Health Dataset.csv")

# Read the CSV file
mental_health_data = pd.read_csv(csv_file_path)

print("\nReady to continue.")
# %%
# Display the first few rows and basic information of the dataset for an overview
mental_health_data_info = mental_health_data.info()
mental_health_data_head = mental_health_data.head()

print(mental_health_data_info) 
print(mental_health_data_head)

# %%

# Check for missing values
missing_values = mental_health_data.isnull().sum()

# Fill missing values in 'self_employed' with 'Unknown' since it's categorical
mental_health_data['self_employed'].fillna('Unknown', inplace=True)

# Convert categorical columns  to lowercase for consistency
categorical_columns = mental_health_data.select_dtypes(include='object').columns
mental_health_data[categorical_columns] = mental_health_data[categorical_columns].apply(lambda x: x.str.strip().str.lower())

# Check unique values in key columns for inconsistencies
unique_values = {
    column: mental_health_data[column].unique()
    for column in ['Gender', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mood_Swings']
}

missing_values, unique_values

# %%
# Convert the Timestamp column to datetime and extract the year
mental_health_data['Timestamp'] = pd.to_datetime(mental_health_data['Timestamp'], errors='coerce')
mental_health_data['Year'] = mental_health_data['Timestamp'].dt.year

# Check for any invalid datetime conversions
invalid_timestamps = mental_health_data['Timestamp'].isnull().sum()

# Assess the impact of dropping rows with missing 'self_employed'
rows_with_missing_self_employed = mental_health_data['self_employed'].isnull().sum()
remaining_rows_if_dropped = len(mental_health_data) - rows_with_missing_self_employed

# Provide details about the size of the dataset after dropping rows
invalid_timestamps, rows_with_missing_self_employed, remaining_rows_if_dropped


# %%
# Distribution of Mental Health Treatment Across Countries
treatment_by_country = mental_health_data['Country'].value_counts().reset_index()
treatment_by_country.columns = ['Country', 'Count']
treatment_by_country = treatment_by_country.head(1)
chart1 = alt.Chart(treatment_by_country).mark_bar().encode(
    x=alt.X('Country:N', sort='-y', title='Country'),  
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Country:N', scale=alt.Scale(scheme='tableau10'), legend=None),  # Improved color palette
    tooltip=['Country', 'Count']
).properties(
    title='Top 10 Countries by Mental Health Treatment Responses',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart1.show()
# %%[markdown]
# It shows that US has a larger number of  mental health treatment response than other countries

# %%
#Family History vs. Seeking Treatment
family_vs_treatment = mental_health_data.groupby(['family_history', 'treatment']).size().reset_index(name='Count')

chart2 = alt.Chart(family_vs_treatment).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('family_history:N', title='Family History'),
    color=alt.Color('treatment:N', title='Treatment'),
    tooltip=['family_history', 'treatment', 'Count']
).properties(
    title='Family History vs. Seeking Treatment',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart2.show()
# %%[markdown]
# This bar graph shows that individuals with a family history of mental health issues are more likely to seek treatment, as indicated by the larger orange bar for "yes" in the "Family History" group. 
# However, even those without a family history also seek treatment in significant numbers, suggesting other factors influence treatment-seeking behavior.

# %%
#Days Indoors and Growing Stress
stress_by_days = mental_health_data.groupby(['Days_Indoors', 'Growing_Stress']).size().reset_index(name='Count')

chart3 = alt.Chart(stress_by_days).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Days_Indoors:N', title='Days Indoors'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress'),
    tooltip=['Days_Indoors', 'Growing_Stress', 'Count']
).properties(
    title='Days Indoors vs. Growing Stress Levels',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)
chart3.show()
# %%[markdown]
# Spending long periods indoors (over two months) strongly correlates with higher stress levels, 
# while going out daily appears to reduce stress. Shorter indoor durations (1-30 days) show mixed stress outcomes.
# %%
# Mental Health History vs. Mood Swings
history_vs_moods = mental_health_data.groupby(['Mental_Health_History', 'Mood_Swings']).size().reset_index(name='Count')

chart4 = alt.Chart(history_vs_moods).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Mental_Health_History:N', title='Mental Health History'),
    color=alt.Color('Mood_Swings:N', title='Mood Swings'),
    tooltip=['Mental_Health_History', 'Mood_Swings', 'Count']
).properties(
    title='Mental Health History vs. Mood Swings',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart4.show()

# %%[markdown]
# Individuals with a mental health history tend to report higher levels of medium mood swings 
# compared to those without a history or with uncertain mental health history.

# %%
#Self-Employment vs. Care Options
self_employment_vs_care = mental_health_data.groupby(['self_employed', 'care_options']).size().reset_index(name='Count')

chart5 = alt.Chart(self_employment_vs_care).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('self_employed:N', title='Self-Employment'),
    color=alt.Color('care_options:N', title='Care Options'),
    tooltip=['self_employed', 'care_options', 'Count']
).properties(
    title='Self-Employment vs. Care Options',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart5.show()
# %%[markdown]
# unemployed individuals are more likely to report awareness or access to care options, while self-employed individuals face lack of access to care options

# %%
#Changes in Habits vs. Coping Struggles
habits_vs_coping = mental_health_data.groupby(['Changes_Habits', 'Coping_Struggles']).size().reset_index(name='Count')

chart6 = alt.Chart(habits_vs_coping).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Changes_Habits:N', title='Changes in Habits'),
    color=alt.Color('Coping_Struggles:N', title='Coping Struggles'),
    tooltip=['Changes_Habits', 'Coping_Struggles', 'Count']
).properties(
    title='Changes in Habits vs. Coping Struggles',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart6.show()
# %%[markdown]
# Individuals who report changes in habits are the most likely to experience coping struggles, while those without changes in habits show slightly fewer struggles.

# %%
#Social Weakness vs. Mental Health Interview Openness
social_vs_interview = mental_health_data.groupby(['Social_Weakness', 'mental_health_interview']).size().reset_index(name='Count')

chart7 = alt.Chart(social_vs_interview).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Social_Weakness:N', title='Social Weakness'),
    color=alt.Color('mental_health_interview:N', title='Interview Openness'),
    tooltip=['Social_Weakness', 'mental_health_interview', 'Count']
).properties(
    title='Social Weakness vs. Mental Health Interview Openness',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    anchor='start',
    font='Arial',
    color='gray'
)

chart7.show()
# %%[markdown]
# there is a strong trend of reluctance to participate in mental health interviews, with very few individuals expressing openness


# %%
import altair as alt

# Ensure Altair renderer is correctly set (depends on the environment)
alt.renderers.enable('default')
students_data = mental_health_data[mental_health_data['Occupation'].str.contains("student", case=False, na=False)]
students_data['Stress_Level'] = students_data['Growing_Stress'].map({'yes': 1, 'no': 0, 'maybe': 0.5})

# %%
#Distribution of Stress Levels
stress_distribution = students_data.groupby(['Growing_Stress']).size().reset_index(name='Count')

chart1 = alt.Chart(stress_distribution).mark_bar().encode(
    x=alt.X('Growing_Stress:N', title='Growing Stress Level'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Growing_Stress:N', legend=None),
    tooltip=['Growing_Stress', 'Count']
).properties(
    title='Distribution of Stress Levels Among Students',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)
chart1.show()
# %%[markdown]
# A large number of students either experience growing stress levels or are unsure about their stress status, with relatively few feeling no stress growth.

# %%
#Stress vs Days Indoors
stress_vs_days = students_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

chart2 = alt.Chart(stress_vs_days).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Days_Indoors:N', title='Days Indoors', sort='-x'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress Level'),
    tooltip=['Growing_Stress', 'Days_Indoors', 'Count']
).properties(
    title='Distribution of Growing Stress by Days Indoors',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)

chart2.show()
# %%[markdown]
# Students who spend more time indoors, especially over extended periods (more than 2 months), is strongly associated with growing stress levels. Conversely, individuals who go out daily report significantly lower stress

# %%
#Stress Levels vs Mood Swings
stress_vs_moods = students_data.groupby(['Growing_Stress', 'Mood_Swings']).size().reset_index(name='Count')

chart3 = alt.Chart(stress_vs_moods).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Mood_Swings:N', title='Mood Swings'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress Level'),
    tooltip=['Growing_Stress', 'Mood_Swings', 'Count']
).properties(
    title='Stress Levels vs. Mood Swings Among Students',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)

chart3.show()

# %%[markdown]
# Students experiencing high and medium mood swings are more likely to report growing stress levels, while those with low mood swings exhibit a more even distribution of stress responses, suggesting mood stability is linked to reduced stress.

# %%
# Changes in Habits vs. Stress Levels
habits_vs_stress = students_data.groupby(['Growing_Stress', 'Changes_Habits']).size().reset_index(name='Count')

chart4 = alt.Chart(habits_vs_stress).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Changes_Habits:N', title='Changes in Habits'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress Level'),
    tooltip=['Growing_Stress', 'Changes_Habits', 'Count']
).properties(
    title='Changes in Habits vs. Stress Levels Among Students',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)
chart4.show()
# %%[markdown]
# Students who report habit changes are more likely to experience growing stress levels. Conversely, students with no habit changes are less likely to report growing stress
# %%
# Stress Trends Over Time
# Convert 'Timestamp' to datetime and extract the year
students_data['Timestamp'] = pd.to_datetime(students_data['Timestamp'], errors='coerce')
students_data['Year'] = students_data['Timestamp'].dt.year
# Verify if the 'Year' column exists and has valid values
print(f"Unique Years: {students_data['Year'].unique()}")

stress_over_time = students_data.groupby(['Year', 'Growing_Stress']).size().reset_index(name='Count')

chart5 = alt.Chart(stress_over_time).mark_line(point=True).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress Level'),
    tooltip=['Year', 'Growing_Stress', 'Count']
).properties(
    title='Stress Trends Over Time Among Students',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)

chart5.show()

# %%[markdown]
# Stress levels among students appear to decrease over time, with significant reductions in all categories from 2014 to 2016.

# %%
# Social Weakness vs. Stress Levels
social_vs_stress = students_data.groupby(['Growing_Stress', 'Social_Weakness']).size().reset_index(name='Count')

chart6 = alt.Chart(social_vs_stress).mark_bar().encode(
    x=alt.X('Count:Q', title='Count'),
    y=alt.Y('Social_Weakness:N', title='Social Weakness'),
    color=alt.Color('Growing_Stress:N', title='Growing Stress Level'),
    tooltip=['Growing_Stress', 'Social_Weakness', 'Count']
).properties(
    title='Social Weakness vs. Stress Levels Among Students',
    width=600,
    height=400
).configure_title(
    fontSize=18,
    font='Arial',
    anchor='start',
    color='gray'
)

chart6.show()
# %%[markdown]
# Social weakness appears to be strongly linked to growing stress levels, with students experiencing social weakness more likely to report stress. However, even students without social weakness report notable levels of growing stress


# %%
# model building
# Filtering students based on 'Occupation' and create  a copy of student data
students_data = mental_health_data[mental_health_data['Occupation'].str.contains("student", case=False, na=False)].copy()
# Normalize 'Growing_Stress' column using .loc
students_data.loc[:, 'Growing_Stress'] = students_data['Growing_Stress'].str.lower()

students_data.loc[:, 'Stress_Binary'] = students_data['Growing_Stress'].map({'yes': 1, 'no': 0, 'maybe': 0})
# Drop rows with NaN in 'Stress_Binary'
students_data_cleaned = students_data.dropna(subset=['Stress_Binary'])


# Define features and target
features = ['Days_Indoors', 'Mood_Swings', 'Changes_Habits', 'Year']
target = 'Stress_Binary'
X = students_data_cleaned[features]
y = students_data_cleaned[target]


# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Days_Indoors', 'Mood_Swings', 'Changes_Habits']),
        ('num', StandardScaler(), ['Year'])
    ])

#Define Ridge Classifier Pipeline
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])

#Train the Ridge Regression Model
ridge_model.fit(X_train, y_train)

#Predictions and Evaluation
y_pred = ridge_model.predict(X_test)
y_pred_proba = ridge_model.decision_function(X_test)  

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# AUC
auc = roc_auc_score(y_test, y_pred_proba)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Ridge (AUC = {auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Ridge Classifier')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#Evaluation Metrics results
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")

# %%[markdown]
# The ROC curve shows the performance of the Ridge Classifier, with an AUC (Area Under Curve) of 0.67 and an accuracy of 0.64. Since the AUC is below 0.8, the model struggles to effectively distinguish between classes.

# %%
