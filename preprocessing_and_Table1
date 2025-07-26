import pandas as pd
import numpy as np

# Read the data from the CSV file
df = pd.read_csv('/content/drive/MyDrive/bq-results-20250619-094026-1750326149078/bq-results-20250619-094026-1750326149078.csv')

# Convert DataFrame to JSON and save to the specified path
df.to_json('/content/drive/MyDrive/jobrecocases.json', orient='records')

print("CSV successfully converted and saved as JSON to: /content/drive/MyDrive/jobrecocases.json")

import pandas as pd
import numpy as np
from collections import Counter

# --- Pre-processing: Get unique patients and aggregate data ---
# First, let's get visit counts per patient
visit_counts_per_patient = df.groupby('subject_id').size()

# Get one row per patient (using first visit for most characteristics)
df_patients = df.groupby('subject_id').first().reset_index()

# Add visit counts to patient data
df_patients['num_visits'] = df_patients['subject_id'].map(visit_counts_per_patient)

# --- Grouping Race Categories ---
if 'race' in df_patients.columns:
    def group_race(race_string):
        race_string = str(race_string).upper()
        if 'WHITE' in race_string:
            return 'WHITE'
        elif 'BLACK/AFRICAN AMERICAN' in race_string or 'BLACK/AFRICAN' in race_string or 'BLACK/CARIBBEAN ISLAND' in race_string or 'BLACK/CAPE VERDEAN' in race_string:
            return 'BLACK/AFRICAN AMERICAN'
        elif 'HISPANIC' in race_string or 'LATINO' in race_string or 'SOUTH AMERICAN' in race_string:
            return 'HISPANIC/LATINO'
        elif 'ASIAN' in race_string:
            return 'ASIAN'
        elif 'UNKNOWN' in race_string or pd.isna(race_string) or race_string == '':
            return 'UNKNOWN'
        else:
            return 'OTHER'
    df_patients['grouped_race'] = df_patients['race'].apply(group_race)
else:
    df_patients['grouped_race'] = 'UNKNOWN' # Default if race column is missing


# --- Table 1 generation ---
table = []

# === PATIENT OVERVIEW ===
total_patients = len(df_patients)
table.append({"Characteristic": "Total patients", "Value": f"{total_patients}"})
table.append({"Characteristic": "", "Value": ""})

# Age (from first visit)
if 'age' in df_patients.columns:
    age_mean = df_patients['age'].mean()
    age_std = df_patients['age'].std()
    age_n = df_patients['age'].notna().sum()
    table.append({"Characteristic": "Age (SD)", "Value": f"{age_mean:.1f} \u00B1 {age_std:.1f} ($N={age_n}$)"})

table.append({"Characteristic": "", "Value": ""})

# === DEMOGRAPHICS ===
# Sex (from first visit)
if 'sex' in df_patients.columns:
    table.append({"Characteristic": "Sex", "Value": ""})
    sex_counts = df_patients['sex'].value_counts()

    # Sort sex for consistent order (M first, then F)
    sex_order = ['M', 'F'] if 'M' in sex_counts.index and 'F' in sex_counts.index else sex_counts.index
    for sex_val in sex_order:
        if sex_val in sex_counts.index:
            count = sex_counts[sex_val]
            pct = (count / total_patients) * 100
            table.append({"Characteristic": sex_val, "Value": f"{pct:.1f}%\t{count}"})

    table.append({"Characteristic": "", "Value": ""})

# Grouped Race (from first visit) - sort by frequency of the grouped category
table.append({"Characteristic": "Race", "Value": ""})
grouped_race_counts = df_patients['grouped_race'].value_counts()

# Define a desired order for grouped races for consistent output
race_display_order = [
    'WHITE',
    'BLACK/AFRICAN AMERICAN',
    'HISPANIC/LATINO',
    'ASIAN',
    'OTHER',
    'UNKNOWN'
]

# Iterate through the desired order
for race_group in race_display_order:
    if race_group in grouped_race_counts.index:
        count = grouped_race_counts[race_group]
        pct = (count / total_patients) * 100
        table.append({"Characteristic": race_group, "Value": f"{pct:.1f}%\t{count}"})

# Add any 'OTHER' categories that might have been missed by the explicit mapping but grouped as 'OTHER'
# This handles cases where original race values that were not explicitly mapped above fall into 'OTHER'
for race_group in grouped_race_counts.index:
    if race_group not in race_display_order: # If there are any categories that weren't in our explicit order
        count = grouped_race_counts[race_group]
        pct = (count / total_patients) * 100
        table.append({"Characteristic": race_group, "Value": f"{pct:.1f}%\t{count}"})


table.append({"Characteristic": "", "Value": ""})

# === CLINICAL CHARACTERISTICS ===
# Number of Visits per Patient
table.append({"Characteristic": "Number of Visits", "Value": ""})
visit_frequency = visit_counts_per_patient.value_counts().sort_index()

for num_visits in sorted(visit_frequency.index):
    count = visit_frequency[num_visits]
    pct = (count / total_patients) * 100
    table.append({"Characteristic": str(num_visits), "Value": f"{pct:.1f}%\t{count}"})

table.append({"Characteristic": "", "Value": ""})

# Create clean, formatted output
output_lines = []
output_lines.append("Table 1: Patient Cohort Characteristics")
output_lines.append("=" * 60)
output_lines.append("")

# Create formatted table with logical sections
table1_df = pd.DataFrame(table)

for _, row in table1_df.iterrows():
    characteristic = row['Characteristic']
    value = row['Value']

    if characteristic == "" and value == "":
        output_lines.append("")  # Add blank line
    elif value == "":
        output_lines.append(f"{characteristic}")  # Section header
        output_lines.append("")  # Add space after section header
    else:
        # Handle the spacing for the table format
        if '\t' in str(value):
            parts = str(value).split('\t')
            if len(parts) == 2:
                # Format: "  Item name                     12.3%      123"
                output_lines.append(f"  {characteristic:<35} {parts[0]:>8} {parts[1]:>8}")
            else:
                output_lines.append(f"  {characteristic:<35} {value}")
        else:
            # For continuous variables like Age, Length of Stay
            output_lines.append(f"  {characteristic:<35} {value}")

# Print the formatted table
formatted_output = "\n".join(output_lines)
print(formatted_output)

# Save to text file
with open('table1_patient_cohort.txt', 'w') as f:
    f.write(formatted_output)

# Also create a clean CSV version
csv_data = []
current_section = ""

for _, row in table1_df.iterrows():
    characteristic = row['Characteristic']
    value = row['Value']

    if value == "" and characteristic != "":
        current_section = characteristic
    elif '\t' in str(value):
        parts = str(value).split('\t')
        if len(parts) == 2:
            csv_data.append({
                'Section': current_section,
                'Characteristic': characteristic,
                'Percentage': parts[0],
                'Count': parts[1]
            })
    elif value != "" and characteristic != "":
        # For continuous variables (like Age)
        csv_data.append({
            'Section': current_section,
            'Characteristic': characteristic,
            'Value': value,
            'Percentage': '',
            'Count': ''
        })

csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('table1_patient_cohort.csv', index=False)

print("\n" + "="*60)
print("✓ Table saved to: table1_patient_cohort.txt")
print("✓ Data saved to: table1_patient_cohort.csv")
print("="*60)

# Brief summary
print(f"\nSummary Statistics:")
print(f"  • Total unique patients: {total_patients:,}")
print(f"  • Total visits: {len(df):,}")
print(f"  • Average visits per patient: {visit_counts_per_patient.mean():.1f}")
if 'age' in df_patients.columns:
    print(f"  • Mean age: {df_patients['age'].mean():.1f} \u00B1 {df_patients['age'].std():.1f} years")


print("\nFrequencies of Sex:")
print(df['sex'].value_counts())

print("\nFrequencies of Age:")
# Depending on whether 'age' is numerical or categorical, you might want to bin it first
# For now, let's assume it's numerical and show some summary stats or value counts for unique ages
if df['age'].dtype in ['int64', 'float64']:
  print(df['age'].describe())
  # If you still want frequencies of each unique age:
  # print(df['age'].value_counts().sort_index())
else:
  print(df['age'].value_counts())


print("\nFrequencies of Race:")
print(df['race'].value_counts())

print("\nFrequencies of Diagnosis:")
print(df['diagnosis'].value_counts())
