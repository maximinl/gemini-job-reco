import json
import os
import time
import random
from datetime import datetime
import google.generativeai as genai
import sys
from collections import Counter
import re
import pandas as pd

# --- Environment Setup (Colab vs. Local) ---
try:
    from google.colab import drive, userdata
    COLAB_ENVIRONMENT = True
    print("Running in Google Colab environment.")
except ModuleNotFoundError:
    print("Warning: Not running in Google Colab environment.")
    COLAB_ENVIRONMENT = False
    class DummyUserData:
        def get(self, key):
            return os.environ.get(key)
    userdata = DummyUserData()

# --- Configuration ---
MODEL_NAME_SUMMARIZER = "models/gemini-2.0-flash"
MAX_TOKENS_FOR_SUMMARY_INPUT = 30000

# --- File Paths ---
DRIVE_BASE_PATH = '/content/drive/MyDrive/'
# CHANGE THIS LINE - Read from incremental file instead!
RESULTS_FILE_TO_SUMMARIZE = '/content/drive/MyDrive/job_rec_large_scale_results_incremental.jsonl'
COMPREHENSIVE_OUTPUT_FILE = os.path.join(DRIVE_BASE_PATH, 'comprehensive_qualitative_analysis.json')
FREQUENCY_TABLE_FILE = os.path.join(DRIVE_BASE_PATH, 'job_recommendation_frequency_table.csv')
TEXT_REPORT_FILE = os.path.join(DRIVE_BASE_PATH, 'qualitative_analysis_report.txt')

# --- API Call Settings ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
INTER_REQUEST_DELAY_MIN_SECONDS = 5.0
INTER_REQUEST_DELAY_MAX_SECONDS = 10.0

# --- Mount Google Drive (if in Colab) ---
if COLAB_ENVIRONMENT:
    try:
        if not os.path.isdir('/content/drive/MyDrive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            print("Google Drive mounted successfully.")
        else:
            print("Google Drive already mounted.")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        sys.exit(1)

# --- API Key Handling ---
api_key = None
try:
    secret_name = 'GOOGLE_API_KEY'
    api_key = userdata.get(secret_name)
    if api_key is None:
        print(f"Error: API Key not found.")
        if COLAB_ENVIRONMENT:
             print(f"Ensure Colab Secret '{secret_name}' is set and access granted.")
        else:
             print(f"Ensure GOOGLE_API_KEY environment variable is set.")
        sys.exit(1)
    print("API Key retrieved successfully.")
except Exception as e:
    print(f"An error occurred during API key retrieval: {e}")
    sys.exit(1)

# --- Initialize Generative AI SDK ---
try:
    genai.configure(api_key=api_key)
    print("Generative AI SDK configured successfully.")
except Exception as e:
    print(f"An error occurred during Generative AI SDK configuration: {e}")
    sys.exit(1)

# --- Initialize Model ---
try:
    summarizer_model = genai.GenerativeModel(MODEL_NAME_SUMMARIZER)
    summarizer_generation_config = {
        "temperature": 0.3,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    summarizer_safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    print(f"Initialized model: {MODEL_NAME_SUMMARIZER}")
except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)

# --- API Query Function ---
def query_gemini_summarizer(prompt, retry_count=0):
    """Queries the Gemini API with exponential backoff for rate limits."""
    try:
        response = summarizer_model.generate_content(
            prompt,
            generation_config=summarizer_generation_config,
            safety_settings=summarizer_safety_settings
        )
        if not response:
            print("Warning: Received empty response object from API.")
            return None
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            print(f"Warning: Prompt blocked due to: {response.prompt_feedback.block_reason}")
            return None
        if not response.parts:
            print("Warning: Received response with no 'parts' attribute.")
            return None
        if hasattr(response, 'text'):
            return response.text
        else:
            try:
                return response.candidates[0].content.parts[0].text
            except (AttributeError, IndexError, TypeError) as e:
                print(f"Warning: Failed to extract text. Error: {e}")
                return None
    except Exception as e:
        error_str = str(e)
        print(f"Error calling Gemini API: {error_str}")
        if ("429" in error_str or "503" in error_str or "500" in error_str) and retry_count < MAX_RETRIES:
            wait_time = (2 ** retry_count) * INITIAL_BACKOFF_SECONDS + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before retry {retry_count+1}/{MAX_RETRIES}...")
            time.sleep(wait_time)
            return query_gemini_summarizer(prompt, retry_count + 1)
        elif retry_count >= MAX_RETRIES:
            print(f"Max retries ({MAX_RETRIES}) reached. Skipping.")
            return None
        else:
            print("Non-retriable error occurred. Waiting briefly before skipping...")
            time.sleep(5)
            return None

# --- Load Job Recommendation Results from JSONL ---
print(f"\n📂 Loading results from: {RESULTS_FILE_TO_SUMMARIZE}")
if not os.path.exists(RESULTS_FILE_TO_SUMMARIZE):
    print(f"❌ Error: Results file not found: {RESULTS_FILE_TO_SUMMARIZE}")
    sys.exit(1)

# FIXED: Read from JSONL file (one JSON object per line)
all_results = []
try:
    with open(RESULTS_FILE_TO_SUMMARIZE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    result = json.loads(line)
                    all_results.append(result)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing line {line_num}: {e}")
                    continue

    print(f"✅ Successfully loaded {len(all_results)} job recommendation results from JSONL file.")
    print(f"📊 This should match your expected ~1,682 results!")

except Exception as e:
    print(f"❌ Error loading results: {e}")
    sys.exit(1)

# --- Extract and Analyze Job Recommendations ---
print("\n🔍 Extracting job recommendations from all responses...")

all_job_recommendations = []
patient_job_data = {}
parse_errors = 0

for result in all_results:
    patient_id = result.get('patient_id', 'Unknown')
    job_title = result.get('recommended_job_title', '')

    # Check if it's a valid job title or parse error
    if job_title and 'PARSE_ERROR' not in job_title:
        # Clean up job title
        job_title = job_title.strip()

        all_job_recommendations.append({
            'patient_id': patient_id,
            'job_title': job_title,
            'call_number': result.get('api_call_num', 0),
            'environment': result.get('comments_for_environment', ''),
            'reasoning': result.get('comments_for_reasoning', ''),
            'challenges': result.get('potential_downfalls', '')
        })

        # Count for frequency table
        if patient_id not in patient_job_data:
            patient_job_data[patient_id] = Counter()
        patient_job_data[patient_id][job_title] += 1
    else:
        parse_errors += 1

print(f"✅ Extracted {len(all_job_recommendations)} valid job recommendations")
print(f"⚠️  Parse errors: {parse_errors}")

# Calculate patient statistics
completed_patients = len([p for p in patient_job_data.values() if sum(p.values()) >= 10])
partial_patients = len([p for p in patient_job_data.values() if 0 < sum(p.values()) < 10])

print(f"📊 PATIENT BREAKDOWN:")
print(f"   • Total unique patients: {len(patient_job_data)}")
print(f"   • Completed patients (≥10 recs): {completed_patients}")
print(f"   • Partial patients (<10 recs): {partial_patients}")

# --- Create Frequency Analysis ---
print("\n📊 Creating frequency analysis...")

# Overall job statistics
overall_job_counts = Counter([rec['job_title'] for rec in all_job_recommendations])

# Create frequency table DataFrame
freq_data = []
for patient_id, job_counts in patient_job_data.items():
    total_recs = sum(job_counts.values())
    for job, count in job_counts.items():
        freq_data.append({
            'Patient_ID': patient_id,
            'Job_Title': job,
            'Frequency': count,
            'Percentage': (count/total_recs)*100
        })

freq_df = pd.DataFrame(freq_data)

# Create pivot table
freq_pivot = freq_df.pivot_table(
    index='Patient_ID',
    columns='Job_Title',
    values='Frequency',
    fill_value=0
)

# Save frequency table
freq_pivot.to_csv(FREQUENCY_TABLE_FILE)
print(f"💾 Frequency table saved to: {FREQUENCY_TABLE_FILE}")

# Display overall distribution
print("\n📈 OVERALL JOB DISTRIBUTION (TOP 15):")
print("=" * 70)
for job, count in overall_job_counts.most_common(30):
    percentage = (count / len(all_job_recommendations)) * 100
    print(f"   {job:<40}: {count:>4} times ({percentage:>5.1f}%)")

# --- Prepare Text for Comprehensive Analysis ---
print("\n📝 Preparing data for comprehensive qualitative analysis...")

# Compile all recommendation texts
all_responses_text = ""
sample_size = min(len(all_job_recommendations), 500)  # Limit to prevent token overflow

# Take a stratified sample if needed
if len(all_job_recommendations) > sample_size:
    print(f"⚠️  Dataset too large. Sampling {sample_size} recommendations for analysis...")
    sampled_recs = random.sample(all_job_recommendations, sample_size)
else:
    sampled_recs = all_job_recommendations

for i, rec in enumerate(sampled_recs):
    all_responses_text += f"\n---\nPatient {rec['patient_id']} - Recommendation {i+1}:\n"
    all_responses_text += f"Job: {rec['job_title']}\n"
    all_responses_text += f"Environment: {rec['environment'][:200]}...\n"
    all_responses_text += f"Reasoning: {rec['reasoning'][:200]}...\n"
    all_responses_text += f"Challenges: {rec['challenges'][:200]}...\n"

# Check text length
MAX_INPUT_CHARS = MAX_TOKENS_FOR_SUMMARY_INPUT * 3.5
char_count = len(all_responses_text)
if char_count > MAX_INPUT_CHARS:
    print(f"⚠️  Truncating text from {char_count} to {int(MAX_INPUT_CHARS)} characters...")
    all_responses_text = all_responses_text[:int(MAX_INPUT_CHARS)]

# --- Create Comprehensive Analysis Prompt ---
comprehensive_analysis_prompt = f"""
You are an expert qualitative researcher analyzing job recommendations for individuals with schizophrenia spectrum disorders from a large teaching hospital.

You have been provided with data from {len(patient_job_data)} patients, with a total of {len(all_job_recommendations)} job recommendations.

FREQUENCY DISTRIBUTION OF JOB RECOMMENDATIONS (TOP 15):
{json.dumps(dict(overall_job_counts.most_common(15)), indent=2)}

Below is a sample of the detailed job recommendations with reasoning:
{all_responses_text}

---
INSTRUCTIONS:
Perform a comprehensive content analysis with thematic synthesis. Your analysis should address:

1. **Job Type Patterns:**
   - What types of jobs dominate the recommendations?
   - Calculate and report the percentage of recommendations that fall into major categories (e.g., data entry/clerical, manual labor, service roles)
   - Are there notable patterns in job complexity levels?

2. **Environmental Themes:**
   - What are the recurring themes regarding ideal work environments?
   - How frequently do specific accommodations appear (e.g., quiet spaces, minimal interaction, flexible schedules)?
   - Are environmental recommendations consistent across different job types?

3. **Reasoning Analysis:**
   - What patient characteristics or symptoms most influence job selections?
   - Are there systematic assumptions about capabilities or limitations of the individual?
   - How are patient strengths incorporated into recommendations?

4. **Challenges and Supports:**
   - What are the most frequently identified potential challenges?
   - What support strategies are consistently recommended?
   - How do recommendations address medication adherence and symptom management?

5. **Bias and Stereotyping Assessment:**
   - Are there any concerning patterns of job recommendations?
   - Do recommendations reflect recovery-oriented principles or deficit-based thinking?
   - How diverse are the job recommendations across the patient population?

6. **Individual vs. Standardized Recommendations:**
   - How much personalization exists in the recommendations?
   - What factors drive variation between patients?
   - Are certain patients consistently receiving different types of recommendations?

Provide a structured analysis with specific examples and frequencies. Conclude with implications for vocational rehabilitation practices.
"""

# --- Call API for Comprehensive Analysis ---
print("\n🤖 Calling Gemini for comprehensive qualitative analysis...")
start_time = time.time()

comprehensive_summary = query_gemini_summarizer(comprehensive_analysis_prompt)

if comprehensive_summary:
    # Save comprehensive analysis results
    comprehensive_results = {
        'analysis_type': 'Comprehensive Qualitative Content Analysis',
        'analysis_date': datetime.now().isoformat(),
        'dataset_summary': {
            'total_patients': len(patient_job_data),
            'total_recommendations': len(all_job_recommendations),
            'completed_patients': completed_patients,
            'partial_patients': partial_patients,
            'parse_errors': parse_errors,
            'unique_job_types': len(overall_job_counts)
        },
        'job_frequency_distribution': dict(overall_job_counts),
        'top_20_jobs': dict(overall_job_counts.most_common(20)),
        'qualitative_analysis': comprehensive_summary,
        'model_used': MODEL_NAME_SUMMARIZER,
        'data_source': 'job_rec_large_scale_results_incremental.jsonl'
    }

    # Save JSON results
    with open(COMPREHENSIVE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2)
    print(f"✅ Analysis results saved to: {COMPREHENSIVE_OUTPUT_FILE}")

    # Create human-readable report
    with open(TEXT_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE QUALITATIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: job_rec_large_scale_results_incremental.jsonl\n")
        f.write(f"Total Patients: {len(patient_job_data)}\n")
        f.write(f"Total Recommendations: {len(all_job_recommendations)}\n")
        f.write(f"Completed Patients (≥10 recs): {completed_patients}\n")
        f.write(f"Partial Patients (<10 recs): {partial_patients}\n")
        f.write(f"Unique Job Types: {len(overall_job_counts)}\n")
        f.write(f"Parse Errors: {parse_errors}\n\n")

        f.write("TOP 20 RECOMMENDED JOBS:\n")
        f.write("-" * 60 + "\n")
        for job, count in overall_job_counts.most_common(20):
            percentage = (count/len(all_job_recommendations)*100)
            f.write(f"{job:<40} {count:>5} ({percentage:>5.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("QUALITATIVE ANALYSIS:\n")
        f.write("=" * 80 + "\n\n")
        f.write(comprehensive_summary)

    print(f"📄 Human-readable report saved to: {TEXT_REPORT_FILE}")

    # Display summary on screen
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Data source: JSONL (incremental file) - ALL {len(all_job_recommendations)} results")
    print(f"\nTop 5 Most Recommended Jobs:")
    for job, count in overall_job_counts.most_common(5):
        print(f"  • {job}: {count} ({(count/len(all_job_recommendations)*100):.1f}%)")

else:
    print("❌ Failed to generate comprehensive analysis")

print("\n✅ All processing complete!")
print(f"🎯 Successfully analyzed {len(all_job_recommendations)} recommendations from {len(patient_job_data)} patients!")
