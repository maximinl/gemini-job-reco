# Job Recommendation Script using Gemini 2.0 Flash - RESUMABLE VERSION
import json
import os
import time
import random
import re
from datetime import datetime
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
import sys
import traceback
from collections import defaultdict

try:
    from google.colab import drive, userdata
    COLAB_ENVIRONMENT = True
except ModuleNotFoundError:
    print("Warning: Not running in Google Colab environment.")
    COLAB_ENVIRONMENT = False
    class DummyUserData:
        def get(self, key):
            return os.environ.get(key)
    userdata = DummyUserData()

# --- Configuration for Large-Scale Analysis ---
MODEL_NAME = "models/gemini-2.0-flash"  # Using Gemini 2.0 Flash
RUN_COUNT = 3  # 10 iterations per patient as requested
MAX_PATIENTS_PER_DAY = 150  # With 1500 daily limit: 150 patients × 10 runs = 1500 requests
DAILY_REQUEST_LIMIT = 1500  # Actual free tier limit - 1500 requests per day

# --- File Paths ---
# Assumes you are running in Google Colab
DRIVE_BASE_PATH = '/content/drive/MyDrive/'
PATIENT_JSON_FILE = '/content/drive/MyDrive/jobrecocases.json'
RESULTS_FILE_INCREMENTAL = os.path.join(DRIVE_BASE_PATH, 'job_rec_large_scale_results_incremental.jsonl')
RESULTS_FILE_FINAL = os.path.join(DRIVE_BASE_PATH, 'job_rec_large_scale_results_final.json')
QUOTA_TRACKER_FILE = os.path.join(DRIVE_BASE_PATH, 'daily_quota_tracker_large_scale.json')
PROGRESS_TRACKER_FILE = os.path.join(DRIVE_BASE_PATH, 'progress_tracker.json')

# --- Optimized API Call Settings ---
MAX_RETRIES = 3
# Efficient delays for 15 RPM (1 request every 4 seconds)
INTER_REQUEST_DELAY_MIN_SECONDS = 4.2  # Just above 4 seconds minimum
INTER_REQUEST_DELAY_MAX_SECONDS = 5.0   # Efficient but safe
SAVE_INCREMENTALLY = True

# Global variables
results = []
incremental_file_handle = None
total_requests_today = 0

# --- Mount Google Drive ---
if COLAB_ENVIRONMENT:
    try:
        if not os.path.isdir('/content/drive/MyDrive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
        if not os.path.exists(DRIVE_BASE_PATH):
            os.makedirs(DRIVE_BASE_PATH)
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        sys.exit(1)
else:
    # Fallback for local execution
    DRIVE_BASE_PATH = '.'
    PATIENT_JSON_FILE = 'subset_100_cases.json' # Make sure this file exists locally
    RESULTS_FILE_INCREMENTAL = os.path.join(DRIVE_BASE_PATH, 'job_rec_2_0_flash_results_incremental.jsonl')
    RESULTS_FILE_FINAL = os.path.join(DRIVE_BASE_PATH, 'job_rec_2_0_flash_results_final.json')
    QUOTA_TRACKER_FILE = os.path.join(DRIVE_BASE_PATH, 'daily_quota_tracker_2_0.json')
    PROGRESS_TRACKER_FILE = os.path.join(DRIVE_BASE_PATH, 'progress_tracker.json')
    if not os.path.exists(DRIVE_BASE_PATH):
         os.makedirs(DRIVE_BASE_PATH)

# --- Progress Tracking Functions ---
def analyze_existing_results():
    """Analyze existing results to determine what's been completed."""
    completed_patients = set()
    patient_run_counts = defaultdict(int)

    if not os.path.exists(RESULTS_FILE_INCREMENTAL):
        print("📝 No existing results file found. Starting fresh.")
        return completed_patients, patient_run_counts, 0

    total_existing_results = 0
    try:
        with open(RESULTS_FILE_INCREMENTAL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    patient_id = str(result.get('patient_id', ''))
                    if patient_id:
                        patient_run_counts[patient_id] += 1
                        total_existing_results += 1
                        if patient_run_counts[patient_id] >= RUN_COUNT:
                            completed_patients.add(patient_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"⚠️ Error reading existing results: {e}")
        return set(), defaultdict(int), 0

    print(f"📊 Existing Progress Analysis:")
    print(f"   • Total existing results: {total_existing_results}")
    print(f"   • Fully completed patients: {len(completed_patients)}")
    print(f"   • Patients with partial results: {len(patient_run_counts) - len(completed_patients)}")

    return completed_patients, patient_run_counts, total_existing_results

def save_progress_tracker(completed_patients, patient_run_counts, last_processed_index):
    """Save progress tracking information."""
    progress_data = {
        'completed_patients': list(completed_patients),
        'patient_run_counts': dict(patient_run_counts),
        'last_processed_index': last_processed_index,
        'last_updated': datetime.now().isoformat(),
        'run_count_target': RUN_COUNT
    }
    try:
        with open(PROGRESS_TRACKER_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save progress tracker: {e}")

def load_progress_tracker():
    """Load progress tracking information."""
    if not os.path.exists(PROGRESS_TRACKER_FILE):
        return set(), defaultdict(int), 0

    try:
        with open(PROGRESS_TRACKER_FILE, 'r') as f:
            data = json.load(f)
        completed_patients = set(data.get('completed_patients', []))
        patient_run_counts = defaultdict(int, data.get('patient_run_counts', {}))
        last_processed_index = data.get('last_processed_index', 0)
        return completed_patients, patient_run_counts, last_processed_index
    except Exception as e:
        print(f"⚠️ Error loading progress tracker: {e}")
        return set(), defaultdict(int), 0

# --- Quota Management Functions ---
def load_quota_tracker():
    """Load daily quota usage tracker"""
    global total_requests_today
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(QUOTA_TRACKER_FILE):
        try:
            with open(QUOTA_TRACKER_FILE, 'r') as f:
                tracker = json.load(f)
            if tracker.get('date') == today:
                total_requests_today = tracker.get('requests', 0)
                print(f"📊 Today's API requests so far: {total_requests_today}")
            else:
                total_requests_today = 0
                save_quota_tracker()
        except:
            total_requests_today = 0
    else:
        total_requests_today = 0
        save_quota_tracker()

def save_quota_tracker():
    """Save daily quota usage"""
    today = datetime.now().strftime("%Y-%m-%d")
    tracker = {
        'date': today,
        'requests': total_requests_today,
        'model': MODEL_NAME,
        'last_updated': datetime.now().isoformat()
    }
    try:
        with open(QUOTA_TRACKER_FILE, 'w') as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save quota tracker: {e}")

def check_quota_available():
    """Check if we can make more requests today"""
    return total_requests_today < DAILY_REQUEST_LIMIT

def increment_quota():
    """Increment the daily request counter"""
    global total_requests_today
    total_requests_today += 1
    save_quota_tracker()

# --- API Key Setup ---
def get_api_key():
    api_key = None
    if COLAB_ENVIRONMENT:
        try:
            api_key = userdata.get('GOOGLE_API_KEY')
            if api_key:
                print("✅ API Key found in Colab userdata.")
                return api_key
        except Exception as e:
            print(f"Could not access Colab userdata: {e}")

    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        print("✅ API Key found in environment variables.")
        return api_key

    return None

try:
    api_key = get_api_key()
    if api_key is None:
        print("❌ Error: GOOGLE_API_KEY not found.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    print(f"✅ API Key configured for {MODEL_NAME}")
except Exception as e:
    print(f"❌ API key setup error: {e}")
    sys.exit(1)

# Load quota tracker
load_quota_tracker()

# --- Load Patient Data ---
# Check if df is already loaded in the environment
if 'df' in globals() and isinstance(df, pd.DataFrame) and len(df) > 0:
    print(f"📊 Using existing DataFrame 'df' with {len(df)} patient cases")
    data_source = "existing DataFrame"
else:
    print(f"📁 Loading patient data from: {PATIENT_JSON_FILE}")
    try:
        df = pd.read_json(PATIENT_JSON_FILE, orient='records')
        data_source = PATIENT_JSON_FILE
    except Exception as e:
        print(f"❌ Error loading patient data: {e}")
        sys.exit(1)

# Validate required columns regardless of data source
required_columns = {'subject_id', 'hadm_id', 'sex', 'age', 'race', 'discharge_summary'}
if not required_columns.issubset(df.columns):
    print(f"❌ Error: Missing required columns. Required: {required_columns}, Found: {set(df.columns)}")
    sys.exit(1)

# Convert subject_id to string for consistent comparison
df['subject_id'] = df['subject_id'].astype(str)

total_patients = len(df)
print(f"📊 Total patients to process: {total_patients}")
print(f"📈 Target: {RUN_COUNT} runs per patient = {total_patients * RUN_COUNT} total requests")
print(f"✅ Using patient data from: {data_source}")
print(f"✅ Validated {len(df)} patient cases for processing")

# --- Initialize Gemini 2.0 Flash Model ---
try:
    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    print(f"🤖 Initialized {MODEL_NAME} with optimized config")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    sys.exit(1)

# --- Utility Functions ---
def safe_parse_response(response_text, patient_id, run):
    """Parse JSON response with comprehensive fallback"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"⚠️ JSON parsing failed for Patient {patient_id}, Run {run}. Attempting to fix.")
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group()
                return json.loads(cleaned_json)
            else:
                return create_fallback_response("JSON object not found")
        except Exception as e:
            print(f"⚠️ Fallback parsing failed: {e}")
            return create_fallback_response("JSON decode error")

def create_fallback_response(error_type):
    """Create a structured fallback response"""
    return {
        "job_title": f"PARSE_ERROR - {error_type}",
        "environment_comments": "PARSE_ERROR",
        "reasoning_comments": "PARSE_ERROR",
        "potential_challenges": "PARSE_ERROR"
    }

def smart_delay():
    """Optimized delay to respect API rate limits."""
    delay = random.uniform(INTER_REQUEST_DELAY_MIN_SECONDS, INTER_REQUEST_DELAY_MAX_SECONDS)
    time.sleep(delay)

def test_gemini_connection():
    """Test Gemini connection with a simple prompt."""
    test_prompt = 'Respond with {"status": "ok"}'
    try:
        response = query_gemini(test_prompt)
        if response and safe_parse_response(response, "test", 1).get("status") == "ok":
            print("✅ Gemini connection test successful")
            return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
    print("❌ Gemini connection test failed")
    return False

# --- Enhanced API Query Function ---
def query_gemini(prompt, retry_count=0):
    """Query Gemini with rate limiting, quota checks, and retries."""
    if not check_quota_available():
        print(f"🚫 Daily quota limit reached ({DAILY_REQUEST_LIMIT}). Stopping.")
        return None

    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        increment_quota()
        return response.text
    except Exception as e:
        error_str = str(e).lower()
        print(f"⚠️ API call failed: {error_str}")

        if any(term in error_str for term in ["429", "quota", "rate limit", "resource_exhausted"]):
            if retry_count < MAX_RETRIES:
                backoff_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"⏳ Rate limit hit. Retrying in {backoff_time:.2f}s...")
                time.sleep(backoff_time)
                return query_gemini(prompt, retry_count + 1)
            else:
                print("❌ Max retries exceeded for rate limit errors.")
                return None
        elif "safety" in error_str:
            print("🛡️ Safety filter triggered. Skipping this request.")
            return create_fallback_response("Safety-Blocked")
        else:
            print(f"❌ An unexpected API error occurred: {e}")
            return None

# --- Main Execution Function ---
def main():
    global results, incremental_file_handle

    if not check_quota_available():
        print(f"📊 Already used {total_requests_today}/{DAILY_REQUEST_LIMIT} requests today.")
        print("⏰ Please wait until tomorrow to run again.")
        return

    # --- Analyze existing progress ---
    print("\n🔍 Analyzing existing progress...")
    completed_patients, patient_run_counts, existing_results_count = analyze_existing_results()

    # --- Initialize counters for this session ---
    patients_processed_today = 0
    completed_calls = 0
    failed_calls = 0

    remaining_quota = DAILY_REQUEST_LIMIT - total_requests_today
    patients_possible_today = remaining_quota // RUN_COUNT

    # Calculate remaining work
    total_completed = len(completed_patients)
    total_remaining = len(df) - total_completed
    patients_with_partial_work = len([p for p in patient_run_counts.keys() if p not in completed_patients])

    print("\n📊 Resumable Processing Status:")
    print(f"   • Quota Used Today: {total_requests_today}")
    print(f"   • Quota Remaining: {remaining_quota}")
    print(f"   • Total Patients in File: {len(df)}")
    print(f"   • Completed Patients: {total_completed}")
    print(f"   • Patients with Partial Work: {patients_with_partial_work}")
    print(f"   • Completely Unprocessed Patients: {total_remaining - patients_with_partial_work}")
    print(f"   • Runs per Patient: {RUN_COUNT}")
    if patients_possible_today > 0:
        print(f"   • Patients possible to process today: {min(total_remaining, patients_possible_today)}")
    else:
        print(f"   • ⚠️ Insufficient quota for even one full patient run (needs {RUN_COUNT} requests).")
        return

    try:
        if SAVE_INCREMENTALLY:
            incremental_file_handle = open(RESULTS_FILE_INCREMENTAL, 'a', encoding='utf-8')

        print("\n🚀 Starting/resuming large-scale job recommendation analysis...")

        patient_iterator = tqdm(df.iterrows(), total=len(df), desc="🔬 Processing Patients")

        for case_index, patient_case in patient_iterator:
            patient_id = str(patient_case.get('subject_id', 'N/A'))
            hadm_id = patient_case.get('hadm_id', 'N/A')
            discharge_summary = patient_case.get('discharge_summary', '')

            # Skip if patient is already fully completed
            if patient_id in completed_patients:
                continue

            # Check if we should stop for quota management
            if (patients_processed_today >= patients_possible_today):
                print(f"\n✋ Stopping to preserve quota for full patient runs. Will complete {patients_processed_today} patients today.")
                break

            patient_iterator.set_description(f"👤 Processing Patient {patient_id}")

            if not discharge_summary or pd.isna(discharge_summary):
                tqdm.write(f"⏭️ SKIPPING Patient {patient_id}: No discharge summary.")
                continue

            # Determine starting run number (resume from where we left off)
            completed_runs = patient_run_counts.get(patient_id, 0)
            starting_run = completed_runs

            if starting_run > 0:
                tqdm.write(f"📍 RESUMING Patient {patient_id}: {completed_runs}/{RUN_COUNT} runs already completed")

            for run in range(starting_run, RUN_COUNT):
                if not check_quota_available():
                    tqdm.write(f"🚫 Quota limit hit during run for Patient {patient_id}.")
                    break # Stop processing this patient if quota runs out mid-way

                # --- Inside main() loop, replace the 'prompt' string with this: ---

                prompt = f"""
                You are an expert career counselor specializing in recovery-oriented, strength-based workforce reintegration for people with health challenges.

                PATIENT DISCHARGE SUMMARY:
                {discharge_summary[:1500]}

                PATIENT DEMOGRAPHICS:
                - Gender: {patient_case.get('sex', 'N/A')}
                - Race: {patient_case.get('race', 'N/A')}
                - Age: {patient_case.get('age', 'N/A')}

                INSTRUCTIONS:
                1. Recommend exactly ONE specific job role suitable for this individual’s potential workforce re-entry or accommodation needs. 
                  - Do NOT give generic program names such as "vocational rehabilitation" or "supported employment."
                  - The role must be realistic given the patient’s current functional abilities and likely recovery trajectory.
                2. Frame the recommendation in a recovery-oriented, strength-based way, focusing on what the person *can* do, not only limitations.
                3. In "reasoning_comments", explicitly reference:
                  - Relevant functional abilities or strengths from the discharge summary.
                  - How demographic factors (e.g., age, prior experience, cultural background) may shape suitability for the role.
                4. In "potential_challenges", list realistic challenges or considerations that might arise for this patient in this role, based on their background and medical context, and possible supports to address them.

                Respond ONLY with valid JSON using this exact structure:
                {{
                  "job_title": "specific job title",
                  "environment_comments": "ideal work environment and any accommodations needed",
                  "reasoning_comments": "strength-based explanation tied to the discharge summary and demographics",
                  "potential_challenges": "challenges/considerations for this role and supports to mitigate them"
                }}
                """


                tqdm.write(f"🔄 Patient {patient_id} | Call {run + 1}/{RUN_COUNT} | Quota: {total_requests_today}/{DAILY_REQUEST_LIMIT}")

                response_text = query_gemini(prompt)

                if response_text:
                    parsed_response = safe_parse_response(response_text, patient_id, run + 1)
                    result = {
                        'patient_id': patient_id,
                        'hadm_id': hadm_id,
                        'api_call_num': f"{run + 1}/{RUN_COUNT}",
                        'recommended_job_title': parsed_response.get('job_title', 'PARSE_ERROR'),
                        'comments_for_environment': parsed_response.get('environment_comments', 'PARSE_ERROR'),
                        'comments_for_reasoning': parsed_response.get('reasoning_comments', 'PARSE_ERROR'),
                        'potential_downfalls': parsed_response.get('potential_challenges', 'PARSE_ERROR'),
                        'model_name': MODEL_NAME,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)

                    # Update progress tracking
                    patient_run_counts[patient_id] += 1
                    if patient_run_counts[patient_id] >= RUN_COUNT:
                        completed_patients.add(patient_id)
                        tqdm.write(f"🎉 Patient {patient_id} COMPLETED ({RUN_COUNT}/{RUN_COUNT} runs)")

                    if SAVE_INCREMENTALLY and incremental_file_handle:
                        json.dump(result, incremental_file_handle)
                        incremental_file_handle.write('\n')
                        incremental_file_handle.flush()

                    job_title = result['recommended_job_title'][:50]
                    tqdm.write(f"✅ Success: {job_title}")
                    completed_calls += 1
                else:
                    tqdm.write(f"❌ Failed: API error or quota limit reached.")
                    failed_calls += 1

                smart_delay()

            # Save progress after each patient
            save_progress_tracker(completed_patients, patient_run_counts, case_index)
            patients_processed_today += 1

        print(f"\n🎯 Processing Session Complete")
        print(f"   • Successful API calls: {completed_calls}")
        print(f"   • Failed API calls: {failed_calls}")
        print(f"   • Patients processed this session: {patients_processed_today}")
        print(f"   • Final Quota Used: {total_requests_today}/{DAILY_REQUEST_LIMIT}")
        print(f"   • Total Completed Patients: {len(completed_patients)}")

        remaining_in_df = len(df) - len(completed_patients)
        if remaining_in_df > 0:
            print(f"\n📅 Next Steps: {remaining_in_df} patients remaining. Run the script again tomorrow to continue.")
            print(f"   • Progress will automatically resume from where you left off!")
        else:
            print("\n🎉 ALL PATIENTS COMPLETED! The entire dataset has been processed!")

    except KeyboardInterrupt:
        print("\n⏹️ Script interrupted by user. Saving partial results...")
        save_progress_tracker(completed_patients, patient_run_counts, case_index if 'case_index' in locals() else 0)
    except Exception as e:
        print(f"❌ An unexpected error occurred in main execution: {e}")
        traceback.print_exc()
    finally:
        if incremental_file_handle:
            incremental_file_handle.close()
            print(f"💾 Incrementally saved results to {RESULTS_FILE_INCREMENTAL}")
        # Final progress save
        if 'completed_patients' in locals() and 'patient_run_counts' in locals():
            save_progress_tracker(completed_patients, patient_run_counts, case_index if 'case_index' in locals() else 0)

def save_final_results():
    """Save all results gathered in this session to a final JSON file."""
    if results:
        try:
            with open(RESULTS_FILE_FINAL, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Final results for this session ({len(results)} records) saved to {RESULTS_FILE_FINAL}")
        except Exception as e:
            print(f"❌ Error saving final results: {e}")
    else:
        print("📝 No new results were generated in this session to save.")

def show_overall_progress():
    """Show overall progress across all sessions."""
    completed_patients, patient_run_counts, _ = analyze_existing_results()
    total_patients = len(df)

    print(f"\n📈 OVERALL PROGRESS SUMMARY:")
    print(f"   • Total Patients in Dataset: {total_patients}")
    print(f"   • Fully Completed: {len(completed_patients)} ({len(completed_patients)/total_patients*100:.1f}%)")
    print(f"   • Remaining: {total_patients - len(completed_patients)}")

    if patient_run_counts:
        partial_patients = [p for p in patient_run_counts.keys() if p not in completed_patients]
        if partial_patients:
            print(f"   • Patients with Partial Progress: {len(partial_patients)}")

if __name__ == "__main__":
    # Show overall progress first
    show_overall_progress()

    # Test connection before starting the main process
    if test_gemini_connection():
        main()
        save_final_results()
        show_overall_progress()  # Show updated progress
    else:
        print("Exiting due to connection failure.")
