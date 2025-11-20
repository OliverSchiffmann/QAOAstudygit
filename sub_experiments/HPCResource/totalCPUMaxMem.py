import pandas as pd
import re
import numpy as np

# Define the input filename based on the previous step
inputFile = "qaoaStudyResults.csv"
# Define the memory unit conversion factor (KB to GB)
KILOBYTES_TO_GIGABYTES = 1024 * 1024


def convertSlurmTimeToSeconds(timeStr: str) -> int:
    """
    Converts a SLURM time string (e.g., 'D-HH:MM:SS' or 'HH:MM:SS') to total seconds.
    """
    if pd.isna(timeStr) or timeStr == "None":
        return 0

    # Handle day format (D-HH:MM:SS)
    if "-" in timeStr:
        dayPart, timePart = timeStr.split("-", 1)
        days = int(dayPart)
    else:
        days = 0
        timePart = timeStr

    # Handle time parts (HH:MM:SS)
    parts = timePart.split(":")

    hours, minutes, seconds = 0, 0, 0

    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        # Assumes format is MM:SS or HH:MM if only two parts are present (common for short times)
        minutes, seconds = map(int, parts)
        hours = 0
    else:
        # Fallback for unexpected format
        return 0

    totalSeconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return totalSeconds


def getMaxMemoryInGB(memoryStr: str) -> float:
    """
    Converts a SLURM memory string (e.g., '4743112K', '4743112', or '0') to Gigabytes.
    """
    if pd.isna(memoryStr) or memoryStr == "None":
        return 0.0

    # Clean the string, removing trailing 'K' or 'M' if present (case-insensitive)
    cleanedStr = str(memoryStr).strip().upper()

    if "K" in cleanedStr:
        cleanedStr = cleanedStr.replace("K", "")
        unit_multiplier = 1
    elif "M" in cleanedStr:
        cleanedStr = cleanedStr.replace("M", "")
        unit_multiplier = 1024  # Convert MB to KB
    else:
        # Assume it's already in Kilobytes (K) or is 0 if no unit is specified
        unit_multiplier = 1

    # Use regex to find the numeric part
    match = re.match(r"(\d+)", cleanedStr)
    if match:
        kilobytes = int(match.group(1)) * unit_multiplier
        # Convert total KB to GB
        return kilobytes / KILOBYTES_TO_GIGABYTES
    return 0.0


def analyzeSlurmData(fileName: str):
    """
    Main function to load, process, and analyze the SLURM job data.
    """
    try:
        # Read the file using whitespace ('\s+') as the separator
        # skiprows=[1]: Skips the header-separator line (------------)
        jobData = pd.read_csv(
            fileName, sep=r"\s+", skiprows=[1], engine="python", header=0
        )
    except FileNotFoundError:
        print(f"Error: The file '{fileName}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- FIX: Clean up column names by stripping whitespace ---
    jobData.columns = jobData.columns.str.strip()

    # --- Renaming Columns for Consistency ---
    if "JobID" in jobData.columns:
        jobData.rename(columns={"JobID": "JobId"}, inplace=True)

    # Check if critical columns exist after potential renaming
    requiredColumns = ["JobId", "JobName", "MaxRSS", "CPUTime"]
    for col in requiredColumns:
        if col not in jobData.columns:
            print(
                f"Error: Critical column '{col}' not found after cleaning. Available columns: {jobData.columns.tolist()}"
            )
            return

    # --- Print Header and Last 10 Rows ---
    print(f"\n--- Last 10 rows of Cleaned Data (Total {len(jobData)} rows) ---")
    print(jobData.tail(10))

    # --- Data Filtering and Cleaning ---

    # 1. Filter for 'batch' steps: MaxRSS (Peak Memory) is most reliable here.
    # The JobName column is the most reliable filter.
    jobData["JobName"] = jobData["JobName"].astype(str).str.strip()
    batchSteps = jobData[jobData["JobName"] == "batch"].copy()

    # Convert MaxRSS to Gigabytes.
    batchSteps["MaxRSS_GB"] = batchSteps["MaxRSS"].apply(getMaxMemoryInGB)

    # Convert CPUTime to seconds for summation
    batchSteps["CPUTime_S"] = batchSteps["CPUTime"].apply(convertSlurmTimeToSeconds)

    # --- Calculation ---

    # 1. Calculate Total CPU Time (Sum of all CPUTime_S)
    totalCpuTimeSeconds = batchSteps["CPUTime_S"].sum()
    totalCpuTimeHours = totalCpuTimeSeconds / 3600

    # 2. Calculate Maximum Memory Needed (Max of all MaxRSS_GB)
    # Use np.nanmax to safely get the max, ignoring NaN if the batchSteps DataFrame is empty
    if not batchSteps["MaxRSS_GB"].empty:
        maxMemoryNeededGB = batchSteps["MaxRSS_GB"].max()
    else:
        maxMemoryNeededGB = np.nan

    # --- Output ---

    print("-" * 50)
    print("Classical HPC Resource Consumption Summary")
    print("-" * 50)
    print(f"Total Number of QAOA Runs Analyzed (Batch Steps): {len(batchSteps)}")
    print(f"Total Classical CPU Time Used: {totalCpuTimeSeconds:,.0f} seconds")
    print(f"Total Classical CPU Time Used: {totalCpuTimeHours:,.2f} hours")
    print(
        f"Maximum Peak Memory (RAM) Required by Any Single Job: {maxMemoryNeededGB:,.2f} GB"
    )
    print("-" * 50)


if __name__ == "__main__":
    analyzeSlurmData(inputFile)
