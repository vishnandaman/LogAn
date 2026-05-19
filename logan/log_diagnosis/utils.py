import base64
import html
import json
import re
import os
from argparse import ArgumentParser
import shutil
import sys

from jinja2 import Environment, FileSystemLoader
import pandas as pd

from logan.telemetry.es import get_elasticsearch_config, get_feedback_index

def get_b64_encoded_credentials(username, passwd):
    credentials = f'{username}:{passwd}'
    return base64.b64encode(credentials.encode()).decode('ascii')

def row_size(row):
    """
    Calculates the memory size of a given row.
    
    Args:
        row (pd.Series): A row of the DataFrame.
    
    Returns:
        int: Size of the row in bytes.
    """
    return sys.getsizeof(row)

def replace_tags(text):
    """
    Replace specific HTML tags with &lt; and &gt; and escape HTML entities in the given text.
    Args:
        text (str): The input string containing HTML tags and entities.
    Returns:
        str: The cleaned string with specific HTML tags replaced and entities escaped.
    """
    try:
        # Replace specific tags
        text = re.sub(r'</?script>', '', text, flags=re.IGNORECASE)
        # Replace multiple consecutive <br> tags with a single <br>
        text = re.sub(r'(<br\s*/?>\s*)+', '<br>', text)
        
        # Replace all other HTML tags with &lt; and &gt;, except <br> and <br/> tags
        parts = re.split(r'(<[^>]+>)', text)
        for i in range(len(parts)):
            if parts[i].lower() not in ('<br>', '<br/>', '</br>'):
                parts[i] = html.escape(parts[i],quote=False)
        text = ''.join(parts)
    except TypeError:
        print(f"Error in replacing HTML tags for text: {text}")
    
    return text

def split_df_on_size(df, threshold):
    """
    Splits a DataFrame into chunks such that each chunk's total memory size is below the specified threshold.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        threshold (int): The maximum size (in bytes) for each chunk.
    
    Returns:
        list: A list of DataFrames, each representing a chunk within the size limit.
    """
    df['row_size_bytes'] = df.apply(row_size, axis=1)  # Calculate the size of each row in bytes.

    current_size = 0  # Tracks the size of the current chunk.
    chunks = []  # Holds all the chunks created from splitting.
    chunk = []  # Holds the current chunk of rows.

    # Iterate through each row in the DataFrame.
    for index, row in df.iterrows():
        row_bytes = row["row_size_bytes"]
        # If adding the current row doesn't exceed the threshold, add it to the chunk.
        if current_size + row_bytes <= threshold:
            chunk.append(row)
            current_size += row_bytes
        else:
            # If the current chunk exceeds the threshold, store it and start a new chunk.
            if len(chunk) > 0: 
                chunks.append(pd.DataFrame(chunk))
            chunk = [row]  # Start a new chunk with the current row.
            current_size = row_bytes

    # Add the last chunk if it exists.
    if len(chunk) > 0:  
        chunks.append(pd.DataFrame(chunk))

    return chunks

def create_feedback_variable():
    """
    Creates a configuration dictionary for Elasticsearch feedback, including credentials and index information.
    
    Returns:
        dict: Configuration dictionary with Elasticsearch credentials and feedback index.
    """
    es_config = get_elasticsearch_config()  # Load Elasticsearch configuration.
    es_config['feedback_index'] = get_feedback_index()  # Set the feedback index.
    # Encode the username and password as base64 and store as a token.
    es_config['token'] = get_b64_encoded_credentials(es_config['username'], es_config['password'])  
    return es_config

def get_anomaly_html_str(df_final_anomalies, output_dir):
    """
    Generates an HTML string for anomaly data and saves the JSON representation of the data in chunks.
    
    Args:
        df_final_anomalies (pd.DataFrame): DataFrame containing anomaly information.
        output_dir (str): Directory path where output files should be saved.
    
    Returns:
        str: Rendered HTML string for the anomalies.
    """
    
    # If no anomalies are present, notify the user.
    if len(df_final_anomalies) == 0:
        print("No anomalies detected") 
    
    # Remove HTML tags from log entries.
    tag_pattern = re.compile(r'<[^>]*>')
    df_final_anomalies['list_logs'] = df_final_anomalies['list_logs'].apply(lambda log: re.sub(tag_pattern, '', log))
    
    # Split the DataFrame into smaller chunks, each under 2.5 MB.
    list_chunked_df = split_df_on_size(df_final_anomalies, threshold=2.5*1024*1024)  # 2.5 MB splits
    
    chunk_size = []  # To track the size of each chunk.
    output_prefix = f'{output_dir}/developer_debug_files/data'  # Prefix for the output JSON files.
    
    # Iterate through each chunk and save it as a JSON file.
    for idx, df_anomaly in enumerate(list_chunked_df):
        chunk_size.append(len(df_anomaly))  # Record the size of the current chunk.
        output_json_obj = []  # Prepare a list to store JSON objects.
        
        # Convert each row of the DataFrame into a JSON object.
        for _, row in df_anomaly.iterrows():
            temp_json_obj = {}
            start, end, each_window, file_window, templates = row["start_ts"], row["end_ts"], row["list_logs"], row["list_files"], row["list_templates"]

            duration = f"{start} -- \n {end}"
            logs = each_window.split('\n')  # Split logs by newline.
            files = file_window.split('\n')  # Split file names by newline.

            # Extract the last element in logs that represents the golden signal (gs).
            list_of_gs = [item.split('=>')[-1].split()[-1].strip() for item in logs]
            list_of_templates = templates.split(" ")

            temp_json_obj['duration'] = duration
            temp_json_obj['logs'] = logs
            temp_json_obj['files'] = files
            temp_json_obj['gs'] = [gs.strip() for gs in list_of_gs]
            temp_json_obj['templateIds'] = list_of_templates
            output_json_obj.append(temp_json_obj)
        
        # Save the JSON object to a file.
        output_file = f"{output_prefix}_{idx+1}.json"
        
        with open(output_file, 'w') as output_json_file:
            json.dump(output_json_obj, output_json_file, indent=4)
            
        print(f"Written chunk {idx+1} to {output_file}")
    
    # Create a Jinja environment to render the HTML template.
    path = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))
    html_template = env.get_template('anomalies.html')

    # Render the HTML template using the provided data.
    rendered_template = html_template.render(
        chunk_size=chunk_size, 
        no_of_chunks=len(list_chunked_df), 
        no_of_windows=sum(chunk_size), 
        zip=zip, set=set, 
        output_dir='../developer_debug_files', 
        min=min
    )
    
    return rendered_template

def _format_file_size_bytes(size_bytes):
    """Format byte count as human-readable string (e.g. 4.0 MB)."""
    if size_bytes is None or size_bytes < 0:
        return None
    for unit, suffix in [(1e9, 'GB'), (1e6, 'MB'), (1e3, 'KB')]:
        if size_bytes >= unit:
            return f'{size_bytes / unit:.1f} {suffix}'
    return f'{size_bytes} B'


def _load_preprocessing_metrics(output_dir):
    """Load num_log_lines_total and file_size_bytes from output_dir/metrics/preprocessing.json if present."""
    if not output_dir:
        return None, None, None
    path = os.path.join(output_dir, 'metrics', 'preprocessing.json')
    if not os.path.isfile(path):
        return None, None, None
    try:
        with open(path) as f:
            data = json.load(f)
        num_lines = data.get('num_log_lines_total')
        file_size = data.get('file_size_bytes')
        file_size_display = _format_file_size_bytes(file_size) if file_size is not None else None
        return num_lines, file_size, file_size_display
    except (json.JSONDecodeError, OSError):
        return None, None, None


def get_summary_html_str(df_for_summary_html, include_golden_signal_dropdown, ignored_file_list, processed_file_list, output_dir=None, has_timeline_data=False):
    """
    Generates an HTML string for the summary report, including details about golden signals and processed files.

    When output_dir is provided, loads preprocessing metrics (total log lines, file size) from
    output_dir/metrics/preprocessing.json and passes them to the template.

    Args:
        df_for_summary_html (pd.DataFrame): DataFrame containing summary data.
        include_golden_signal_dropdown (bool): Whether to include a dropdown for golden signals in the report.
        ignored_file_list (list): List of files that were ignored during processing.
        processed_file_list (list): List of files that were successfully processed.
        output_dir (str, optional): Output directory; if set, preprocessing metrics are loaded and shown.

    Returns:
        str: Rendered HTML string for the summary report.
    """
    # Round coverage values to four decimal places.
    df_for_summary_html = df_for_summary_html[['d_tid', 'text', 'gs', 'd_tid_count', 'coverage', 'file_names']]
    df_for_summary_html['coverage'] = df_for_summary_html['coverage'].apply(lambda val: round(val, 4))

    df_for_summary_html['text'] = df_for_summary_html['text'].apply(replace_tags)

    num_log_lines_total, file_size_bytes, file_size_display = _load_preprocessing_metrics(output_dir)
    num_log_lines_display = f'{num_log_lines_total:,}' if num_log_lines_total is not None else None

    # Create a Jinja environment to render the summary HTML template.
    path = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))
    html_template = env.get_template('summary_golden_signal_error.html')

    # Convert to native Python types so Jinja2 renders valid JS literals (not np.str_)
    import numpy as np
    def _to_native(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        return v
    summary_table = [[_to_native(v) for v in row] for row in df_for_summary_html.values.tolist()]

    # Render the summary HTML template.
    rendered_template = html_template.render(
        summary_table=summary_table,
        include_golden_signal_dropdown=include_golden_signal_dropdown,
        ignored_file_list=ignored_file_list,
        processed_file_list=processed_file_list,
        unique_file_names=sorted(df_for_summary_html['file_names'].unique().tolist()),
        num_log_lines_total=num_log_lines_total,
        num_log_lines_display=num_log_lines_display,
        file_size_bytes=file_size_bytes,
        file_size_display=file_size_display,
        has_timeline_data=has_timeline_data,
    )

    return rendered_template

def compute_golden_signal_timeline(df_for_anomaly_html, output_dir):
    """
    Compute time-binned golden signal counts from the anomaly DataFrame and write
    the result to output_dir/log_diagnosis/golden_signal_timeline.json.

    Returns True if non-empty bins were written, False otherwise.
    """
    import math
    from datetime import datetime, timezone

    output_path = os.path.join(output_dir, "developer_debug_files", "golden_signal_timeline.json")
    empty_result = {"bin_seconds": 0, "bin_label": "", "signals": [], "bins": []}

    if df_for_anomaly_html is None or df_for_anomaly_html.empty:
        with open(output_path, "w") as f:
            json.dump(empty_result, f)
        return False

    epochs = df_for_anomaly_html["epoch"].dropna().values
    if len(epochs) == 0:
        with open(output_path, "w") as f:
            json.dump(empty_result, f)
        return False

    min_epoch = float(epochs.min())
    max_epoch = float(epochs.max())
    time_span = max_epoch - min_epoch

    CANDIDATE_INTERVALS = [30, 60, 300, 900, 1800, 3600, 21600, 86400]
    INTERVAL_LABELS = {
        30: "30 sec", 60: "1 min", 300: "5 min", 900: "15 min",
        1800: "30 min", 3600: "1 hour", 21600: "6 hours", 86400: "1 day",
    }
    TARGET_MAX_BINS = 100

    bin_seconds = CANDIDATE_INTERVALS[-1]
    for candidate in CANDIDATE_INTERVALS:
        num_bins = max(1, math.ceil(time_span / candidate)) if time_span > 0 else 1
        if num_bins <= TARGET_MAX_BINS:
            bin_seconds = candidate
            break
    else:
        bin_seconds = max(3600, math.ceil(time_span / TARGET_MAX_BINS / 3600) * 3600)

    bin_label = INTERVAL_LABELS.get(bin_seconds, f"{bin_seconds // 3600} hours")

    # Discover all signals from the data
    all_signals = set()
    bin_counts = {}
    for _, row in df_for_anomaly_html.iterrows():
        epoch_val = row.get("epoch")
        gs_str = row.get("golden_signal", "")
        if pd.isna(epoch_val) or not gs_str:
            continue
        bin_idx = int((float(epoch_val) - min_epoch) // bin_seconds)
        signals_in_row = gs_str.strip().split()
        if bin_idx not in bin_counts:
            bin_counts[bin_idx] = {}
        for sig in signals_in_row:
            sig_lower = sig.lower().strip()
            if sig_lower:
                all_signals.add(sig_lower)
                bin_counts[bin_idx][sig_lower] = bin_counts[bin_idx].get(sig_lower, 0) + 1

    if not bin_counts:
        with open(output_path, "w") as f:
            json.dump(empty_result, f)
        return False

    # Stable ordering: known signals first (in a sensible order), then extras alphabetically
    KNOWN_ORDER = ["error", "latency", "saturation", "traffic", "availability", "information"]
    signals = [s for s in KNOWN_ORDER if s in all_signals]
    signals += sorted(all_signals - set(KNOWN_ORDER))

    max_bin_idx = max(bin_counts.keys())
    bins = []
    for i in range(max_bin_idx + 1):
        bin_start = min_epoch + i * bin_seconds
        dt = datetime.fromtimestamp(bin_start, tz=timezone.utc)
        if bin_seconds >= 86400:
            label = dt.strftime("%Y-%m-%d")
        elif bin_seconds >= 3600:
            label = dt.strftime("%Y-%m-%d %H:%M")
        else:
            label = dt.strftime("%m-%d %H:%M")
        counts = bin_counts.get(i, {})
        bin_entry = {"start": bin_start, "label": label}
        for s in signals:
            bin_entry[s] = counts.get(s, 0)
        bins.append(bin_entry)

    result = {
        "bin_seconds": bin_seconds,
        "bin_label": bin_label,
        "signals": signals,
        "bins": bins,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return True


def get_explorer_html_str(store_meta: dict) -> str:
    """
    Render explorer.html with the store metadata embedded so the page knows
    which JSON files to fetch for templates and log entries.

    Args:
        store_meta: dict returned by LogStore.save_json_for_explorer()

    Returns:
        Rendered HTML string.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))
    html_template = env.get_template('explorer.html')
    return html_template.render(
        total_templates=store_meta.get("total_templates", 0),
        total_entries=store_meta.get("total_entries", 0),
        templates_file=store_meta.get("templates_file", ""),
        parquet_entries=store_meta.get("parquet_entries", ""),
        parquet_templates=store_meta.get("parquet_templates", ""),
    )


def prepare_output_dir(output_dir: str, clean_up: bool = False):
    """
    Prepares the output directory by creating necessary subdirectories.
    
    Args:
        output_dir (str): The directory path where output will be stored.
    """
    # Clean up output directory if it exists.
    if clean_up and os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    # Create necessary subdirectories.
    FOLDERS = ['run', 'pandarallel_cache', 'test_templates', 'log_diagnosis', 'developer_debug_files', 'metrics', 'store']
    os.makedirs(output_dir, exist_ok=True)
    for folder in FOLDERS:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Copy HTML templates libs to the output directory.
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'templates', 'libs'), os.path.join(output_dir, 'log_diagnosis', 'libs'), dirs_exist_ok=True)

    # Download (once) and copy DuckDB WASM assets for the log explorer.
    from logan.store.duckdb_assets import copy_duckdb_to_output
    copy_duckdb_to_output(os.path.join(output_dir, 'log_diagnosis', 'libs'))

    # Setup Env Variables
    os.environ['MEMORY_FS_ROOT'] = os.path.join(output_dir, 'pandarallel_cache')

if __name__ == '__main__':
    """
    Main entry point for the script. Parses command-line arguments and generates either an anomaly or summary report.
    """
    parser = ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    debug_dir = os.path.join(os.path.dirname(args.output_dir), 'developer_debug_files')

    # Generate the anomaly report and save it.
    df_final_anomalies = pd.DataFrame(columns=['start_ts', 'end_ts', 'list_logs', 'list_files', 'list_templates'])    
    with open(os.path.join(args.output_dir, 'anomalies.html'), 'w') as writer:
        writer.write(get_anomaly_html_str(df_final_anomalies, output_dir=debug_dir))
    
    # Generate the summary report and save it.
    df_for_summmary_html = pd.DataFrame(columns=['d_tid', 'text', 'gs', 'd_tid_count', 'coverage', 'file_names'])
    with open(os.path.join(args.output_dir, 'summary.html'), 'w') as writer:
        writer.write(get_summary_html_str(df_for_summmary_html, include_golden_signal_dropdown=True, ignored_file_list=[], processed_file_list=[]))
