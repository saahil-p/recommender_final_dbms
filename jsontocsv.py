import json
import csv

# Specify the fields that you want to start from
fields_to_extract = [
    "job_title", "category", "company_name", "city", "state", "country", 
    "job_description", "job_type", "is_remote"
]

# Open the LDJSON file and process each line as a separate JSON object
with open('dataset.ldjson', 'r') as ldjson_file:
    json_lines = []
    for line in ldjson_file:
        try:
            # Load each line as a JSON object
            json_obj = json.loads(line.strip())
            # Extract only the fields starting from "job_title"
            filtered_obj = {key: json_obj.get(key, '') for key in fields_to_extract}
            json_lines.append(filtered_obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on this line: {line}")
            print(f"Error details: {e}")
            continue  # Skip problematic lines and continue processing

# Open or create CSV file to write to
with open('job_data_filtered.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write headers (fields to extract)
    csv_writer.writerow(fields_to_extract)

    # Write rows (values corresponding to the extracted fields)
    for json_obj in json_lines:
        csv_writer.writerow([', '.join(v) if isinstance(v, list) else v for v in json_obj.values()])
