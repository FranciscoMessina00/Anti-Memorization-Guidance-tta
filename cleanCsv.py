import csv

# Input CSV file with errors and the output CSV for the cleaned data
input_csv = "output.csv"
output_csv = "cleaned_output.csv"

def is_valid_description(description):
    """
    Returns True if the description does not start with error messages.
    """
    return not (description.startswith("Error:") or description.startswith("Exception:"))

# Open the input CSV file for reading and output CSV file for writing
with open(input_csv, mode="r", newline="", encoding="utf-8-sig") as infile, \
     open(output_csv, mode="w", newline="", encoding="utf-8-sig") as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()  # Write column headers
    
    # Process each row and filter based on the "description" field
    for row in reader:
        if is_valid_description(row["description"]):
            writer.writerow(row)
        # else:
        #     print(f"Skipping row with id {row.get('id')} due to error: {row['description']}")

print(f"Cleaned CSV saved to {output_csv}")
