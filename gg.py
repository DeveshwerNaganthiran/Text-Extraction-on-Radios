import csv

display_styles = [34, 38, 39, 40, 41, 50, 51, 52, 53, 54, 55]
valid_string_tags = []

# Added 'r' before the path to fix the unicode escape error
# Make sure this points to your CSV file, not the Excel file
file_path = r'C:\Users\amb3942\Desktop\walkie-tracker\Batch_Summary_Reportv2.xlsx - Batch Summary.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        remark = row.get('Remark by Adib', '')
        # Filter out anything with the word "truncated"
        if 'truncated' not in remark.lower() and row.get('Index', '').isdigit():
            valid_string_tags.append(row['Index'])

# Output file path
output_path = r'C:\Users\amb3942\Desktop\Untitled spreadsheet - Sheet1.csv'

with open(output_path, 'w', encoding='utf-8') as f:
    for i, st_index in enumerate(valid_string_tags):
        # 1:1 mapping looping through styles
        ds_index = display_styles[i % len(display_styles)]
        
        # Zero-pad indices to 4 digits
        ds_idx_padded = str(ds_index).zfill(4)
        st_idx_padded = str(st_index).zfill(4)
        
        # Write ONLY the requested command format
        f.write(f"STR_TEST:FIX:{ds_idx_padded}:{st_idx_padded}\n")

print(f"Done! {len(valid_string_tags)} commands saved to commands.txt")