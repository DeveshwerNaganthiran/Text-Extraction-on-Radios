import pandas as pd
import os
import re

# 1. Your original input file with all 1102 commands
MASTER_BATCH_FILE = r"C:\Users\amb3942\Desktop\Untitled spreadsheet - Sheet1.csv"

# 2. The output Excel file that contains the Passes/Fails/Warns
OUTPUT_EXCEL_FILE = r"C:\Users\amb3942\Desktop\walkie-tracker\output\batch_run_20260715_093100\HAFIZ DRY RUN_NewRun_20260713_163239.xlsx"

def normalize(s):
    # Nuke all spaces, quotes, brackets, and make lowercase for bulletproof matching
    return re.sub(r'[\s\'"\[\]]+', '', str(s)).lower()

def recover_all_languages():
    if not os.path.exists(MASTER_BATCH_FILE) or not os.path.exists(OUTPUT_EXCEL_FILE):
        print("❌ ERROR: Cannot find the files. Check your paths!")
        return

    # 1. Load the Master CSV
    print("1. Reading Master CSV file...")
    if MASTER_BATCH_FILE.lower().endswith('.csv'):
        df_master = pd.read_csv(MASTER_BATCH_FILE, header=None)
    else:
        df_master = pd.read_excel(MASTER_BATCH_FILE, header=None, engine="openpyxl")
    
    master_cmds = df_master.iloc[:, 0].dropna().astype(str).tolist()
    print(f"   -> Total commands in Master file: {len(master_cmds)}\n")

    # 2. Load the Output Excel
    print("2. Reading Output Excel file...")
    df_out = pd.read_excel(OUTPUT_EXCEL_FILE, engine="openpyxl")
    
    if 'Command' not in df_out.columns:
        print("❌ ERROR: Could not find 'Command' column in the output Excel.")
        return

    lang_col = next((col for col in df_out.columns if 'lang' in col.lower() or 'locale' in col.lower()), None)
    
    if not lang_col:
        print("❌ ERROR: Could not find a Language column!")
        return

    # 3. Process EACH language with Substring Matching!
    unique_languages = df_out[lang_col].dropna().unique()
    print(f"✅ Found {len(unique_languages)} languages to check: {', '.join(unique_languages)}\n")

    for lang in unique_languages:
        lang_clean = str(lang).strip()
        
        # Grab only the commands that completed for THIS specific language
        completed_for_lang = df_out[df_out[lang_col] == lang]['Command'].dropna().astype(str).tolist()
        
        # Normalize the completed commands for flexible matching
        completed_norm = [normalize(c) for c in completed_for_lang]
        
        # Subtraction via Substring
        missing = []
        matched_count = 0
        
        for raw_cmd in master_cmds:
            norm_raw = normalize(raw_cmd)
            is_matched = False
            
            for comp_cmd in completed_norm:
                # If the CSV command is hidden inside the Excel command (or vice versa)
                if norm_raw in comp_cmd or comp_cmd in norm_raw:
                    is_matched = True
                    break
                    
            if is_matched:
                matched_count += 1
            else:
                missing.append(raw_cmd)
                
        print(f"--- Language: {lang_clean} ---")
        print(f"   Completed in Excel: {len(completed_for_lang)}")
        print(f"   Successfully Matched: {matched_count} / {len(master_cmds)}")
        print(f"   Missing / Skipped to rerun: {len(missing)}")
        
        if missing:
            safe_lang_name = "".join([c for c in lang_clean if c.isalnum() or c == '_'])
            out_name = f"Skips_{safe_lang_name}.xlsx"
            
            out_df = pd.DataFrame({"Command": missing, "Tag": [""] * len(missing)})
            out_df.to_excel(out_name, index=False)
            print(f"   💾 Saved to: {out_name}\n")
        else:
            print("   ✅ 100% Complete! No skips.\n")

if __name__ == "__main__":
    recover_all_languages()