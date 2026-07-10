import re
import pandas as pd

def parse_map_file(map_filename):
    """
    Parses the layout columns (Item Name, Start, Len) from a DHS .MAP file.
    """
    col_specs = []
    col_names = []
    
    # Regex to capture: Item Name (1), Item Label (2), Start position (3), Length (4)
    # Adjusts to grab rows with numeric start and length positions
    item_regex = re.compile(r'^(\w+)\s+(.*?)\s+(\d+)\s+(\d+)\s+[AN|N]')
    
    with open(map_filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = item_regex.match(line.strip())
            if match:
                name, _, start, length = match.groups()
                start = int(start)
                length = int(length)
                
                # Convert 1-based inclusive indices to 0-based Python slices (start, end_exclusive)
                py_start = start - 1
                py_end = py_start + length
                
                col_specs.append((py_start, py_end))
                col_names.append(name)
                
    return col_specs, col_names

def extract_nasarawa_data(map_file, dat_file, output_csv):
    """
    Loads data (either Stata .dta or fixed-width text file) and filters for Nasarawa (Region 15).
    """
    print("Parsing metadata layout from .MAP file...")
    specs, names = parse_map_file(map_file)
    
    if dat_file.lower().endswith('.dta'):
        print(f"Reading Stata data file ({dat_file})...")
        df = pd.read_stata(dat_file, convert_categoricals=False)
        
        # Convert column names to uppercase to align with .MAP layout naming convention
        df.columns = [col.upper() for col in df.columns]
        
        # Determine the correct region/state column
        if 'SSTATE' in df.columns:
            region_col = 'V024'
            print("Aligning SSTATE to V024 for schema compatibility...")
            df['V024'] = df['SSTATE']
        elif 'SHSTATE' in df.columns:
            region_col = 'HV024'
            print("Aligning SHSTATE to HV024 for schema compatibility...")
            df['HV024'] = df['SHSTATE']
        else:
            region_col = 'V024' if 'V024' in df.columns else 'HV024'
            
        # Cleanly convert numeric columns to string (removing .0 and handling NaN)
        # to ensure the output CSV looks identical to the ASCII text-based representation
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x == int(x) else ('' if pd.isnull(x) else str(x)))
            else:
                df[col] = df[col].astype(str).replace('nan', '')
                
        # Filter and order the columns to match the exact schema defined in the .MAP file
        existing_cols = [col for col in names if col in df.columns]
        df = df[existing_cols]
    else:
        print(f"Reading fixed-width data file ({dat_file})...")
        # Read fixed-width text file into a pandas dataframe
        df = pd.read_fwf(dat_file, colspecs=specs, names=names, dtype=str)
        region_col = 'V024' if 'V024' in names else 'HV024'
    
    # Convert region column to integer to safely filter
    df[region_col] = pd.to_numeric(df[region_col], errors='coerce')
    
    print("Filtering rows for Nasarawa (Code 15)...")
    nasarawa_df = df[df[region_col] == 15]
    
    # Save the filtered subset to a clean CSV
    nasarawa_df.to_csv(output_csv, index=False)
    print(f"Successfully saved Nasarawa data to {output_csv} ({len(nasarawa_df)} rows).")

# Example Usage:
# Replace with your actual file paths
map_path = "NGKR8BFL.MAP"  # Or "NGPR81FL.MAP"
dat_path = "NGKR8BFL.dta"  # Or "NGPR81FL.DAT"
output_path = "nasarawa_extracted_data_2.csv"

extract_nasarawa_data(map_path, dat_path, output_path)