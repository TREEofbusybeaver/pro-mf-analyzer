from mftool import Mftool

def search_mutual_fund(keyword):
    mf = Mftool()
    print("Fetching all mutual fund schemes from AMFI. This might take a moment...")
    
    # This returns a dictionary: { 'Scheme_Code': 'Scheme_Name' }
    all_schemes = mf.get_scheme_codes()
    
    print(f"\n--- Search Results for '{keyword}' ---")
    match_count = 0
    
    # Loop through the dictionary to find matches
    for code, name in all_schemes.items():
        if keyword.lower() in name.lower():
            print(f"Code: {code} | Name: {name}")
            match_count += 1
            
    if match_count == 0:
        print("No matching schemes found. Try a different keyword.")
    else:
        print(f"\nTotal matches found: {match_count}")

# --- Example Usage ---
# Change the string inside the quotes to search for your specific funds
search_mutual_fund("Midcap")
search_mutual_fund("Mid Cap")