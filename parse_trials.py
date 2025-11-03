import xml.etree.ElementTree as ET
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_element(element, path: str) -> str:
    """Extract text from XML element, handling None cases"""
    found = element.find(path)
    return found.text.strip() if found is not None and found.text else ''

def parse_trial_xml(xml_path: Path) -> Dict:
    """Parse single trial XML file and extract relevant fields"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        trial_data = {
            'trial_id': extract_text_from_element(root, './/nct_id'),
            'title': extract_text_from_element(root, './/brief_title'),
            'condition': extract_text_from_element(root, './/condition'),
            'summary': extract_text_from_element(root, './/brief_summary/textblock'),
            'detailed_description': extract_text_from_element(root, './/detailed_description/textblock'),
            'eligibility_text': extract_text_from_element(root, './/eligibility/criteria/textblock')
        }
        
        return trial_data
        
    except ET.ParseError as e:
        logging.error(f"Failed to parse {xml_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing {xml_path}: {e}")
        return None

def parse_trials_folder(folder_path: Path) -> List[Dict]:
    """Parse all XML files in the trials folder"""
    trials_data = []
    
    # More specific path pattern for NCT XML files
    xml_files = list(folder_path.rglob('*.xml'))  # Use rglob instead of glob
    
    logging.info(f"Looking for XML files in: {folder_path}")
    logging.info(f"Found {len(xml_files)} XML files to process")
    
    # Debug output first few file paths found
    for xml_file in xml_files[:5]:
        logging.debug(f"Sample file found: {xml_file}")
        
    for xml_file in xml_files:
        logging.debug(f"Processing {xml_file}")
        trial_data = parse_trial_xml(xml_file)
        if trial_data:
            trials_data.append(trial_data)
            
    return trials_data

def main():
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Folder containing NCT trials - make sure path exists
    trials_dir = script_dir / "NCT0000xxxx"
    output_csv = script_dir / "trials.csv"
    
    # Verify folder exists
    if not trials_dir.exists():
        logging.error(f"Trials directory not found: {trials_dir}")
        raise FileNotFoundError(f"Directory not found: {trials_dir}")
        
    try:
        # Parse all trials
        logging.info(f"Starting to parse trials from {trials_dir}")
        trials_data = parse_trials_folder(trials_dir)
        
        # Add check for empty results
        if not trials_data:
            logging.warning("No trial data was extracted!")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(trials_data)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logging.info(f"Successfully saved {len(trials_data)} trials to {output_csv}")
        
    except Exception as e:
        logging.error(f"Failed to process trials: {e}")
        raise

if __name__ == "__main__":
    main()
