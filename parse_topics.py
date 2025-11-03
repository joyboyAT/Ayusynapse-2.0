import xml.etree.ElementTree as ET
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_topics_xml(xml_path):
    """
    Parse the topics XML file and extract topic_id and patient_text.
    Returns a list of dictionaries containing the extracted information.
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Add debug logging
        logging.debug(f"XML Root tag: {root.tag}")
        
        # List to store extracted data
        topics_data = []
        
        # Extract data from each topic
        for topic in root.findall('.//topic'):
            topic_id = topic.get('number')
            
            # Get text directly from topic element, stripping whitespace
            patient_text = topic.text.strip() if topic.text else ''
            
            # Debug log first few characters of found text
            if patient_text:
                logging.info(f"Topic {topic_id} - Found text: {patient_text[:100]}...")
            else:
                logging.warning(f"No text found for topic {topic_id}")
            
            topics_data.append({
                'topic_id': topic_id,
                'patient_text': patient_text
            })
            
        return topics_data
    
    except ET.ParseError as e:
        logging.error(f"XML parsing error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error processing XML file: {e}")
        raise

def main():
    # Add debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Input and output paths
    input_xml = script_dir / "topics2021.xml"
    output_csv = script_dir / "topics.csv"
    
    try:
        # Parse XML
        logging.info(f"Parsing {input_xml}")
        topics_data = parse_topics_xml(input_xml)
        
        # Convert to DataFrame
        df = pd.DataFrame(topics_data)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logging.info(f"Successfully saved {len(topics_data)} topics to {output_csv}")
        
    except Exception as e:
        logging.error(f"Failed to process topics: {e}")
        raise

if __name__ == "__main__":
    main()
