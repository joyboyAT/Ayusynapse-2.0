import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load the topics, trials and judgements data"""
    base_dir = Path(__file__).parent
    
    # Load topics data
    topics_df = pd.read_csv(base_dir / 'topics.csv')
    logging.info(f"Loaded {len(topics_df)} topics")
    
    # Load trials data
    trials_df = pd.read_csv(base_dir / 'trials.csv')
    logging.info(f"Loaded {len(trials_df)} trials")
    
    # Load judgements data
    judgements_df = pd.read_csv(base_dir / 'judgement.txt', sep='\t')
    logging.info(f"Loaded {len(judgements_df)} judgements")
    
    return topics_df, trials_df, judgements_df

def create_training_examples():
    """Create training examples by joining topics, trials and judgements"""
    topics_df, trials_df, judgements_df = load_data()
    
    # Map text labels to numeric
    label_map = {
        'eligible': 2,
        'excluded': 1, 
        'not_relevant': 0
    }
    judgements_df['label'] = judgements_df['relevance'].map(label_map)
    
    # Merge judgements with topics to get patient text
    merged_df = judgements_df.merge(topics_df, on='topic_id', how='left')
    
    # Merge with trials to get eligibility text
    merged_df = merged_df.merge(trials_df[['trial_id', 'eligibility_text', 'title']], 
                               on='trial_id', 
                               how='left')
    
    # ✅ FIXED: Use list instead of set
    training_df = merged_df[['patient_text', 'eligibility_text', 'title', 'label']]
    
    # Remove any rows with missing data
    training_df = training_df.dropna()
    
    # Save to CSV
    output_file = Path(__file__).parent / 'training_examples.csv'
    training_df.to_csv(output_file, index=False)
    logging.info(f"Created {len(training_df)} training examples")
    
    return training_df

if __name__ == "__main__":
    try:
        df = create_training_examples()
        print("\nSample of training data:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    except Exception as e:
        logging.error(f"Failed to create training data: {e}")
        raise
