# c:\AyuSynapse\load_qrels.py
import pandas as pd

def load_qrels(file_path, map_labels=True):
    # Read qrels file with space delimiter, specifying column names
    qrels_df = pd.read_csv(file_path, sep='\s+', names=['topic_id', 'Q0', 'trial_id', 'relevance'])
    
    # Drop the Q0 column as it's not needed
    qrels_df = qrels_df.drop('Q0', axis=1)
    
    # Optionally map numeric labels to text labels
    if map_labels:
        label_mapping = {
            2: 'eligible',
            1: 'excluded',
            0: 'not_relevant'
        }
        qrels_df['relevance'] = qrels_df['relevance'].map(label_mapping)
    
    # Save to judgement.txt
    qrels_df.to_csv('judgement.txt', sep='\t', index=False)
    
    return qrels_df

# Example usage
if __name__ == "__main__":
    # Replace with your qrels file path
    qrels_path = "qrels2021.txt"
    judgements = load_qrels(qrels_path)
    print("Judgements loaded and saved to judgement.txt")
    print("\nFirst few entries:")
    print(judgements.head())