"""
BioBERT Pipeline for Medical Named Entity Recognition
Uses official BioBERT model from dmis-lab (Google/Korea University)
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class BioBERTPipeline:
    """
    BioBERT-based medical entity extraction pipeline
    Uses the official BioBERT v1.1 model trained on PubMed + PMC
    """
    
    # Official BioBERT models from dmis-lab
    BIOBERT_MODELS = {
        'base': 'dmis-lab/biobert-base-cased-v1.2',  # BioBERT v1.2 (latest)
        'ner': 'dmis-lab/biobert-v1.1',  # BioBERT v1.1 for NER
        'clinical': 'emilyalsentzer/Bio_ClinicalBERT',  # Clinical variant
        'pubmed': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'  # Microsoft's variant
    }
    
    # Medical entity types mapping
    ENTITY_TYPE_MAP = {
        'DISEASE': 'condition',
        'CHEMICAL': 'medication',
        'GENE': 'gene',
        'SPECIES': 'organism',
        'MUTATION': 'mutation',
        'CELLLINE': 'cell_line',
        'PROTEIN': 'protein',
        'DNA': 'dna',
        'RNA': 'rna',
        'CELL': 'cell_type'
    }
    
    def __init__(self, model_name='base', device=None, use_ner_model=True):
        """
        Initialize BioBERT pipeline
        
        Args:
            model_name: Model variant to use ('base', 'ner', 'clinical', 'pubmed')
            device: Device to run on (cuda/cpu), auto-detected if None
            use_ner_model: Whether to use fine-tuned NER model
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading BioBERT model: {self.BIOBERT_MODELS.get(model_name, model_name)}")
        print(f"Using device: {self.device}")
        
        try:
            if use_ner_model:
                # Use pre-trained NER model for better entity recognition
                self._load_ner_model()
            else:
                # Use base BioBERT with simple tokenization
                self._load_base_model()
                
        except Exception as e:
            print(f"Error loading BioBERT model: {str(e)}")
            print("Falling back to rule-based extraction...")
            self.model = None
            self.tokenizer = None
            self.ner_pipeline = None
    
    def _load_ner_model(self):
        """Load fine-tuned NER model for medical entity extraction"""
        try:
            # Try to load a fine-tuned medical NER model
            model_options = [
                'alvaroalon2/biobert_diseases_ner',  # Disease NER
                'dmis-lab/biobert-v1.1',  # Base BioBERT
                'emilyalsentzer/Bio_ClinicalBERT',  # Clinical BERT
            ]
            
            for model_path in model_options:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModelForTokenClassification.from_pretrained(model_path)
                    
                    # Create NER pipeline
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == 'cuda' else -1,
                        aggregation_strategy="simple"
                    )
                    
                    print(f"✓ Loaded NER model: {model_path}")
                    return
                    
                except Exception as e:
                    continue
            
            # If all NER models fail, fall back to base model
            print("NER models unavailable, using base BioBERT...")
            self._load_base_model()
            
        except Exception as e:
            print(f"Error loading NER model: {str(e)}")
            self._load_base_model()
    
    def _load_base_model(self):
        """Load base BioBERT model for general medical text processing"""
        model_path = self.BIOBERT_MODELS.get(self.model_name, self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=9,  # Default number of entity types
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.ner_pipeline = None
        
        print(f"✓ Loaded base model: {model_path}")
    
    def extract_entities(self, text: str, min_score: float = 0.5, 
                        map_to_terminology: bool = False) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text using BioBERT
        
        Args:
            text: Input medical text
            min_score: Minimum confidence score for entities
            map_to_terminology: Whether to map to medical terminologies
            
        Returns:
            List of extracted entities with metadata
        """
        if not text:
            return []
        
        entities = []
        
        try:
            if self.ner_pipeline:
                # Use fine-tuned NER pipeline
                entities = self._extract_with_ner_pipeline(text, min_score)
            else:
                # Use rule-based extraction with BioBERT tokenization
                entities = self._extract_with_rules(text)
            
            # Post-process and categorize entities
            entities = self._post_process_entities(entities, text)
            
            # Optional: Map to medical terminologies (UMLS, SNOMED, etc.)
            if map_to_terminology and entities:
                entities = self._map_to_terminology(entities)
            
        except Exception as e:
            print(f"Error in entity extraction: {str(e)}")
            # Fallback to simple rule-based extraction
            entities = self._extract_with_rules(text)
        
        return entities
    
    def _extract_with_ner_pipeline(self, text: str, min_score: float) -> List[Dict[str, Any]]:
        """Extract entities using NER pipeline"""
        raw_entities = self.ner_pipeline(text)
        
        entities = []
        for ent in raw_entities:
            if ent['score'] >= min_score:
                entity_type = self._normalize_entity_type(ent.get('entity_group', 'UNKNOWN'))
                
                entities.append({
                    'entity': ent['word'],
                    'type': entity_type,
                    'score': ent['score'],
                    'start': ent['start'],
                    'end': ent['end']
                })
        
        return entities
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """Fallback rule-based entity extraction"""
        entities = []
        text_lower = text.lower()
        
        # Medical condition patterns
        condition_keywords = [
            'cancer', 'carcinoma', 'tumor', 'diabetes', 'hypertension',
            'disease', 'syndrome', 'disorder', 'infection', 'inflammation',
            'melanoma', 'leukemia', 'lymphoma', 'breast cancer', 'lung cancer'
        ]
        
        # Medication patterns
        medication_keywords = [
            'therapy', 'treatment', 'medication', 'drug', 'chemotherapy',
            'immunotherapy', 'radiation', 'metformin', 'insulin', 'aspirin',
            'antibiotics', 'inhibitor', 'blocker'
        ]
        
        # Extract conditions
        for keyword in condition_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                entities.append({
                    'entity': keyword.title(),
                    'type': 'condition',
                    'score': 0.8,
                    'start': start_idx,
                    'end': start_idx + len(keyword)
                })
        
        # Extract medications
        for keyword in medication_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                entities.append({
                    'entity': keyword.title(),
                    'type': 'medication',
                    'score': 0.7,
                    'start': start_idx,
                    'end': start_idx + len(keyword)
                })
        
        return entities
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity types to standard categories"""
        entity_type_upper = entity_type.upper()
        
        # Map BioBERT entity types to our categories
        if any(x in entity_type_upper for x in ['DISEASE', 'DISORDER', 'CONDITION']):
            return 'condition'
        elif any(x in entity_type_upper for x in ['CHEMICAL', 'DRUG', 'MEDICATION']):
            return 'medication'
        elif any(x in entity_type_upper for x in ['GENE', 'PROTEIN']):
            return 'gene'
        elif any(x in entity_type_upper for x in ['PROCEDURE', 'TREATMENT']):
            return 'procedure'
        else:
            return 'other'
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """Post-process and clean extracted entities"""
        processed = []
        seen = set()
        
        for entity in entities:
            entity_text = entity['entity'].strip()
            
            # Skip duplicates
            if entity_text.lower() in seen:
                continue
            
            # Skip very short entities (likely noise)
            if len(entity_text) < 3:
                continue
            
            # Skip common words
            stop_words = {'the', 'and', 'or', 'with', 'for', 'a', 'an', 'in', 'on', 'at'}
            if entity_text.lower() in stop_words:
                continue
            
            seen.add(entity_text.lower())
            processed.append(entity)
        
        return processed
    
    def _map_to_terminology(self, entities: List[Dict]) -> List[Dict]:
        """Map entities to medical terminologies (placeholder for UMLS integration)"""
        # This would integrate with UMLS client for concept mapping
        # For now, just add terminology hints based on entity type
        
        for entity in entities:
            entity_type = entity['type']
            
            if entity_type == 'condition':
                entity['terminology'] = 'SNOMED-CT'
                entity['category'] = 'Clinical Finding'
            elif entity_type == 'medication':
                entity['terminology'] = 'RxNorm'
                entity['category'] = 'Pharmaceutical Substance'
            elif entity_type == 'gene':
                entity['terminology'] = 'HGNC'
                entity['category'] = 'Gene/Protein'
        
        return entities
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between medical entities (future enhancement)
        
        Args:
            text: Input medical text
            
        Returns:
            List of entity relationships
        """
        # Placeholder for relation extraction
        # This would use BioBERT for relation classification
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_path': self.BIOBERT_MODELS.get(self.model_name, 'unknown'),
            'device': self.device,
            'has_ner_pipeline': self.ner_pipeline is not None,
            'pytorch_version': torch.__version__,
        }


# Standalone function for quick testing
def extract_medical_entities(text: str, min_score: float = 0.5) -> List[Dict[str, Any]]:
    """
    Quick extraction function for testing
    
    Args:
        text: Medical text to analyze
        min_score: Minimum confidence score
        
    Returns:
        List of extracted entities
    """
    pipeline = BioBERTPipeline()
    return pipeline.extract_entities(text, min_score=min_score)


if __name__ == "__main__":
    # Test the BioBERT pipeline
    test_text = "65-year-old female with stage 2 breast cancer, HER2 positive, currently on chemotherapy"
    
    print("="*70)
    print("BioBERT Medical Entity Extraction Test")
    print("="*70)
    print(f"\nInput: {test_text}\n")
    
    # Initialize pipeline
    biobert = BioBERTPipeline()
    
    # Extract entities
    entities = biobert.extract_entities(test_text, min_score=0.5, map_to_terminology=True)
    
    print(f"Extracted {len(entities)} entities:\n")
    for i, entity in enumerate(entities, 1):
        print(f"{i}. {entity['entity']}")
        print(f"   Type: {entity['type']}")
        print(f"   Score: {entity['score']:.3f}")
        if 'terminology' in entity:
            print(f"   Terminology: {entity['terminology']}")
        print()
    
    # Model info
    print("\nModel Information:")
    info = biobert.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
