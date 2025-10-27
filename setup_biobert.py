"""
Setup script to download and verify BioBERT models
Run this before first use to ensure all models are downloaded
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'tokenizers': 'Tokenizers'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True

def download_biobert_models():
    """Download BioBERT models from Hugging Face"""
    print("\n" + "="*70)
    print("Downloading BioBERT Models")
    print("="*70 + "\n")
    
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    
    models = {
        'BioBERT Base': 'dmis-lab/biobert-base-cased-v1.2',
        'BioBERT NER': 'alvaroalon2/biobert_diseases_ner',
        'Clinical BERT': 'emilyalsentzer/Bio_ClinicalBERT'
    }
    
    for name, model_path in models.items():
        try:
            print(f"Downloading {name}...")
            print(f"  Model: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                ignore_mismatched_sizes=True
            )
            
            print(f"✓ {name} downloaded successfully\n")
            
        except Exception as e:
            print(f"✗ Error downloading {name}: {str(e)}\n")

def test_biobert():
    """Test BioBERT pipeline"""
    print("\n" + "="*70)
    print("Testing BioBERT Pipeline")
    print("="*70 + "\n")
    
    try:
        from ayusynapse.models.biobert_pipeline import BioBERTPipeline
        
        test_text = "Patient diagnosed with type 2 diabetes and hypertension"
        
        print(f"Test input: {test_text}\n")
        
        biobert = BioBERTPipeline()
        entities = biobert.extract_entities(test_text, min_score=0.5)
        
        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  • {entity['entity']} ({entity['type']}) - Score: {entity['score']:.3f}")
        
        print("\n✓ BioBERT pipeline working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing BioBERT: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*20 + "BioBERT Setup Wizard" + " "*29 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nPlease install missing packages and run this script again.")
        sys.exit(1)
    
    # Step 2: Download models
    download_biobert_models()
    
    # Step 3: Test pipeline
    if test_biobert():
        print("\n" + "="*70)
        print("Setup Complete!")
        print("="*70)
        print("\nYou can now use BioBERT for medical entity extraction.")
        print("Run: python test_trial_matching.py")
    else:
        print("\nSetup completed with warnings. BioBERT may use fallback extraction.")

if __name__ == "__main__":
    main()
