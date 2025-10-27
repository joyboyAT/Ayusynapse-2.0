# BioBERT Setup Guide

## Quick Start (Recommended)

**After cloning the repository, run these commands to get back to working state:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download BioBERT models automatically
python setup_biobert.py

# 3. Verify installation
python test_trial_matching.py
```

## What the setup script does:

1. Downloads BioBERT models from HuggingFace
2. Caches them locally in `~/.cache/huggingface/`
3. Verifies model integrity
4. No manual downloads needed!

## Manual Setup (if needed)

If automatic setup fails:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install Transformers
pip install transformers tokenizers

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Models Used

- **BioBERT v1.2**: `dmis-lab/biobert-v1.1` - Base model trained on PubMed + PMC
- **BioBERT NER**: Fine-tuned for disease entity recognition
- **Clinical BERT**: Optimized for clinical notes

Models are automatically downloaded to: `C:\Users\YourUsername\.cache\huggingface\hub\`

## Troubleshooting

**Issue**: Out of memory error

- Solution: Use CPU instead of GPU or reduce batch size

**Issue**: Model download fails

- Solution: Check internet connection, try again later, or manually download from HuggingFace

**Issue**: Import errors

- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

## File Structure After Setup

```
AyuSynapse/
├── app.py
├── setup_biobert.py          # Run this first!
├── requirements.txt
├── BIOBERT_SETUP.md          # This file
├── README.md
├── cache/                    # Auto-generated (git ignored)
└── __pycache__/              # Auto-generated (git ignored)
```

**Note:** The `models/`, `cache/`, and `__pycache__/` folders are not tracked by Git. They will be regenerated when you run the setup script.
