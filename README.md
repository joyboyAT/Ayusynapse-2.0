# Ayusynapse Project

⚠️ **DISCLAIMER - RESEARCH PROJECT ONLY**

**This project is currently under active development and is NOT intended for clinical use.**

- ⚠️ **NOT FDA/Medical Authority Approved**: This system is a research prototype and has not been validated for clinical decision-making
- 🔬 **Development Stage**: Training accuracy, model performance, and clinical validation are ongoing goals for the near future
- 🏥 **Not a Substitute for Medical Advice**: Always consult qualified healthcare professionals for medical decisions
- ✅ **Authenticated Data Sources**: While we use government-authenticated APIs (ClinicalTrials.gov, UMLS) and well-established models (BioBERT), the integration and outputs require further clinical validation
- 📊 **Research Purpose**: This tool is designed for educational and research purposes to explore AI applications in healthcare

**NEVER use this system for actual patient care, diagnosis, or treatment decisions without proper medical oversight.**

---

## 🚀 Quick Start

**Clone and setup:**

```bash
# 1. Clone the repository
git clone https://github.com/joyboyAT/Ayusynapse-2.0.git
cd Ayusynapse-2.0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download BioBERT models (required!)
python setup_biobert.py

# 4. Run the application
python app.py
```

**First time setup takes 5-10 minutes to download models (~1GB)**

## System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AYUSYNAPSE PIPELINE                                 │
│                        ⚠️ RESEARCH PROTOTYPE ONLY                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Patient Input   │
│  - Clinical Notes│
│  - EHR Data      │──────┐
│  - Reports       │      │
└──────────────────┘      │
                          ▼
                ┌──────────────────────┐
                │  Data Preprocessing  │
                │  - Text Cleaning     │
                │  - Tokenization      │
                │  - Normalization     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   BioBERT NER Model  │───────────────────┐
                │                      │                   │
                │  Why BioBERT?        │                   │
                │  ✓ Pre-trained on    │                   │
                │    PubMed + PMC      │                   │
                │  ✓ Domain-specific   │                   │
                │    biomedical vocab  │                   │
                │  ✓ F1: 0.75-0.90     │                   │
                │  ✓ Recognizes:       │                   │
                │    - Diseases        │                   │
                │    - Medications     │                   │
                │    - Symptoms        │                   │
                │    - Procedures      │                   │
                └──────────┬───────────┘                   │
                           │                               │
                           ▼                               │
                ┌──────────────────────┐                   │
                │  Named Entities      │                   │
                │  Extracted           │                   │
                │  - Cancer types      │                   │
                │  - Drug names        │                   │
                │  - Lab values        │                   │
                └──────────┬───────────┘                   │
                           │                               │
                           ▼                               │
                ┌──────────────────────┐                   │
                │  Medical Terminology │                   │
                │  Mapping (UMLS)      │                   │
                │                      │                   │
                │  - UMLS CUI mapping  │                   │
                │  - SNOMED CT codes   │                   │
                │  - RxNorm (drugs)    │                   │
                │  - ICD-10 codes      │                   │
                │                      │                   │
                │  Purpose:            │                   │
                │  ✓ Standardization   │                   │
                │  ✓ Interoperability  │                   │
                │  ✓ Semantic linking  │                   │
                └──────────┬───────────┘                   │
                           │                               │
                           ▼                               │
                ┌──────────────────────┐                   │
                │  Enhanced Patient    │◄──────────────────┘
                │  Profile Generation  │
                │                      │
                │  Structured Data:    │
                │  - Diagnoses (coded) │
                │  - Medications       │
                │  - Allergies         │
                │  - Lab results       │
                │  - Treatment history │
                │  - Risk factors      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  FHIR Conversion     │
                │                      │
                │  Resources Created:  │
                │  - Patient           │
                │  - Condition         │
                │  - MedicationRequest │
                │  - Observation       │
                │  - Procedure         │
                │                      │
                │  Why FHIR?           │
                │  ✓ HL7 standard      │
                │  ✓ REST API ready    │
                │  ✓ EHR integration   │
                │  ✓ JSON format       │
                └──────────┬───────────┘
                           │
                           ▼
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────┐
│ Clinical Trial   │              │  Treatment       │
│ Matching API     │              │  Recommendation  │
│                  │              │  Engine          │
│ ✅ ClinicalTrials│              │                  │
│   .gov API       │              │ - Evidence-based │
│   (Govt Auth)    │              │   guidelines     │
│ - Eligibility    │              │ - Drug-drug      │
│   criteria match │              │   interactions   │
│ - Geographic     │              │ - Personalized   │
│   filtering      │              │   protocols      │
│ - Phase filtering│              │                  │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
         │                                 │
         ▼                                 ▼
┌─────────────────────────────────────────────────────┐
│           Clinical Decision Support                 │
│               ⚠️ FOR RESEARCH ONLY                   │
│                                                     │
│  - Trial recommendations                            │
│  - Treatment options                                │
│  - Risk assessments                                 │
│  - Follow-up protocols                              │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Output Dashboard    │
         │  - Patient summary   │
         │  - Matched trials    │
         │  - Treatment plans   │
         │  - FHIR export       │
         └──────────────────────┘
```

## Technical Stack

| Component | Technology | Purpose | Status |
|-----------|-----------|---------|--------|
| NER Model | BioBERT (BERT-base + PubMed/PMC) | Domain-specific entity extraction | ✅ Well-established |
| Terminology | UMLS, SNOMED CT, RxNorm | Medical concept standardization | ✅ Govt-authenticated |
| Interoperability | FHIR R4 | Healthcare data exchange | ✅ HL7 standard |
| Clinical Trials | ClinicalTrials.gov API | Trial matching & recruitment | ✅ NIH/Govt-authenticated |
| Caching | Redis/In-memory | Fast inference & response | 🔧 In development |
| Backend | Flask/FastAPI | REST API endpoints | 🔧 In development |

**Legend:**
- ✅ = Established/Authenticated source
- 🔧 = Under active development
- 🎯 = Future enhancement

## 📦 What's Included

- ✅ Source code for all components
- ✅ Setup scripts for automatic model download
- ✅ Sample data and test cases
- ✅ Documentation and API guides
- ❌ Pre-trained models (download via `setup_biobert.py`)
- ❌ Cache files (auto-generated)

## ⚠️ Important Notes

### Data Sources (Authenticated & Reliable)
- **ClinicalTrials.gov**: Official U.S. National Library of Medicine database
- **UMLS (Unified Medical Language System)**: NLM's comprehensive medical vocabulary
- **BioBERT**: Published, peer-reviewed model trained on PubMed/PMC
- **SNOMED CT & RxNorm**: International clinical terminology standards

### Current Limitations
- 🔬 **Training Accuracy**: Model fine-tuning and validation are ongoing
- 📊 **Clinical Validation**: Outputs have not undergone clinical trials or regulatory approval
- 🎯 **Performance Metrics**: Achieving production-grade accuracy is a near-future goal
- 🔐 **Data Privacy**: HIPAA compliance features under development

### Roadmap
- [ ] Clinical validation with healthcare professionals
- [ ] Improved NER accuracy for rare conditions
- [ ] Integration with real EHR systems
- [ ] Regulatory compliance (HIPAA, FDA guidelines)
- [ ] Multi-language support
- [ ] Real-time monitoring and alerting

## 🏥 Medical Disclaimer

**THIS SOFTWARE IS PROVIDED "AS IS" FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

While this system integrates government-authenticated data sources (ClinicalTrials.gov, UMLS) and medically renowned models (BioBERT), the **system as a whole has not been clinically validated or approved for medical use**.

**For actual medical advice:**
- ✅ Consult licensed healthcare professionals
- ✅ Use FDA-approved diagnostic tools
- ✅ Follow established clinical guidelines
- ❌ Do NOT rely solely on this system for medical decisions

## 🤝 Contributing & Collaboration

This is an **open-source research project** dedicated to advancing cancer research and clinical trial accessibility through AI-driven solutions.

### We Welcome Contributions

We actively encourage contributions from:
- 🔬 **Researchers** in oncology, bioinformatics, and AI/ML
- 💻 **Developers** interested in healthcare technology
- 🏥 **Healthcare professionals** with domain expertise
- 📊 **Data scientists** working on medical NLP

### How to Contribute

**Code Contributions:**
```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and commit
git commit -m "Add: description of your contribution"

# 4. Push and create a Pull Request
git push origin feature/your-feature-name
```

**Areas for Contribution:**
- Model optimization and accuracy improvements
- Additional medical terminology mappings
- Enhanced clinical trial matching algorithms
- FHIR resource template expansions
- Documentation and testing
- Bug fixes and performance enhancements

### Contact for Collaboration

For constructive feedback, research collaboration, or significant contributions to cancer research and clinical trial advancement:

📧 **Email**: img_2023010@iiitm.ac.in

**Please include in your message:**
- Your area of expertise
- Proposed contribution or collaboration idea
- Relevant background or publications (if applicable)

### Code of Conduct

We are committed to fostering an inclusive and professional research environment. All contributors are expected to:
- Maintain scientific rigor and ethical standards
- Respect patient privacy and data protection regulations
- Provide constructive feedback and collaborate respectfully
- Acknowledge the limitations of AI in healthcare

**Together, we can advance cancer research and improve clinical trial accessibility through open collaboration.** 
