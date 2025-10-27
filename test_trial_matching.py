from trial_matching import PatientTrialMatcher
from medical_text_processor import MedicalTextProcessor
from umls_client import UMLSClient
from fhir_converter import FHIRConverter
from fhir_client import FHIRClient
import os
import sys
import re
import json

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def get_user_input():
    """Get unstructured medical text from user"""
    print_section_header("PATIENT MEDICAL HISTORY INPUT")
    print("\nPlease enter the patient's medical history in natural language.")
    print("Example: '65-year-old female with stage 2 breast cancer, HER2 positive, currently on chemotherapy'")
    print("\nYour input:")
    
    medical_text = input().strip()
    
    if not medical_text:
        print("Error: No input provided. Using default example.")
        medical_text = "65-year-old female with stage 2 breast cancer, HER2 positive, currently on chemotherapy"
    
    return medical_text

def extract_biobert_entities(medical_text):
    """Extract medical entities using BioBERT"""
    print_section_header("STEP 1: BIOBERT ENTITY EXTRACTION")
    
    try:
        # Import BioBERT pipeline from the correct location
        from ayusynapse.models.biobert_pipeline import BioBERTPipeline
        
        print(f"Input text: {medical_text}\n")
        print("Extracting medical entities using BioBERT...")
        
        # Initialize BioBERT pipeline
        biobert_pipeline = BioBERTPipeline()
        
        # Extract entities using BioBERT
        raw_entities = biobert_pipeline.extract_entities(medical_text, min_score=0.5, map_to_terminology=True)
        
        # Convert to expected format
        entities = {
            'conditions': [],
            'medications': [],
            'procedures': [],
            'anatomy': [],
            'age': None,
            'gender': None,
            'cancer_details': {}
        }
        
        # Process BioBERT results and categorize
        for entity in raw_entities:
            entity_text = entity.get('entity', '').strip()
            entity_type = entity.get('type', '').upper()
            
            # Categorize based on entity text content
            if any(term in entity_text.lower() for term in ['cancer', 'tumor', 'carcinoma', 'diabetes', 'hypertension', 'disease']):
                if entity_text not in entities['conditions']:
                    entities['conditions'].append(entity_text)
            elif any(term in entity_text.lower() for term in ['therapy', 'treatment', 'medication', 'drug']):
                if entity_text not in entities['medications']:
                    entities['medications'].append(entity_text)
        
        # Use fallback processor for demographics and cancer details
        processor = MedicalTextProcessor()
        fallback_entities = processor.extract_medical_entities(medical_text)
        
        # Merge demographics and cancer details from fallback
        entities['age'] = fallback_entities.get('age')
        entities['gender'] = fallback_entities.get('gender')
        entities['cancer_details'] = fallback_entities.get('cancer_details', {})
        
        # Add any conditions/medications missed by BioBERT
        for condition in fallback_entities.get('conditions', []):
            if condition not in entities['conditions']:
                entities['conditions'].append(condition)
        
        for medication in fallback_entities.get('medications', []):
            if medication not in entities['medications']:
                entities['medications'].append(medication)
        
        print("\nExtracted Entities (BioBERT + Fallback):")
        for entity_type, entity_list in entities.items():
            if entity_list and isinstance(entity_list, list):
                print(f"  {entity_type.upper()}: {', '.join(entity_list)}")
            elif entity_list:
                print(f"  {entity_type.upper()}: {entity_list}")
        
        return entities
        
    except ImportError as e:
        print(f"Warning: BioBERT pipeline not found ({str(e)}). Using fallback extraction...")
        # Fallback to MedicalTextProcessor
        processor = MedicalTextProcessor()
        entities = processor.extract_medical_entities(medical_text)
        
        print("\nExtracted Entities (Fallback):")
        for entity_type, entity_list in entities.items():
            if entity_list and entity_type in ['conditions', 'medications']:
                print(f"  {entity_type.upper()}: {', '.join(entity_list) if isinstance(entity_list, list) else entity_list}")
        
        return entities
    except Exception as e:
        print(f"Error in BioBERT extraction: {str(e)}. Using fallback...")
        # Fallback to MedicalTextProcessor
        processor = MedicalTextProcessor()
        entities = processor.extract_medical_entities(medical_text)
        
        print("\nExtracted Entities (Fallback due to error):")
        for entity_type, entity_list in entities.items():
            if entity_list and entity_type in ['conditions', 'medications']:
                print(f"  {entity_type.upper()}: {', '.join(entity_list) if isinstance(entity_list, list) else entity_list}")
        
        return entities

def map_to_umls_realtime(entities, umls_client):
    """Map extracted entities to UMLS concepts using real-time API"""
    print_section_header("STEP 2: UMLS CONCEPT MAPPING (Real-time API)")
    
    umls_concepts = {
        'conditions': [],
        'medications': []
    }
    
    # Map conditions to UMLS
    conditions = entities.get('conditions', [])
    if conditions:
        print(f"\nMapping {len(conditions)} conditions to UMLS...")
        umls_concepts['conditions'] = umls_client.map_medical_terms(conditions)
        
        print("\nUMLS Condition Mappings:")
        for concept in umls_concepts['conditions']:
            print(f"  • {concept['original_term']}")
            print(f"    → {concept['preferred_name']} (CUI: {concept['cui']})")
            if concept.get('semantic_types'):
                print(f"    → Semantic Types: {', '.join(concept['semantic_types'][:2])}")
    
    # Map medications to UMLS
    medications = entities.get('medications', [])
    if medications:
        print(f"\nMapping {len(medications)} medications to UMLS...")
        umls_concepts['medications'] = umls_client.map_medical_terms(medications)
        
        print("\nUMLS Medication Mappings:")
        for concept in umls_concepts['medications']:
            print(f"  • {concept['original_term']}")
            print(f"    → {concept['preferred_name']} (CUI: {concept['cui']})")
    
    return umls_concepts

def create_structured_profile(medical_text, entities, umls_concepts):
    """Create structured patient profile"""
    print_section_header("STEP 3: STRUCTURED PATIENT PROFILE")
    
    # Extract demographics
    processor = MedicalTextProcessor()
    demographics = processor.extract_basic_demographics(medical_text)
    
    # Add data validation
    def validate_demographics(profile):
        age = profile['structured_data'].get('age')
        if age and (age < 0 or age > 150):
            print(f"Warning: Invalid age {age}, setting to None")
            profile['structured_data']['age'] = None
        
        gender = profile['structured_data'].get('gender')
        if gender and gender not in ['M', 'F']:
            print(f"Warning: Invalid gender {gender}, setting to None")
            profile['structured_data']['gender'] = None
    
    # Preserve detailed cancer information
    def enhance_cancer_details(profile, text):
        cancer_details = {}
        # Extract stage with better parsing
        stage_pattern = r'stage\s*([1-4iv]+)\s*(a|b)?'
        stage_match = re.search(stage_pattern, text.lower())
        if stage_match:
            cancer_details['stage'] = stage_match.group(0).upper()
            cancer_details['substage'] = stage_match.group(2) if stage_match.group(2) else None
        
        # Extract biomarkers with confidence
        biomarkers = ['HER2', 'ER', 'PR', 'BRCA1', 'BRCA2']
        for marker in biomarkers:
            pos_pattern = f'{marker.lower()}[\\s-]*positive|{marker.lower()}\\+'
            neg_pattern = f'{marker.lower()}[\\s-]*negative|{marker.lower()}\\-'
            if re.search(pos_pattern, text.lower()):
                cancer_details[f'{marker}_status'] = 'positive'
            elif re.search(neg_pattern, text.lower()):
                cancer_details[f'{marker}_status'] = 'negative'
        
        profile['structured_data']['cancer_details'] = cancer_details
    
    # Add medication detail extraction
    def extract_medication_details(text):
        medications = []
        # Pattern for "drug dose frequency"
        med_pattern = r'(\w+)\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?)?\s*(daily|twice|bid|tid|qid)?'
        matches = re.finditer(med_pattern, text.lower())
        
        for match in matches:
            medications.append({
                'name': match.group(1),
                'dose': f"{match.group(2)} {match.group(3)}" if match.group(3) else match.group(2),
                'frequency': match.group(4) if match.group(4) else 'unspecified'
            })
        return medications
    
    # Build structured profile
    profile = {
        'patient_id': 'USER_INPUT_001',
        'raw_text': medical_text,
        'structured_data': {
            'age': entities.get('age') or demographics.get('age'),
            'gender': entities.get('gender') or demographics.get('gender'),
            'conditions': entities.get('conditions', []),
            'medications': entities.get('medications', []),
            'cancer_details': entities.get('cancer_details', {})
        },
        'umls_concepts': umls_concepts
    }
    
    validate_demographics(profile)
    enhance_cancer_details(profile, medical_text)
    profile['structured_data']['detailed_medications'] = extract_medication_details(medical_text)
    
    # Print structured profile
    print("\nStructured Patient Profile:")
    print(f"  Patient ID: {profile['patient_id']}")
    print(f"  Age: {profile['structured_data'].get('age', 'Not specified')}")
    print(f"  Gender: {profile['structured_data'].get('gender', 'Not specified')}")
    print(f"  Conditions: {', '.join(profile['structured_data']['conditions']) if profile['structured_data']['conditions'] else 'None'}")
    print(f"  Medications: {', '.join(profile['structured_data']['medications']) if profile['structured_data']['medications'] else 'None'}")
    
    if profile['structured_data'].get('cancer_details'):
        print(f"  Cancer Details: {profile['structured_data']['cancer_details']}")
    
    return profile

def convert_to_fhir_format(profile):
    """Convert patient profile to FHIR format"""
    print_section_header("STEP 3.5: FHIR CONVERSION (Global Interoperability)")
    
    print("\nConverting patient profile to HL7 FHIR R4 format...")
    
    # Initialize FHIR converter
    fhir_converter = FHIRConverter()
    
    # Convert to FHIR Bundle
    fhir_bundle = fhir_converter.convert_to_fhir_bundle(profile)
    
    # Validate FHIR bundle
    is_valid, errors = fhir_converter.validate_fhir_bundle(fhir_bundle)
    
    if is_valid:
        print("✓ FHIR Bundle created successfully")
    else:
        print("⚠ FHIR Bundle validation warnings:")
        for error in errors:
            print(f"  • {error}")
    
    # Display FHIR bundle summary
    print("\nFHIR Bundle Summary:")
    print(f"  Resource Type: {fhir_bundle['resourceType']}")
    print(f"  Bundle Type: {fhir_bundle['type']}")
    print(f"  Total Entries: {len(fhir_bundle.get('entry', []))}")
    
    # Show resource breakdown
    resource_types = {}
    for entry in fhir_bundle.get('entry', []):
        resource_type = entry.get('resource', {}).get('resourceType', 'Unknown')
        resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
    
    print("\n  Resource Breakdown:")
    for resource_type, count in resource_types.items():
        print(f"    - {resource_type}: {count}")
    
    # Save FHIR bundle to file
    fhir_output_path = "patient_fhir_bundle.json"
    with open(fhir_output_path, 'w', encoding='utf-8') as f:
        json.dump(fhir_bundle, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ FHIR Bundle saved to: {fhir_output_path}")
    
    # Optional: Try to post to FHIR server (HAPI FHIR test server)
    print("\nAttempting to validate with HAPI FHIR test server...")
    try:
        fhir_client = FHIRClient()
        
        # Validate Patient resource
        patient_resource = None
        for entry in fhir_bundle.get('entry', []):
            if entry.get('resource', {}).get('resourceType') == 'Patient':
                patient_resource = entry['resource']
                break
        
        if patient_resource:
            validation_result = fhir_client.validate_resource(patient_resource)
            if validation_result.get('success'):
                print("✓ Patient resource validated successfully with FHIR server")
            else:
                print(f"⚠ Validation response: {validation_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"⚠ FHIR server validation skipped: {str(e)}")
    
    print("\nFHIR Bundle Benefits:")
    print("  • Global interoperability with EHR systems")
    print("  • Standardized clinical data exchange")
    print("  • Compatible with clinical trial management systems")
    print("  • Enables data sharing across healthcare organizations")
    
    return fhir_bundle

def match_clinical_trials(profile, matcher):
    """Match patient with clinical trials using real-time API"""
    print_section_header("STEP 4: CLINICAL TRIAL MATCHING (Real-time API)")
    
    print("\nSearching ClinicalTrials.gov for matching trials...")
    matches = matcher.match_patient_to_trials(profile, max_results=5, use_real_api=True)
    
    print(f"\nFound {len(matches)} matching clinical trials")
    
    return matches

def assess_recruitment_eligibility(match, profile):
    """Enhanced eligibility assessment with medical logic"""
    trial = match['trial']
    score = match['score']
    reasons = match.get('match_reasons', {})
    
    # Multi-factor eligibility determination
    eligibility_factors = {
        'condition_match': False,
        'age_eligible': False,
        'gender_eligible': False,
        'no_contraindications': False,
        'appropriate_phase': False,
        'location_accessible': False
    }
    
    explanation_parts = []
    warnings = []
    
    # Detailed condition assessment
    if 'primary_condition' in reasons or 'biomarker_matches' in reasons:
        eligibility_factors['condition_match'] = True
        explanation_parts.append("✓ Medical condition matches trial focus")
    else:
        explanation_parts.append("✗ No clear condition match found")
    
    # Enhanced age assessment with medical context
    patient_age = profile['structured_data'].get('age')
    if patient_age:
        age_range = trial.get('age_range', {})
        min_age = age_range.get('min', 0)
        max_age = age_range.get('max', 150)
        
        if min_age <= patient_age <= max_age:
            eligibility_factors['age_eligible'] = True
            explanation_parts.append(f"✓ Age {patient_age} meets requirements ({min_age}-{max_age} years)")
        else:
            explanation_parts.append(f"✗ Age {patient_age} outside required range ({min_age}-{max_age} years)")
            
        # Age-related warnings
        if patient_age >= 75:
            warnings.append("Advanced age may require additional safety monitoring")
    
    # Check for potential contraindications
    contraindications = _check_contraindications(profile, trial)
    if not contraindications:
        eligibility_factors['no_contraindications'] = True
        explanation_parts.append("✓ No obvious contraindications identified")
    else:
        explanation_parts.append("⚠ Potential contraindications found:")
        for contraindication in contraindications:
            explanation_parts.append(f"  • {contraindication}")
    
    # Phase appropriateness assessment
    phase_appropriate, phase_explanation = _assess_phase_appropriateness(profile, trial)
    eligibility_factors['appropriate_phase'] = phase_appropriate
    explanation_parts.append(phase_explanation)
    
    # Determine final eligibility
    required_factors = ['condition_match', 'age_eligible', 'gender_eligible']
    recommended_factors = ['no_contraindications', 'appropriate_phase']
    
    meets_requirements = all(eligibility_factors[factor] for factor in required_factors)
    meets_recommendations = sum(eligibility_factors[factor] for factor in recommended_factors) >= 1
    
    if meets_requirements and meets_recommendations and score >= 70:
        eligible = True
        recommendation = "STRONGLY RECOMMENDED"
    elif meets_requirements and score >= 50:
        eligible = True
        recommendation = "RECOMMENDED WITH CAUTION"
    else:
        eligible = False
        recommendation = "NOT RECOMMENDED"
    
    # Add next steps
    if eligible:
        explanation_parts.append("\nNext Steps:")
        explanation_parts.append("• Discuss with your oncologist/physician")
        explanation_parts.append("• Review detailed inclusion/exclusion criteria")
        explanation_parts.append("• Contact study coordinator if interested")
    else:
        explanation_parts.append("\nAlternative Options:")
        explanation_parts.append("• Ask physician about similar trials")
        explanation_parts.append("• Consider broadening geographic search")
        explanation_parts.append("• Explore compassionate use programs")
    
    return eligible, explanation_parts

def _check_contraindications(profile, trial):
    """Check for potential medical contraindications"""
    contraindications = []
    
    conditions = profile['structured_data'].get('conditions', [])
    medications = profile['structured_data'].get('medications', [])
    
    # Common contraindication patterns
    if 'liver disease' in [c.lower() for c in conditions] and 'chemotherapy' in trial.get('title', '').lower():
        contraindications.append("Liver disease may limit chemotherapy eligibility")
    
    if 'kidney disease' in [c.lower() for c in conditions] and any(med.lower() in ['cisplatin', 'carboplatin'] for med in medications):
        contraindications.append("Kidney disease concerns with platinum-based therapy")
    
    return contraindications

def _assess_phase_appropriateness(profile, trial):
    """Assess if trial phase is appropriate for patient's condition stage"""
    phase = trial.get('phase', '').lower()
    conditions = profile['structured_data'].get('conditions', [])
    cancer_details = profile['structured_data'].get('cancer_details', {})
    
    if 'phase 1' in phase:
        if any('advanced' in c.lower() or 'metastatic' in c.lower() for c in conditions):
            return True, "✓ Phase 1 appropriate for advanced disease"
        else:
            return False, "⚠ Phase 1 typically for advanced disease only"
    elif 'phase 3' in phase:
        if cancer_details.get('stage', '').lower() in ['stage i', 'stage ii', 'stage iii']:
            return True, "✓ Phase 3 appropriate for earlier-stage disease"
        else:
            return True, "✓ Phase 3 trial (standard of care comparison)"
    else:
        return True, f"✓ {phase.title()} trial appropriate"

def display_trial_results(matches, profile):
    """Display trial matching results with recruitment eligibility"""
    print_section_header("STEP 5: RECRUITMENT ELIGIBILITY ASSESSMENT")
    
    if not matches:
        print("\nNo matching trials found. Consider:")
        print("  1. Broadening search criteria")
        print("  2. Checking for trials in nearby locations")
        print("  3. Consulting with healthcare provider for alternatives")
        return
    
    for i, match in enumerate(matches, 1):
        trial = match['trial']
        
        print(f"\n{'─'*70}")
        print(f"TRIAL #{i}: {trial['nct_id']}")
        print(f"{'─'*70}")
        print(f"Title: {trial['title']}")
        print(f"Match Score: {match['score']}/100")
        
        # Assess eligibility
        eligible, explanation = assess_recruitment_eligibility(match, profile)
        
        # Display eligibility verdict
        if eligible:
            print(f"\n✓ RECRUITMENT ELIGIBILITY: YES - RECOMMENDED")
        else:
            print(f"\n✗ RECRUITMENT ELIGIBILITY: NO - NOT RECOMMENDED")
        
        # Display explanation
        print(f"\nExplanation:")
        for reason in explanation:
            if reason.startswith('  →'):
                print(f"    {reason}")
            else:
                print(f"  • {reason}")
        
        # Additional trial info
        print(f"\nAdditional Information:")
        print(f"  • Conditions: {', '.join(trial.get('conditions', [])[:3])}")
        print(f"  • Phase: {trial.get('phase', 'Not specified')}")
        print(f"  • Study Type: {trial.get('study_type', 'Not specified')}")
        
        # Show locations
        locations = trial.get('locations', [])
        if locations:
            print(f"  • Available at {len(locations)} location(s):")
            for loc in locations[:3]:  # Show first 3
                loc_parts = []
                if loc.get('facility'):
                    loc_parts.append(loc['facility'])
                if loc.get('city'):
                    loc_parts.append(loc['city'])
                if loc.get('state'):
                    loc_parts.append(loc['state'])
                if loc_parts:
                    print(f"    - {', '.join(loc_parts)}")

def main():
    """Main execution flow"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "CLINICAL TRIAL MATCHING SYSTEM" + " "*23 + "║")
    print("║" + " "*10 + "BioBERT + UMLS + FHIR + ClinicalTrials.gov" + " "*16 + "║")
    print("╚" + "═"*68 + "╝")
    
    try:
        # Initialize UMLS client (real-time API)
        print("\nInitializing UMLS API connection...")
        umls_client = UMLSClient()
        print("✓ UMLS API connected successfully")
        
        # Initialize trial matcher
        matcher = PatientTrialMatcher(umls_client=umls_client)
        
        # Step 1: Get user input
        medical_text = get_user_input()
        
        # Step 2: Extract entities using BioBERT
        entities = extract_biobert_entities(medical_text)
        
        # Step 3: Map to UMLS (real-time API)
        umls_concepts = map_to_umls_realtime(entities, umls_client)
        
        # Step 4: Create structured profile
        profile = create_structured_profile(medical_text, entities, umls_concepts)
        
        # Step 4.5: Convert to FHIR format (NEW)
        fhir_bundle = convert_to_fhir_format(profile)
        
        # Step 5: Match with clinical trials (real-time API)
        matches = match_clinical_trials(profile, matcher)
        
        # Step 6: Display results with eligibility assessment
        display_trial_results(matches, profile)
        
        print_section_header("PROCESS COMPLETE")
        print("\nGenerated Outputs:")
        print("  1. Structured Patient Profile (in-memory)")
        print("  2. FHIR Bundle (patient_fhir_bundle.json)")
        print("  3. Clinical Trial Matches (displayed above)")
        print("\nFor more information about any trial, visit: https://clinicaltrials.gov/")
        print("Always consult with your healthcare provider before enrolling in a clinical trial.")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
