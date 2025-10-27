from clinical_trials_client import ClinicalTrialsClient
from typing import Dict, List, Any
from umls_client import UMLSClient
import sys

class PatientTrialMatcher:
    """Match patients to relevant clinical trials based on medical criteria"""
    
    def __init__(self, umls_api_key=None, umls_client=None, fhir_enabled=True):
        """Initialize the trial matcher with UMLS integration
        
        Args:
            umls_api_key: API key for UMLS (optional)
            umls_client: Pre-initialized UMLSClient instance (optional)
            fhir_enabled: Enable FHIR format support (default: True)
        """
        # Use the provided client or create a new one
        if umls_client:
            self.umls_client = umls_client
        else:
            try:
                self.umls_client = UMLSClient(api_key=umls_api_key)
            except Exception as e:
                print(f"Warning: UMLS client initialization failed: {str(e)}", file=sys.stderr)
                self.umls_client = None
        
        # Initialize the clinical trials API client
        self.trials_client = ClinicalTrialsClient(cache_enabled=True)
        
        self.fhir_enabled = fhir_enabled
        if fhir_enabled:
            try:
                from fhir_converter import FHIRConverter
                self.fhir_converter = FHIRConverter()
            except ImportError:
                print("Warning: FHIR converter not available", file=sys.stderr)
                self.fhir_enabled = False
    
    def match_patient_to_trials(self, patient_profile, max_results=10, use_real_api=True):
        """Match patients to clinical trials based on patient profile
        
        Args:
            patient_profile: Patient information
            max_results: Maximum number of trials to return
            use_real_api: Whether to use real-time API or sample data
        """
        # Extract primary patient characteristics
        if 'structured_data' in patient_profile:
            # Profile created from raw text
            conditions = patient_profile['structured_data'].get('conditions', [])
            age = patient_profile['structured_data'].get('age')
            gender = patient_profile['structured_data'].get('gender')
            location = patient_profile['structured_data'].get('location', {})
        else:
            # Traditional profile structure
            conditions = patient_profile.get('conditions', [])
            age = patient_profile.get('age')
            gender = patient_profile.get('gender')
            location = patient_profile.get('location', {})
        
        if not conditions:
            return []  # No conditions to match
        
        # Check if we should use real API or samples (for testing)
        if not use_real_api:
            # Use sample data for testing
            if 'umls_concepts' in patient_profile and self.umls_client:
                return self._match_with_umls_concepts_sample(patient_profile, max_results)
            else:
                return self._match_with_structured_data_sample(patient_profile, max_results)
            
        # Use real-time API - fetch trials based on patient conditions
        print("Fetching trials from ClinicalTrials.gov API...")
        
        # Format gender for API
        api_gender = None
        if gender == 'M':
            api_gender = 'Male'
        elif gender == 'F':
            api_gender = 'Female'
        
        # Get country from location if available
        country = location.get('country') if location else None
        
        # For cancer conditions, add more specific search terms
        search_conditions = conditions.copy()
        
        # Make the API call to get real trials
        trials = self.trials_client.search_trials(
            conditions=search_conditions,
            max_results=max_results*2,  # Get more results than needed to allow for scoring
            gender=api_gender,
            min_age=age,
            max_age=age,
            status="Recruiting",
            country=country
        )
        
        # Use UMLS enhanced matching if available
        if 'umls_concepts' in patient_profile and self.umls_client:
            return self._match_real_trials_with_umls(patient_profile, trials, max_results)
        else:
            return self._match_real_trials_structured(patient_profile, trials, max_results)
    
    def match_patient_to_trials_fhir(self, fhir_bundle, max_results=10, use_real_api=True):
        """
        Match patient using FHIR Bundle format
        
        Args:
            fhir_bundle: FHIR Bundle containing patient data
            max_results: Maximum number of trials to return
            use_real_api: Whether to use real-time API
            
        Returns:
            List of matching trials
        """
        # Extract patient profile from FHIR Bundle
        patient_profile = self._extract_profile_from_fhir(fhir_bundle)
        
        # Use standard matching with extracted profile
        return self.match_patient_to_trials(patient_profile, max_results, use_real_api)
    
    def _extract_profile_from_fhir(self, fhir_bundle):
        """Extract patient profile from FHIR Bundle"""
        profile = {
            'patient_id': None,
            'structured_data': {
                'age': None,
                'gender': None,
                'conditions': [],
                'medications': [],
                'cancer_details': {}
            },
            'umls_concepts': {
                'conditions': [],
                'medications': []
            }
        }
        
        # Process bundle entries
        for entry in fhir_bundle.get('entry', []):
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')
            
            if resource_type == 'Patient':
                profile['patient_id'] = resource.get('id')
                profile['structured_data']['gender'] = resource.get('gender', '').upper()[0] if resource.get('gender') else None
                
                # Extract age from birthDate
                birth_date = resource.get('birthDate')
                if birth_date:
                    from datetime import datetime
                    birth_year = int(birth_date.split('-')[0])
                    profile['structured_data']['age'] = datetime.now().year - birth_year
            
            elif resource_type == 'Condition':
                condition_text = resource.get('code', {}).get('text', '')
                if condition_text and condition_text not in profile['structured_data']['conditions']:
                    profile['structured_data']['conditions'].append(condition_text)
                
                # Extract UMLS CUI if available
                codings = resource.get('code', {}).get('coding', [])
                for coding in codings:
                    if coding.get('system') == 'http://www.nlm.nih.gov/research/umls':
                        profile['umls_concepts']['conditions'].append({
                            'cui': coding.get('code'),
                            'preferred_name': coding.get('display'),
                            'original_term': condition_text
                        })
            
            elif resource_type == 'MedicationStatement':
                med_text = resource.get('medicationCodeableConcept', {}).get('text', '')
                if med_text and med_text not in profile['structured_data']['medications']:
                    profile['structured_data']['medications'].append(med_text)
            
            elif resource_type == 'Observation':
                # Extract biomarker observations
                code = resource.get('code', {}).get('text', '')
                value = resource.get('valueCodeableConcept', {}).get('text', '')
                
                if 'HER2' in code:
                    profile['structured_data']['cancer_details']['HER2_status'] = value.lower()
                elif 'ER' in code or 'Estrogen' in code:
                    profile['structured_data']['cancer_details']['ER_status'] = value.lower()
                elif 'PR' in code or 'Progesterone' in code:
                    profile['structured_data']['cancer_details']['PR_status'] = value.lower()
        
        return profile
    
    def _match_real_trials_with_umls(self, patient_profile, trials, max_results):
        """Match using UMLS concept mappings with real trial data"""
        matches = []
        
        # Extract condition CUIs
        patient_condition_cuis = [concept['cui'] for concept in 
                                patient_profile['umls_concepts']['conditions']]
        
        # Handle different profile structures
        if 'structured_data' in patient_profile:
            # Profile created from raw text
            patient_conditions = patient_profile['structured_data']['conditions']
            patient_age = patient_profile['structured_data'].get('age')
            patient_gender = patient_profile['structured_data'].get('gender')
            cancer_details = patient_profile['structured_data'].get('cancer_details', {})
        else:
            # Enhanced existing profile
            patient_conditions = patient_profile.get('conditions', [])
            patient_age = patient_profile.get('age')
            patient_gender = patient_profile.get('gender')
            cancer_details = patient_profile.get('cancer_details', {})
        
        # Pre-calculate semantic types for patient conditions for better matching
        patient_semantics = set()
        if 'umls_concepts' in patient_profile:
            for concept in patient_profile['umls_concepts']['conditions']:
                patient_semantics.update(concept.get('semantic_types', []))
        
        for trial in trials:
            match_score = 0
            match_reasons = {}
            
            # Enhanced condition matching using UMLS CUIs
            trial_conditions = trial.get('conditions', [])
            condition_matches = []
            
            # Map trial conditions to UMLS if not already done
            if self.umls_client:
                try:
                    trial_umls_conditions = self.umls_client.map_medical_terms(trial_conditions)
                    trial_cuis = [c['cui'] for c in trial_umls_conditions]
                    
                    # Direct CUI matching (strongest signal)
                    cui_matches = []
                    for patient_cui in patient_condition_cuis:
                        if any(patient_cui == trial_cui for trial_cui in trial_cuis):
                            cui_matches.append(patient_cui)
                            match_score += 40  # Higher score for exact CUI match
                    
                    if cui_matches:
                        match_reasons['cui_matches'] = cui_matches
                    
                    # Semantic type matching (helps with related conditions)
                    if patient_semantics:
                        trial_semantics = set()
                        for concept in trial_umls_conditions:
                            trial_semantics.update(concept.get('semantic_types', []))
                        
                        semantic_matches = list(patient_semantics.intersection(trial_semantics))
                        if semantic_matches:
                            match_score += 15 * len(semantic_matches)
                            match_reasons['semantic_matches'] = semantic_matches
                            
                except Exception as e:
                    print(f"Error in UMLS mapping: {str(e)}", file=sys.stderr)
            
            # String-based condition matching (fallback)
            for patient_condition in patient_conditions:
                for trial_condition in trial_conditions:
                    if patient_condition.lower() in trial_condition.lower() or trial_condition.lower() in patient_condition.lower():
                        condition_matches.append(f"{patient_condition} -> {trial_condition}")
                        match_score += 30
            
            if condition_matches:
                match_reasons['condition_matches'] = condition_matches
            
            # Age matching
            if patient_age and 'age_range' in trial:
                if trial['age_range']['min'] <= patient_age <= trial['age_range']['max']:
                    match_score += 20
                    match_reasons['age_match'] = f"Age {patient_age} fits range {trial['age_range']['min']}-{trial['age_range']['max']}"
            
            # Gender matching
            if patient_gender and 'gender' in trial:
                if trial['gender'] == 'Both' or trial['gender'] == patient_gender:
                    match_score += 10
                    match_reasons['gender_match'] = f"Gender {patient_gender} accepted"
            
            # Location matching
            if patient_profile.get('location') and trial.get('locations'):
                location_match = self._check_location_match(
                    patient_profile.get('location'), 
                    trial.get('locations')
                )
                if location_match:
                    match_score += 15
                    match_reasons['location_match'] = location_match
            
            if match_score > 0:
                matches.append({
                    'trial': trial,
                    'score': match_score,
                    'match_reasons': match_reasons
                })
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:max_results]
    
    def _match_real_trials_structured(self, patient_profile, trials, max_results):
        """Match real trials using structured data without UMLS"""
        matches = []
        
        # Extract patient data from the appropriate structure
        if 'structured_data' in patient_profile:
            # Profile created from raw text
            patient_conditions = patient_profile['structured_data'].get('conditions', [])
            patient_age = patient_profile['structured_data'].get('age')
            patient_gender = patient_profile['structured_data'].get('gender')
            patient_location = patient_profile['structured_data'].get('location', {})
        else:
            # Traditional profile structure
            patient_conditions = patient_profile.get('conditions', [])
            patient_age = patient_profile.get('age')
            patient_gender = patient_profile.get('gender')
            patient_location = patient_profile.get('location', {})
        
        for trial in trials:
            match_score = 0
            match_reasons = {}
            
            # Condition matching
            trial_conditions = trial.get('conditions', [])
            condition_matches = []
            
            for patient_condition in patient_conditions:
                for trial_condition in trial_conditions:
                    # Check both directions - trial condition could be more specific
                    if patient_condition.lower() in trial_condition.lower() or trial_condition.lower() in patient_condition.lower():
                        condition_matches.append(f"{patient_condition} -> {trial_condition}")
                        match_score += 30
            
            if condition_matches:
                match_reasons['condition_matches'] = condition_matches
            
            # Age matching
            if patient_age and 'age_range' in trial:
                if trial['age_range']['min'] <= patient_age <= trial['age_range']['max']:
                    match_score += 20
                    match_reasons['age_match'] = f"Age {patient_age} fits range {trial['age_range']['min_text']} - {trial['age_range']['max_text']}"
            
            # Gender matching
            if patient_gender and trial.get('gender'):
                if trial['gender'] == 'Both' or trial['gender'] == patient_gender:
                    match_score += 10
                    match_reasons['gender_match'] = f"Gender {patient_gender} accepted"
            
            # Location matching
            if patient_location and trial.get('locations'):
                location_match = self._check_location_match(patient_location, trial.get('locations'))
                if location_match:
                    match_score += 15
                    match_reasons['location_match'] = location_match
            
            # Phase matching for cancer - prefer later phase trials
            if any('cancer' in cond.lower() for cond in patient_conditions):
                phase = trial.get('phase', '')
                if 'Phase 3' in phase:
                    match_score += 10
                    match_reasons['phase_match'] = f"Late-stage trial: {phase}"
                elif 'Phase 2' in phase:
                    match_score += 5
                    match_reasons['phase_match'] = f"Mid-stage trial: {phase}"
            
            if match_score > 0:
                matches.append({
                    'trial': trial,
                    'score': match_score,
                    'match_reasons': match_reasons
                })
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:max_results]
    
    def _check_location_match(self, patient_location, trial_locations):
        """Check if patient location matches any trial location"""
        patient_country = (patient_location.get('country', '') or '').lower()
        patient_state = (patient_location.get('state', '') or '').lower()
        patient_city = (patient_location.get('city', '') or '').lower()
        
        matching_locations = []
        
        for loc in trial_locations:
            loc_country = (loc.get('country', '') or '').lower()
            loc_state = (loc.get('state', '') or '').lower()
            loc_city = (loc.get('city', '') or '').lower()
            
            # Country match is required
            if patient_country and loc_country and patient_country in loc_country:
                # Check for state match if both available
                if patient_state and loc_state and patient_state in loc_state:
                    # Check for city match if both available
                    if patient_city and loc_city and patient_city in loc_city:
                        matching_locations.append(f"{loc_city.title()}, {loc_state.title()}, {loc_country.title()}")
                    else:
                        matching_locations.append(f"{loc_state.title()}, {loc_country.title()}")
                else:
                    matching_locations.append(loc_country.title())
        
        return matching_locations if matching_locations else None
    
    # Keep sample trial matching methods for fallback/testing
    def _match_with_umls_concepts_sample(self, patient_profile, max_results):
        """Match using UMLS concept mappings for better accuracy (with sample data)"""
        matches = []
        trials = self._get_sample_trials()
        
        # Extract primary patient characteristics
        conditions = patient_profile.get("conditions", [])
        age = patient_profile.get("age")
        gender = patient_profile.get("gender", "").upper() if patient_profile.get("gender") else ""
        location = patient_profile.get("location", {})
        
        if not conditions:
            return []  # No conditions to match
            
        # Enhanced matching using UMLS concepts if available
        # Check if patient has UMLS concepts
        if 'umls_concepts' in patient_profile and self.umls_client:
            return self._match_with_umls_concepts(patient_profile, max_results)
        else:
            return self._match_with_structured_data(patient_profile, max_results)
    
    def _match_with_structured_data_sample(self, patient_profile, max_results):
        """Original matching for structured data without UMLS (with sample data)"""
        matches = []
        trials = self._get_sample_trials()
        
        # Extract patient data from the appropriate structure
        if 'structured_data' in patient_profile:
            # Profile created from raw text
            patient_conditions = patient_profile['structured_data'].get('conditions', [])
            patient_age = patient_profile['structured_data'].get('age')
            patient_gender = patient_profile['structured_data'].get('gender')
            patient_location = patient_profile['structured_data'].get('location', {})
        else:
            # Traditional profile structure
            patient_conditions = patient_profile.get('conditions', [])
            patient_age = patient_profile.get('age')
            patient_gender = patient_profile.get('gender')
            patient_location = patient_profile.get('location', {})
        
        for trial in trials:
            match_score = 0
            match_reasons = {}
            
            # Condition matching
            trial_conditions = trial.get('conditions', [])
            condition_matches = []
            
            for patient_condition in patient_conditions:
                for trial_condition in trial_conditions:
                    if patient_condition.lower() in trial_condition.lower():
                        condition_matches.append(f"{patient_condition} -> {trial_condition}")
                        match_score += 30
            
            if condition_matches:
                match_reasons['condition_matches'] = condition_matches
            
            # Age matching
            if patient_age and 'age_range' in trial:
                if trial['age_range']['min'] <= patient_age <= trial['age_range']['max']:
                    match_score += 20
                    match_reasons['age_match'] = f"Age {patient_age} fits range"
            
            # Gender matching
            if patient_gender and 'gender' in trial:
                if trial['gender'] == 'Both' or trial['gender'] == patient_gender:
                    match_score += 10
                    match_reasons['gender_match'] = f"Gender {patient_gender} accepted"
            
            if match_score > 0:
                matches.append({
                    'trial': trial,
                    'score': match_score,
                    'match_reasons': match_reasons
                })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:max_results]
    
    def _conditions_match_umls(self, patient_condition, trial_condition, patient_cuis):
        """Enhanced condition matching using UMLS concepts"""
        # Direct string match
        if patient_condition.lower() in trial_condition.lower():
            return True
        
        # UMLS concept matching (simplified)
        condition_mappings = {
            'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'glucose'],
            'type 2 diabetes': ['type 2', 'diabetes mellitus', 'non-insulin dependent'],
            'hypertension': ['hypertension', 'high blood pressure', 'elevated bp', 'htn'],
            'heart disease': ['cardiac', 'cardiovascular', 'heart', 'coronary'],
            'depression': ['depression', 'depressive', 'mood disorder', 'mental health']
        }
        
        patient_lower = patient_condition.lower()
        trial_lower = trial_condition.lower()
        
        for base_condition, variations in condition_mappings.items():
            if patient_lower in variations or patient_lower == base_condition:
                if any(var in trial_lower for var in variations):
                    return True
        
        return False
    
    def _get_sample_trials(self):
        """Get sample clinical trials for testing"""
        return [
            {
                'nct_id': 'NCT12345001',
                'title': 'A Phase III Study of Metformin in Type 2 Diabetes Patients',
                'conditions': ['Type 2 Diabetes Mellitus', 'Hyperglycemia'],
                'age_range': {'min': 18, 'max': 75},
                'gender': 'Both',
                'location': ['California', 'Texas', 'New York'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345002', 
                'title': 'Cardiovascular Outcomes in Diabetic Patients with Hypertension',
                'conditions': ['Diabetes Mellitus', 'Hypertensive Disease', 'Cardiovascular Risk'],
                'age_range': {'min': 45, 'max': 80},
                'gender': 'Both',
                'location': ['California', 'Massachusetts', 'Florida'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345003',
                'title': 'Depression and Diabetes: A Behavioral Intervention Study',
                'conditions': ['Major Depressive Disorder', 'Type 2 Diabetes'],
                'age_range': {'min': 21, 'max': 65},
                'gender': 'Both',
                'location': ['California', 'Illinois', 'Massachusetts'],
                'status': 'Active'
            },
            {
                'nct_id': 'NCT12345004',
                'title': 'Insulin Therapy Optimization in Advanced Diabetes',
                'conditions': ['Type 1 Diabetes', 'Type 2 Diabetes', 'Insulin Resistance'],
                'age_range': {'min': 25, 'max': 70},
                'gender': 'Both', 
                'location': ['Massachusetts', 'New York', 'Pennsylvania'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345005',
                'title': 'Cardiac Health in Diabetic Women: Prevention Study',
                'conditions': ['Cardiovascular Disease', 'Diabetes Mellitus', 'Heart Disease'],
                'age_range': {'min': 30, 'max': 65},
                'gender': 'F',
                'location': ['California', 'Texas', 'Florida'],
                'status': 'Recruiting'
            },
            # Add breast cancer specific trials
            {
                'nct_id': 'NCT12345006',
                'title': 'Phase II Study of Adjuvant Therapy for HER2-negative Breast Cancer',
                'conditions': ['Breast Cancer', 'HER2-negative Breast Carcinoma', 'Stage II Breast Cancer'],
                'age_range': {'min': 18, 'max': 75},
                'gender': 'F',
                'location': ['Illinois', 'New York', 'California'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345007',
                'title': 'Combination Immunotherapy for Advanced Breast Cancer',
                'conditions': ['Breast Cancer', 'Metastatic Breast Cancer', 'Advanced Breast Cancer'],
                'age_range': {'min': 30, 'max': 80},
                'gender': 'F',
                'location': ['Texas', 'Florida', 'Illinois', 'Washington'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345008',
                'title': 'Novel Chemotherapy Regimen for Early-Stage Breast Cancer',
                'conditions': ['Breast Cancer', 'Stage I Breast Cancer', 'Stage II Breast Cancer'],
                'age_range': {'min': 18, 'max': 65},
                'gender': 'Both',  # Male breast cancer can occur
                'location': ['Illinois', 'Minnesota', 'Wisconsin'],
                'status': 'Recruiting'
            },
            {
                'nct_id': 'NCT12345009',
                'title': 'International Breast Cancer Treatment Study',
                'conditions': ['Breast Cancer', 'Invasive Breast Carcinoma'],
                'age_range': {'min': 20, 'max': 85},
                'gender': 'F',
                'location': ['California', 'New York', 'Texas', 'India', 'United Kingdom'],
                'status': 'Recruiting'
            }
        ]
    
    def _get_match_reasons(self, trial: Dict, patient: Dict) -> Dict:
        """Get detailed reasons for the match"""
        reasons = {}
        
        # Condition match reason
        patient_conditions = [c.lower() for c in patient.get("conditions", []) if c]
        trial_conditions = [c.lower() for c in trial.get("conditions", []) if c]
        
        matching_conditions = []
        for p_cond in patient_conditions:
            for t_cond in trial_conditions:
                if t_cond and p_cond and p_cond in t_cond:
                    matching_conditions.append(t_cond)
        
        if matching_conditions:
            reasons["conditions"] = matching_conditions
            
        # Age eligibility reason
        patient_age = patient.get("age")
        min_age_str = trial.get("min_age", "0 Years")
        max_age_str = trial.get("max_age", "120 Years")
        
        if patient_age is not None:
            reasons["age"] = f"Patient age {patient_age} is within trial criteria ({min_age_str} - {max_age_str})"
            
        # Gender match reason
        patient_gender = (patient.get("gender", "") or "").upper()
        trial_gender = trial.get("gender", "ALL") or "ALL"
        
        if trial_gender == "ALL":
            reasons["gender"] = "Trial accepts all genders"
        elif patient_gender == trial_gender:
            reasons["gender"] = f"Gender match: {patient_gender}"
            
        # Location match reason
        patient_location = patient.get("location", {})
        trial_locations = trial.get("locations", [])
        
        matching_locations = []
        if patient_location and trial_locations:
            patient_country = (patient_location.get("country", "") or "").lower()
            patient_state = (patient_location.get("state", "") or "").lower()
            
            for loc in trial_locations:
                loc_country = (loc.get("country", "") or "").lower()
                loc_state = (loc.get("state", "") or "").lower()
                
                if patient_country and loc_country and patient_country == loc_country:
                    if patient_state and loc_state and patient_state == loc_state:
                        facility = loc.get("facility") or "Unknown Facility"
                        city = loc.get("city") or "Unknown City"
                        state = loc.get("state") or "Unknown State"
                        matching_locations.append(f"{facility} ({city}, {state})")
                    else:
                        facility = loc.get("facility") or "Unknown Facility" 
                        city = loc.get("city") or "Unknown City"
                        country = loc.get("country") or "Unknown Country"
                        matching_locations.append(f"{facility} ({city}, {country})")
        
        if matching_locations:
            reasons["locations"] = matching_locations
            
        return reasons

    def _calculate_advanced_match_score(self, patient_profile, trial, umls_concepts):
        """Advanced scoring with weighted importance and medical logic"""
        base_score = 0
        match_reasons = {}
        
        # Weighted scoring system
        weights = {
            'primary_condition_exact': 50,    # Exact condition match
            'primary_condition_related': 30,   # Related via UMLS
            'biomarker_match': 40,            # Critical for targeted therapy
            'age_eligibility': 25,            # Must meet to be eligible
            'gender_eligibility': 20,         # Must meet to be eligible
            'phase_appropriateness': 15,      # Higher phases better for treatment
            'location_accessibility': 10,      # Geographic feasibility
            'trial_status': 35               # Must be recruiting
        }
        
        # Primary condition matching with medical logic
        patient_conditions = patient_profile['structured_data'].get('conditions', [])
        trial_conditions = trial.get('conditions', [])
        
        # Direct condition matching
        for p_condition in patient_conditions:
            for t_condition in trial_conditions:
                similarity = self._calculate_condition_similarity(p_condition, t_condition, umls_concepts)
                if similarity > 0.8:  # High similarity threshold
                    base_score += weights['primary_condition_exact']
                    match_reasons['primary_condition'] = f"{p_condition} matches {t_condition}"
                    break
        
        # Biomarker matching (critical for cancer trials)
        cancer_details = patient_profile['structured_data'].get('cancer_details', {})
        if cancer_details and 'cancer' in ' '.join(patient_conditions).lower():
            biomarker_score, biomarker_reasons = self._match_biomarkers(cancer_details, trial)
            base_score += biomarker_score
            if biomarker_reasons:
                match_reasons.update(biomarker_reasons)
        
        # Phase appropriateness with medical logic
        phase = trial.get('phase', '')
        patient_stage = cancer_details.get('stage', '').lower()
        
        if 'phase 3' in phase.lower():
            if 'stage iv' not in patient_stage and 'metastatic' not in patient_stage:
                base_score += weights['phase_appropriateness']
                match_reasons['phase_appropriate'] = "Phase 3 suitable for non-metastatic disease"
        elif 'phase 2' in phase.lower():
            base_score += weights['phase_appropriateness'] * 0.8  # Slightly lower preference
        elif 'phase 1' in phase.lower() and 'advanced' in patient_stage:
            base_score += weights['phase_appropriateness'] * 0.6  # Only for advanced disease
        
        # Strict eligibility requirements (must pass to be viable)
        eligibility_passed, eligibility_reasons = self._check_strict_eligibility(patient_profile, trial)
        if not eligibility_passed:
            base_score = max(0, base_score - 50)  # Heavy penalty for ineligibility
            match_reasons['eligibility_issues'] = eligibility_reasons
        
        return min(100, base_score), match_reasons

    def _match_biomarkers(self, cancer_details, trial):
        """Match cancer biomarkers with trial requirements"""
        score = 0
        reasons = {}
        
        # Extract biomarker requirements from trial title/description
        trial_text = (trial.get('title', '') + ' ' + trial.get('summary', '')).lower()
        
        biomarker_matches = []
        if cancer_details.get('HER2_status') == 'positive' and 'her2' in trial_text and 'positive' in trial_text:
            score += 40
            biomarker_matches.append("HER2+ status matches trial requirement")
        
        if cancer_details.get('ER_status') == 'negative' and 'triple negative' in trial_text:
            score += 35
            biomarker_matches.append("Triple negative status matches trial focus")
        
        if biomarker_matches:
            reasons['biomarker_matches'] = biomarker_matches
        
        return score, reasons

    def _check_strict_eligibility(self, patient_profile, trial):
        """Check absolute eligibility requirements"""
        reasons = []
        passed = True
        
        # Age requirements (strict)
        patient_age = patient_profile['structured_data'].get('age')
        age_range = trial.get('age_range', {})
        
        if patient_age and age_range:
            if patient_age < age_range.get('min', 0) or patient_age > age_range.get('max', 150):
                passed = False
                reasons.append(f"Age {patient_age} outside trial range {age_range.get('min')}-{age_range.get('max')}")
        
        # Gender requirements (strict)
        patient_gender = patient_profile['structured_data'].get('gender')
        trial_gender = trial.get('gender')
        
        if patient_gender and trial_gender and trial_gender not in ['Both', patient_gender]:
            passed = False
            reasons.append(f"Gender {patient_gender} not accepted (trial requires {trial_gender})")
        
        return passed, reasons

    def _calculate_condition_similarity(self, patient_condition, trial_condition, umls_concepts):
        """Calculate similarity between patient and trial conditions"""
        # Direct string match
        if patient_condition.lower() in trial_condition.lower() or trial_condition.lower() in patient_condition.lower():
            return 1.0
        
        # Check UMLS concept similarity if available
        if 'conditions' in umls_concepts:
            for concept in umls_concepts['conditions']:
                if concept['original_term'].lower() == patient_condition.lower():
                    preferred_name = concept.get('preferred_name', '').lower()
                    if preferred_name in trial_condition.lower() or trial_condition.lower() in preferred_name:
                        return 0.9
        
        # Word overlap similarity
        patient_words = set(patient_condition.lower().split())
        trial_words = set(trial_condition.lower().split())
        
        if patient_words & trial_words:
            overlap = len(patient_words & trial_words)
            total = len(patient_words | trial_words)
            return overlap / total
        
        return 0.0

# Example patient data format
example_patient = {
    "patient_id": "P12345",
    "name": "John Doe",
    "age": 65,
    "gender": "M",
    "conditions": ["Type 2 Diabetes", "Hypertension"],
    "medications": ["Metformin", "Lisinopril"],
    "location": {
        "city": "San Diego",
        "state": "California",
        "country": "United States"
    }
}
