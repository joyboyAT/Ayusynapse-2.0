import re
from typing import List, Dict, Any

class MedicalTextProcessor:
    """Process raw medical text to extract medical entities and terms"""
    
    def __init__(self):
        # Common medical term patterns and keywords
        self.condition_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'blood sugar', 'glucose', 'a1c'],
            'type 2 diabetes': ['type 2', 'type ii', 't2dm', 'adult onset', 'non-insulin dependent'],
            'hypertension': ['hypertension', 'high blood pressure', 'elevated bp', 'htn', 'elevated blood pressure'],
            'cancer': ['cancer', 'tumor', 'malignancy', 'neoplasm', 'carcinoma', 'sarcoma', 'metastasis', 'oncology'],
            'breast_cancer': ['breast cancer', 'breast carcinoma', 'breast tumor', 'breast malignancy', 'her2', 'estrogen receptor', 'progesterone receptor'],
            'lung_cancer': ['lung cancer', 'lung carcinoma', 'lung tumor', 'nsclc', 'sclc', 'non-small cell', 'small cell lung'],
            'prostate_cancer': ['prostate cancer', 'prostate tumor', 'prostate carcinoma', 'psa elevated'],
            'colon_cancer': ['colon cancer', 'colorectal', 'rectal cancer', 'bowel cancer'],
            'heart_disease': ['heart disease', 'cardiac', 'coronary', 'myocardial', 'cardiovascular', 'angina', 'chf'],
            'asthma': ['asthma', 'bronchial', 'wheezing', 'respiratory', 'inhaler'],
            'copd': ['copd', 'chronic obstructive', 'emphysema', 'chronic bronchitis'],
            'depression': ['depression', 'depressive', 'mood disorder', 'mental health', 'psychiatric'],
            'anxiety': ['anxiety', 'anxious', 'panic', 'generalized anxiety', 'gad'],
            'alzheimer': ['alzheimer', 'dementia', 'memory loss', 'cognitive decline'],
            'arthritis': ['arthritis', 'joint pain', 'rheumatoid', 'osteoarthritis', 'inflammatory joint'],
            'osteoporosis': ['osteoporosis', 'bone density', 'fragile bones', 'bone loss'],
            'obesity': ['obesity', 'obese', 'overweight', 'bmi', 'weight management'],
            'kidney_disease': ['kidney disease', 'renal', 'dialysis', 'ckd', 'renal failure'],
            'liver_disease': ['liver disease', 'hepatic', 'cirrhosis', 'fatty liver', 'hepatitis']
        }
        
        self.medication_keywords = {
            'metformin': ['metformin', 'glucophage', 'glycon'],
            'insulin': ['insulin', 'lantus', 'novolin', 'humalog', 'humulin'],
            'lisinopril': ['lisinopril', 'prinivil', 'zestril', 'ace inhibitor'],
            'atorvastatin': ['atorvastatin', 'lipitor', 'statin'],
            'simvastatin': ['simvastatin', 'zocor'],
            'amlodipine': ['amlodipine', 'norvasc', 'calcium channel blocker'],
            'metoprolol': ['metoprolol', 'lopressor', 'toprol', 'beta blocker'],
            'omeprazole': ['omeprazole', 'prilosec', 'ppi', 'proton pump inhibitor'],
            'albuterol': ['albuterol', 'ventolin', 'proventil', 'inhaler'],
            'prednisone': ['prednisone', 'steroid', 'corticosteroid'],
            'gabapentin': ['gabapentin', 'neurontin'],
            'hydrochlorothiazide': ['hydrochlorothiazide', 'hctz', 'diuretic', 'water pill'],
            'levothyroxine': ['levothyroxine', 'synthroid', 'thyroid medication'],
            'acetaminophen': ['acetaminophen', 'tylenol', 'paracetamol'],
            'ibuprofen': ['ibuprofen', 'advil', 'motrin', 'nsaid'],
            'aspirin': ['aspirin', 'asa', 'baby aspirin'],
            'fluoxetine': ['fluoxetine', 'prozac', 'ssri', 'antidepressant'],
            'sertraline': ['sertraline', 'zoloft', 'ssri'],
            'losartan': ['losartan', 'cozaar', 'arb'],
            'furosemide': ['furosemide', 'lasix', 'loop diuretic'],
            'chemotherapy': ['chemotherapy', 'chemo', 'cytotoxic', 'adriamycin', 'cisplatin', 'taxol', 'paclitaxel', 'docetaxel', 'doxorubicin'],
            'radiation': ['radiation', 'radiotherapy', 'xrt', 'external beam', 'brachytherapy'],
            'immunotherapy': ['immunotherapy', 'keytruda', 'opdivo', 'yervoy', 'pembrolizumab', 'nivolumab']
        }
        
        # Add cancer stage keywords
        self.cancer_stages = {
            'stage_i': ['stage 1', 'stage i', 'stage one'],
            'stage_ii': ['stage 2', 'stage ii', 'stage two'],
            'stage_iii': ['stage 3', 'stage iii', 'stage three'],
            'stage_iv': ['stage 4', 'stage iv', 'stage four', 'metastatic', 'advanced', 'metastasis']
        }
        
        # Add cancer biomarkers
        self.cancer_biomarkers = {
            'her2_pos': ['her2 positive', 'her2+', 'her2 overexpression'],
            'her2_neg': ['her2 negative', 'her2-'],
            'er_pos': ['estrogen receptor positive', 'er positive', 'er+'],
            'er_neg': ['estrogen receptor negative', 'er negative', 'er-'],
            'pr_pos': ['progesterone receptor positive', 'pr positive', 'pr+'],
            'pr_neg': ['progesterone receptor negative', 'pr negative', 'pr-'],
            'triple_neg': ['triple negative', 'tnbc'],
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, Any]:
        """Extract medical entities from raw text"""
        text_lower = text.lower()
        
        entities = {
            'conditions': [],
            'medications': [],
            'symptoms': [],
            'procedures': [],
            'cancer_details': {}
        }
        
        # Extract conditions
        for condition, keywords in self.condition_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    condition_name = condition.replace('_', ' ').title()
                    if condition_name not in entities['conditions']:
                        entities['conditions'].append(condition_name)
                    break
        
        # Extract medications
        for medication, keywords in self.medication_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    medication_name = medication.title()
                    if medication_name not in entities['medications']:
                        entities['medications'].append(medication_name)
                    break
        
        # Extract cancer stage if cancer is detected
        if any('cancer' in cond.lower() for cond in entities['conditions']):
            for stage, patterns in self.cancer_stages.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        entities['cancer_details']['stage'] = stage.replace('_', ' ').upper()
                        break
            
            # Extract cancer biomarkers
            for marker, patterns in self.cancer_biomarkers.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        entities['cancer_details'][marker.split('_')[0]] = marker.split('_')[1]
                        break
        
        # Extract age
        age_patterns = [
            r'(\d+)[- ]years?[- ]old',
            r'(\d+)[- ]yo\b',
            r'\bage[: ]+(\d+)\b',
            r'\b(\d+)[- ]year[- ]old\b'
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, text_lower)
            if age_match:
                entities['age'] = int(age_match.group(1))
                break
        
        # Improved gender extraction
        female_indicators = ['female', 'woman', 'women', 'girl', ' she ', ' her ', 'herself', 'lady']
        male_indicators = ['male', 'man', 'men', 'boy', ' he ', ' his ', 'himself', 'gentleman'] 
        
        # Check for explicit gender mentions with word boundaries
        female_patterns = [r'\b' + re.escape(term) + r'\b' for term in female_indicators]
        male_patterns = [r'\b' + re.escape(term) + r'\b' for term in male_indicators]
        
        if any(re.search(pattern, text_lower) for pattern in female_patterns):
            entities['gender'] = 'F'
        elif any(re.search(pattern, text_lower) for pattern in male_patterns):
            entities['gender'] = 'M'
        
        # Special handling for breast cancer - typically female but not always
        if 'breast cancer' in text_lower and 'gender' not in entities:
            entities['gender'] = 'F'  # Default to female for breast cancer if gender not specified
        
        return entities
    
    def tokenize_medical_text(self, text: str) -> List[str]:
        """Tokenize text and identify medical terms"""
        # Simple tokenization - can be enhanced with spaCy or NLTK
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        medical_tokens = []
        for token in tokens:
            if self._is_medical_term(token):
                medical_tokens.append(token)
        
        return medical_tokens
    
    def _is_medical_term(self, token: str) -> bool:
        """Check if a token is likely a medical term"""
        # Check in condition keywords
        for keywords in self.condition_keywords.values():
            if token in keywords:
                return True
        
        # Check in medication keywords
        for keywords in self.medication_keywords.values():
            if token in keywords:
                return True
        
        return False
    
    def extract_basic_demographics(self, text: str) -> Dict[str, Any]:
        """Extract basic demographic information from text"""
        demographics = {}
        text_lower = text.lower()
        
        # Age extraction
        age_patterns = [
            r'(\d+)[- ]years?[- ]old',
            r'(\d+)[- ]yo\b',
            r'\bage[: ]+(\d+)\b',
            r'\b(\d+)[- ]year[- ]old\b'
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, text_lower)
            if age_match:
                demographics['age'] = int(age_match.group(1))
                break
        
        # Gender extraction
        gender_patterns = {
            'M': [r'\bmale\b', r'\bman\b', r'\bboy\b', r'\bhis\b', r'\bhe\b', r'\bhim\b'],
            'F': [r'\bfemale\b', r'\bwoman\b', r'\bgirl\b', r'\bher\b', r'\bshe\b']
        }
        
        for gender, patterns in gender_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                demographics['gender'] = gender
                break
        
        return demographics
