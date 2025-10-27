"""
Enhanced Patient Profile Module
Provides structured patient data representation for clinical trial matching
"""

class EnhancedPatientProfile:
    """
    Represents a patient's medical profile with structured data
    for clinical trial matching
    """
    
    def __init__(self, patient_id, raw_text=None):
        self.patient_id = patient_id
        self.raw_text = raw_text
        self.structured_data = {
            'age': None,
            'gender': None,
            'conditions': [],
            'medications': [],
            'procedures': [],
            'cancer_details': {},
            'biomarkers': {},
            'detailed_medications': []
        }
        self.umls_concepts = {
            'conditions': [],
            'medications': [],
            'procedures': []
        }
    
    def add_condition(self, condition):
        """Add a medical condition"""
        if condition and condition not in self.structured_data['conditions']:
            self.structured_data['conditions'].append(condition)
    
    def add_medication(self, medication):
        """Add a medication"""
        if medication and medication not in self.structured_data['medications']:
            self.structured_data['medications'].append(medication)
    
    def add_procedure(self, procedure):
        """Add a medical procedure"""
        if procedure and procedure not in self.structured_data['procedures']:
            self.structured_data['procedures'].append(procedure)
    
    def set_demographics(self, age=None, gender=None):
        """Set patient demographics"""
        if age is not None:
            self.structured_data['age'] = age
        if gender is not None:
            self.structured_data['gender'] = gender
    
    def set_cancer_details(self, **kwargs):
        """Set cancer-specific details (stage, biomarkers, etc.)"""
        self.structured_data['cancer_details'].update(kwargs)
    
    def add_umls_concept(self, category, concept):
        """Add a UMLS concept mapping"""
        if category in self.umls_concepts:
            self.umls_concepts[category].append(concept)
    
    def to_dict(self):
        """Convert profile to dictionary format"""
        return {
            'patient_id': self.patient_id,
            'raw_text': self.raw_text,
            'structured_data': self.structured_data,
            'umls_concepts': self.umls_concepts
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create profile from dictionary"""
        profile = cls(data['patient_id'], data.get('raw_text'))
        profile.structured_data = data.get('structured_data', profile.structured_data)
        profile.umls_concepts = data.get('umls_concepts', profile.umls_concepts)
        return profile
    
    def __repr__(self):
        return f"EnhancedPatientProfile(patient_id='{self.patient_id}', conditions={len(self.structured_data['conditions'])})"
