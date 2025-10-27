"""
FHIR Converter Module
Converts enhanced patient profiles to HL7 FHIR R4 format for global interoperability
"""

from datetime import datetime
from typing import Dict, List, Any
import uuid


class FHIRConverter:
    """Convert patient profiles to HL7 FHIR R4 format"""
    
    def __init__(self):
        self.fhir_version = "4.0.1"  # FHIR R4
    
    def convert_to_fhir_bundle(self, patient_profile: Dict) -> Dict:
        """
        Convert enhanced patient profile to FHIR Bundle resource
        
        Args:
            patient_profile: Enhanced patient profile dictionary
            
        Returns:
            FHIR Bundle containing Patient, Condition, and MedicationStatement resources
        """
        bundle_entries = []
        
        # Create Patient resource
        patient_resource = self._create_patient_resource(patient_profile)
        bundle_entries.append({
            "fullUrl": f"urn:uuid:{patient_resource['id']}",
            "resource": patient_resource,
            "request": {
                "method": "POST",
                "url": "Patient"
            }
        })
        
        patient_reference = f"Patient/{patient_resource['id']}"
        
        # Create Condition resources
        conditions = self._get_conditions(patient_profile)
        for condition in conditions:
            condition_resource = self._create_condition_resource(condition, patient_reference, patient_profile)
            bundle_entries.append({
                "fullUrl": f"urn:uuid:{condition_resource['id']}",
                "resource": condition_resource,
                "request": {
                    "method": "POST",
                    "url": "Condition"
                }
            })
        
        # Create MedicationStatement resources
        medications = self._get_medications(patient_profile)
        for medication in medications:
            med_resource = self._create_medication_statement(medication, patient_reference)
            bundle_entries.append({
                "fullUrl": f"urn:uuid:{med_resource['id']}",
                "resource": med_resource,
                "request": {
                    "method": "POST",
                    "url": "MedicationStatement"
                }
            })
        
        # Create Observation resources for cancer biomarkers
        cancer_details = patient_profile.get('structured_data', {}).get('cancer_details', {})
        if cancer_details:
            biomarker_observations = self._create_biomarker_observations(cancer_details, patient_reference)
            for obs in biomarker_observations:
                bundle_entries.append({
                    "fullUrl": f"urn:uuid:{obs['id']}",
                    "resource": obs,
                    "request": {
                        "method": "POST",
                        "url": "Observation"
                    }
                })
        
        # Create FHIR Bundle
        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "transaction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": bundle_entries
        }
        
        return bundle
    
    def _create_patient_resource(self, profile: Dict) -> Dict:
        """Create FHIR Patient resource"""
        structured_data = profile.get('structured_data', {})
        
        patient_id = profile.get('patient_id', str(uuid.uuid4()))
        age = structured_data.get('age')
        gender = structured_data.get('gender')
        
        # Map gender to FHIR codes
        gender_map = {
            'M': 'male',
            'F': 'female',
            'Male': 'male',
            'Female': 'female'
        }
        
        fhir_gender = gender_map.get(gender, 'unknown') if gender else 'unknown'
        
        patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]
            },
            "identifier": [{
                "system": "http://ayusynapse.com/patient-id",
                "value": patient_id
            }],
            "gender": fhir_gender
        }
        
        # Add birth date if age is available
        if age:
            birth_year = datetime.now().year - age
            patient["birthDate"] = f"{birth_year}-01-01"
        
        return patient
    
    def _create_condition_resource(self, condition: str, patient_ref: str, profile: Dict) -> Dict:
        """Create FHIR Condition resource"""
        condition_id = str(uuid.uuid4())
        
        # Try to get UMLS concept for the condition
        umls_concepts = profile.get('umls_concepts', {}).get('conditions', [])
        code_info = self._get_code_for_condition(condition, umls_concepts)
        
        cancer_details = profile.get('structured_data', {}).get('cancer_details', {})
        
        fhir_condition = {
            "resourceType": "Condition",
            "id": condition_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Condition"]
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active"
                }]
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                    "display": "Confirmed"
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "encounter-diagnosis",
                    "display": "Encounter Diagnosis"
                }]
            }],
            "code": code_info,
            "subject": {
                "reference": patient_ref
            },
            "recordedDate": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add stage information for cancer conditions
        if 'cancer' in condition.lower() and cancer_details.get('stage'):
            fhir_condition["stage"] = [{
                "summary": {
                    "text": cancer_details['stage']
                },
                "type": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "258215001",
                        "display": "Cancer staging"
                    }]
                }
            }]
        
        return fhir_condition
    
    def _create_medication_statement(self, medication: str, patient_ref: str) -> Dict:
        """Create FHIR MedicationStatement resource"""
        med_id = str(uuid.uuid4())
        
        return {
            "resourceType": "MedicationStatement",
            "id": med_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/MedicationStatement"]
            },
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "display": medication
                }],
                "text": medication
            },
            "subject": {
                "reference": patient_ref
            },
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z"
        }
    
    def _create_biomarker_observations(self, cancer_details: Dict, patient_ref: str) -> List[Dict]:
        """Create FHIR Observation resources for cancer biomarkers"""
        observations = []
        
        # HER2 status
        if cancer_details.get('HER2_status'):
            obs = self._create_biomarker_observation(
                'HER2',
                cancer_details['HER2_status'],
                patient_ref,
                loinc_code='48676-1',
                loinc_display='HER2 [Interpretation] in Tissue'
            )
            observations.append(obs)
        
        # ER status
        if cancer_details.get('ER_status'):
            obs = self._create_biomarker_observation(
                'ER',
                cancer_details['ER_status'],
                patient_ref,
                loinc_code='16112-5',
                loinc_display='Estrogen receptor [Interpretation] in Tissue'
            )
            observations.append(obs)
        
        # PR status
        if cancer_details.get('PR_status'):
            obs = self._create_biomarker_observation(
                'PR',
                cancer_details['PR_status'],
                patient_ref,
                loinc_code='16113-3',
                loinc_display='Progesterone receptor [Interpretation] in Tissue'
            )
            observations.append(obs)
        
        return observations
    
    def _create_biomarker_observation(self, marker_name: str, status: str, 
                                     patient_ref: str, loinc_code: str, 
                                     loinc_display: str) -> Dict:
        """Create individual biomarker observation"""
        obs_id = str(uuid.uuid4())
        
        # Map status to SNOMED codes
        status_map = {
            'positive': {'code': '10828004', 'display': 'Positive'},
            'negative': {'code': '260385009', 'display': 'Negative'}
        }
        
        status_coding = status_map.get(status.lower(), {'code': '261665006', 'display': 'Unknown'})
        
        return {
            "resourceType": "Observation",
            "id": obs_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]
            },
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory",
                    "display": "Laboratory"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": loinc_display
                }],
                "text": f"{marker_name} Status"
            },
            "subject": {
                "reference": patient_ref
            },
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": status_coding['code'],
                    "display": status_coding['display']
                }],
                "text": status.capitalize()
            }
        }
    
    def _get_code_for_condition(self, condition: str, umls_concepts: List[Dict]) -> Dict:
        """Get coding information for a condition"""
        # Try to find UMLS/SNOMED code
        for concept in umls_concepts:
            if concept.get('original_term', '').lower() == condition.lower():
                cui = concept.get('cui')
                preferred_name = concept.get('preferred_name', condition)
                
                return {
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls",
                        "code": cui,
                        "display": preferred_name
                    }],
                    "text": condition
                }
        
        # Fallback to text-only
        return {
            "text": condition
        }
    
    def _get_conditions(self, profile: Dict) -> List[str]:
        """Extract conditions from profile"""
        return profile.get('structured_data', {}).get('conditions', [])
    
    def _get_medications(self, profile: Dict) -> List[str]:
        """Extract medications from profile"""
        return profile.get('structured_data', {}).get('medications', [])
    
    def validate_fhir_bundle(self, bundle: Dict) -> tuple:
        """
        Basic validation of FHIR bundle
        
        Returns:
            (is_valid: bool, errors: List[str])
        """
        errors = []
        
        # Check required fields
        if bundle.get('resourceType') != 'Bundle':
            errors.append("Invalid resourceType, must be 'Bundle'")
        
        if bundle.get('type') not in ['transaction', 'batch', 'collection']:
            errors.append("Invalid bundle type")
        
        if 'entry' not in bundle or not isinstance(bundle['entry'], list):
            errors.append("Bundle must contain 'entry' array")
        
        # Validate each entry
        for i, entry in enumerate(bundle.get('entry', [])):
            if 'resource' not in entry:
                errors.append(f"Entry {i} missing 'resource' field")
            else:
                resource = entry['resource']
                if 'resourceType' not in resource:
                    errors.append(f"Entry {i} resource missing 'resourceType'")
        
        return len(errors) == 0, errors
