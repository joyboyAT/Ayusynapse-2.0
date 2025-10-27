"""
FHIR Client Module
Interact with FHIR servers using fhirclient library
"""

import json
from typing import Dict, List, Optional
import requests


class FHIRClient:
    """Client for interacting with FHIR servers"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize FHIR client
        
        Args:
            base_url: Base URL of FHIR server (e.g., 'https://hapi.fhir.org/baseR4')
        """
        # Use public HAPI FHIR test server if no URL provided
        self.base_url = base_url or "https://hapi.fhir.org/baseR4"
        self.headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }
    
    def post_bundle(self, bundle: Dict) -> Dict:
        """
        Post FHIR bundle to server
        
        Args:
            bundle: FHIR Bundle resource
            
        Returns:
            Server response
        """
        try:
            response = requests.post(
                self.base_url,
                json=bundle,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return {
                'success': True,
                'status_code': response.status_code,
                'response': response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None)
            }
    
    def search_patient(self, identifier: str) -> Dict:
        """
        Search for patient by identifier
        
        Args:
            identifier: Patient identifier
            
        Returns:
            Search results
        """
        try:
            url = f"{self.base_url}/Patient"
            params = {'identifier': identifier}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'results': response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_patient_conditions(self, patient_id: str) -> Dict:
        """
        Get all conditions for a patient
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            Bundle of Condition resources
        """
        try:
            url = f"{self.base_url}/Condition"
            params = {'patient': patient_id}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'results': response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_resource(self, resource: Dict) -> Dict:
        """
        Validate FHIR resource against server
        
        Args:
            resource: FHIR resource to validate
            
        Returns:
            Validation results
        """
        try:
            resource_type = resource.get('resourceType')
            url = f"{self.base_url}/{resource_type}/$validate"
            
            response = requests.post(
                url,
                json=resource,
                headers=self.headers,
                timeout=30
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response': response.json()
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_to_file(self, bundle: Dict, filepath: str):
        """
        Export FHIR bundle to JSON file
        
        Args:
            bundle: FHIR Bundle
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
    
    def import_from_file(self, filepath: str) -> Dict:
        """
        Import FHIR bundle from JSON file
        
        Args:
            filepath: Input file path
            
        Returns:
            FHIR Bundle
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)