# umls_client.py

import requests
import os
from typing import List, Dict, Any, Optional

# UMLS Endpoints
UMLS_AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
UMLS_SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
UMLS_CONTENT_URL = "https://uts-ws.nlm.nih.gov/rest/content/current"

class UMLSClient:
    """Enhanced UMLS API client for medical concept mapping"""
    
    def __init__(self, api_key=None):
        # Use provided key, environment variable, or fallback to module constant
        self.api_key = api_key or os.environ.get('UMLS_API_KEY') or API_KEY
        self.tgt_url = None
        self.debug = False  # Set to True for debugging API calls
        
    def get_tgt(self):
        """Get a Ticket Granting Ticket (TGT) from UMLS."""
        params = {'apikey': self.api_key}
        response = requests.post(UMLS_AUTH_URL, data=params)

        if response.status_code != 201:
            raise Exception(f"Failed to get TGT: {response.text}")

        # Extract TGT URL from response
        self.tgt_url = response.headers.get("location")
        return self.tgt_url

    def get_service_ticket(self, service=None):
        """Get a Service Ticket (ST) using TGT."""
        if not self.tgt_url:
            self.get_tgt()
        
        # Default service URL if none provided    
        if service is None:
            service = "http://umlsks.nlm.nih.gov"
            
        params = {'service': service}
        response = requests.post(self.tgt_url, data=params)

        if response.status_code != 200:
            raise Exception(f"Failed to get Service Ticket: {response.text}")

        return response.text

    def search_umls(self, query, page=1, page_size=10, sabs=None):
        """
        Search UMLS terms using the API.
        
        Args:
            query: Search term
            page: Page number for results
            page_size: Results per page
            sabs: Source vocabularies to search (comma-separated string)
        """
        if not self.tgt_url:
            self.get_tgt()
        
        st = self.get_service_ticket()
        
        params = {
            'ticket': st,
            'string': query,
            'pageNumber': page,
            'pageSize': page_size
        }
        
        # Add source vocabularies if specified
        if sabs:
            params['sabs'] = sabs
            
        response = requests.get(UMLS_SEARCH_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"Search failed: {response.text}")

        results = response.json()
        return results.get("result", {}).get("results", [])
    
    def get_concept_info(self, cui):
        """Get detailed information about a UMLS concept by CUI"""
        if not self.tgt_url:
            self.get_tgt()
            
        st = self.get_service_ticket()
        
        url = f"{UMLS_CONTENT_URL}/CUI/{cui}"
        params = {'ticket': st}
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get concept info: {response.text}")
            
        return response.json()
    
    def get_concept_semantics(self, cui):
        """Get semantic types for a concept"""
        if not self.tgt_url:
            self.get_tgt()
            
        # Use the content endpoint as the service URL
        service_url = f"{UMLS_CONTENT_URL}/CUI/{cui}/semanticTypes"
        st = self.get_service_ticket(service_url)
        
        # Fixed URL format
        url = f"{UMLS_CONTENT_URL}/CUI/{cui}/semanticTypes"
        params = {'ticket': st}
        
        if self.debug:
            print(f"DEBUG: Calling semantic types API: {url}")
            
        try:
            response = requests.get(url, params=params)
            
            # Special handling for 404 errors
            if response.status_code == 404:
                if self.debug:
                    print(f"DEBUG: Semantic type not found for {cui}, using alternative method")
                return self._get_semantics_alternative(cui)
                
            response.raise_for_status()
            data = response.json()
            return [semantic['name'] for semantic in data.get('result', [])]
            
        except Exception as e:
            # More graceful error handling - return empty list instead of raising exception
            if self.debug:
                print(f"DEBUG: Error in get_concept_semantics: {str(e)}")
            return []
    
    def _get_semantics_alternative(self, cui):
        """Alternative method to get semantic types when the primary method fails"""
        try:
            # Try to get concept info which may contain semantic type info
            concept_info = self.get_concept_info(cui)
            semantics = []
            
            # Extract semantic types from concept info if available
            if 'result' in concept_info and 'semanticTypes' in concept_info['result']:
                semantics = [st['name'] for st in concept_info['result']['semanticTypes']]
                
            # Use hardcoded semantic types for common concept types as fallback
            if not semantics:
                semantic_map = {
                    # Disease concepts
                    'C0011847': ['Disease or Syndrome'],  # Diabetes Mellitus
                    'C0011849': ['Disease or Syndrome'],  # Diabetes Mellitus
                    'C0011860': ['Disease or Syndrome'],  # Type 2 Diabetes
                    'C0020538': ['Disease or Syndrome'],  # Hypertension
                    'C0020456': ['Disease or Syndrome'],  # Hypertensive disease
                    
                    # Medication concepts
                    'C0025598': ['Pharmacologic Substance', 'Organic Chemical'],  # Metformin
                    'C0065374': ['Pharmacologic Substance', 'Organic Chemical'],  # Lisinopril
                    
                    # Generic fallbacks
                    'C0012634': ['Disease or Syndrome'],  # Disease
                    'C0013227': ['Pharmacologic Substance'],  # Drug/Medicine
                }
                
                if cui in semantic_map:
                    semantics = semantic_map[cui]
            
            return semantics
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Alternative semantic type lookup failed: {str(e)}")
            return []
    
    def map_medical_terms(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Map medical terms to UMLS concepts"""
        if self.debug:
            print(f"Mapping {len(terms)} terms to UMLS concepts...")
        
        # Add synonym dictionary for fallback
        synonym_dict = {
            'heart attack': ['myocardial infarction', 'MI', 'acute coronary syndrome'],
            'stroke': ['cerebrovascular accident', 'CVA', 'brain infarction'],
            'high blood pressure': ['hypertension', 'HTN', 'elevated BP']
        }
        
        mapped_concepts = []
        for term in terms:
            try:
                # Try direct mapping first
                results = self.search_umls(term, page_size=5)
                
                # If no results, try synonyms
                if not results and term.lower() in synonym_dict:
                    for synonym in synonym_dict[term.lower()]:
                        results = self.search_umls(synonym, page_size=3)
                        if results:
                            break
                
                # Add confidence scoring based on string similarity
                if results:
                    best_match = max(results, key=lambda x: x.get('score', 0))
                    concept = self._create_concept_with_confidence(term, best_match)
                    mapped_concepts.append(concept)
                else:
                    # Better fallback with local medical dictionary
                    mapped_concepts.append(self._create_fallback_concept(term))
                    
            except Exception as e:
                # Graceful error handling - don't break entire process
                self._log_mapping_error(term, str(e))
                mapped_concepts.append(self._create_error_concept(term, str(e)))
        
        return mapped_concepts

    def _create_concept_with_confidence(self, term, best_match):
        """Create concept with confidence scoring"""
        cui = best_match.get('ui')
        
        # Get additional information
        semantic_types = []
        try:
            if cui:
                semantic_types = self.get_concept_semantics(cui)
        except Exception as e:
            if self.debug:
                print(f"Error getting semantic types for {cui}: {e}")
        
        concept = {
            'cui': cui,
            'original_term': term,
            'preferred_name': best_match.get('name', term),
            'semantic_types': semantic_types,
            'score': best_match.get('score', 0),
            'matched_term': best_match.get('matchedWord', term),
            'confidence': self._calculate_confidence(term, best_match.get('name', ''))
        }
        return concept
    
    def _create_fallback_concept(self, term):
        """Create fallback concept when UMLS mapping fails"""
        return {
            'cui': f"UNKNOWN_{term.upper().replace(' ', '_')}",
            'preferred_name': term,
            'original_term': term,
            'semantic_types': [],
            'score': 0,
            'confidence': 0.0
        }
    
    def _create_error_concept(self, term, error_msg):
        """Create error concept when mapping fails"""
        return {
            'cui': f"ERROR_{term.upper().replace(' ', '_')}",
            'preferred_name': term,
            'original_term': term,
            'semantic_types': [],
            'score': 0,
            'error': error_msg,
            'confidence': 0.0
        }
    
    def _log_mapping_error(self, term, error_msg):
        """Log mapping errors for debugging"""
        if self.debug:
            print(f"DEBUG: Mapping error for '{term}': {error_msg}")
    
    def _calculate_confidence(self, original, mapped):
        """Calculate confidence score based on string similarity"""
        if not original or not mapped:
            return 0.0
        
        original_lower = original.lower()
        mapped_lower = mapped.lower()
        
        # Exact match
        if original_lower == mapped_lower:
            return 1.0
        
        # Contains match
        if original_lower in mapped_lower or mapped_lower in original_lower:
            return 0.8
        
        # Word overlap
        original_words = set(original_lower.split())
        mapped_words = set(mapped_lower.split())
        
        if original_words & mapped_words:
            overlap = len(original_words & mapped_words)
            total = len(original_words | mapped_words)
            return overlap / total
        
        return 0.0

# Keep the existing API key and standalone functions for backward compatibility
API_KEY = "73a53308-df70-45e5-a37a-940cae48d5fc"

def get_tgt(api_key=API_KEY):
    """
    Get a Ticket Granting Ticket (TGT) from UMLS.
    """
    params = {'apikey': api_key}
    response = requests.post(UMLS_AUTH_URL, data=params)

    if response.status_code != 201:
        raise Exception(f"Failed to get TGT: {response.text}")

    # Extract TGT URL from response
    tgt_url = response.headers.get("location")
    return tgt_url


def get_service_ticket(tgt_url, service="http://umlsks.nlm.nih.gov"):
    """
    Get a Service Ticket (ST) using TGT.
    """
    params = {'service': service}
    response = requests.post(tgt_url, data=params)

    if response.status_code != 200:
        raise Exception(f"Failed to get Service Ticket: {response.text}")

    return response.text


def search_umls(query, page=1, page_size=5):
    """
    Search UMLS terms using the API.
    """
    tgt_url = get_tgt()  # Step 2
    st = get_service_ticket(tgt_url)  # Step 3

    params = {
        'ticket': st,
        'string': query,
        'pageNumber': page,
        'pageSize': page_size
    }

    response = requests.get(UMLS_SEARCH_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Search failed: {response.text}")

    results = response.json()
    return results.get("result", {}).get("results", [])


if __name__ == "__main__":
    # Example usage
    term = "diabetes"
    print(f"Searching UMLS for: {term}")
    results = search_umls(term)

    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item.get('name')} (UI: {item.get('ui')})")
        
    # Example using the new class
    print("\nUsing UMLSClient class:")
    client = UMLSClient()
    mapped = client.map_medical_terms(["diabetes", "hypertension"])
    for concept in mapped:
        print(f"{concept['original_term']} -> {concept['preferred_name']} ({concept['cui']})")
