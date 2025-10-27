import requests
import json
import os
import time
from typing import List, Dict, Any, Optional
import urllib.parse

class ClinicalTrialsClient:
    """Client for accessing ClinicalTrials.gov API v2 in real-time"""
    
    def __init__(self, cache_enabled=True, cache_duration=86400):
        """
        Initialize the clinical trials API client
        
        Args:
            cache_enabled: Whether to cache API responses
            cache_duration: How long to cache responses (in seconds, default 24 hours)
        """
        # Use the new API v2 endpoint
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.cache_enabled = cache_enabled
        self.cache_duration = cache_duration
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def search_trials(self, 
                      conditions: List[str] = None, 
                      max_results: int = 20,
                      gender: str = None,
                      min_age: int = None,
                      max_age: int = None,
                      status: str = None,
                      country: str = None,
                      state: str = None) -> List[Dict[str, Any]]:
        """
        Search for clinical trials using the ClinicalTrials.gov API v2
        
        Args:
            conditions: List of medical conditions to search for
            max_results: Maximum number of results to return
            gender: Filter by gender ("Male", "Female", "All")
            min_age: Minimum age in years
            max_age: Maximum age in years
            status: Trial status (e.g., "Recruiting", "Active, not recruiting")
            country: Filter by country
            state: State filter (US only)
            
        Returns:
            List of matching clinical trial data
        """
        # Build query parameters for API v2
        params = {
            'format': 'json',
            'pageSize': max_results
        }
        
        # Build query string - API v2 uses simpler query syntax
        query_parts = []
        
        # Add conditions to query - fix the syntax
        if conditions:
            # Use simple condition search without AREA syntax
            condition_query = ' OR '.join(conditions)
            query_parts.append(f'({condition_query})')
        
        # Add status filter
        if status:
            # Use the correct field name for status
            query_parts.append(f'{status}')
        
        # Add location filter
        if country:
            query_parts.append(country)
        
        if state:
            query_parts.append(state)
        
        # Combine query parts with proper parameter name
        if query_parts:
            params['query.term'] = ' AND '.join(query_parts)
        
        # Check cache first
        cache_key = self._generate_cache_key(params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            print(f"Using cached data for query")
            return cached_data
        
        # Make API request
        try:
            print(f"Calling ClinicalTrials.gov API with params: {params}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            print(f"API Response status: {response.status_code}")
            print(f"Studies found: {data.get('totalCount', 0)}")
            
            # Transform the response
            transformed = self._transform_api_v2_response(data, gender, min_age, max_age)
            
            # Cache the transformed response
            if self.cache_enabled and transformed:
                self._save_to_cache(cache_key, transformed)
            
            return transformed
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching trials: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text[:500]}")
            return []
    
    def get_trial_details(self, nct_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific trial by NCT ID"""
        # Check cache first
        cache_key = f"trial_details_{nct_id}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Use API v2 endpoint for specific study
        url = f"{self.base_url}/{nct_id}"
        params = {'format': 'json'}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            if self.cache_enabled:
                self._save_to_cache(cache_key, data)
            
            return data
                
        except requests.exceptions.RequestException as e:
            print(f"Error getting trial details: {str(e)}")
            return {}
    
    def _transform_api_v2_response(self, api_response: Dict[str, Any], 
                                    filter_gender: str = None,
                                    filter_min_age: int = None,
                                    filter_max_age: int = None) -> List[Dict[str, Any]]:
        """Transform the API v2 response into a standardized format"""
        
        if 'studies' not in api_response:
            return []
        
        studies = api_response.get('studies', [])
        transformed_trials = []
        
        for study_wrapper in studies:
            study = study_wrapper.get('protocolSection', {})
            
            # Extract identification info
            ident_module = study.get('identificationModule', {})
            nct_id = ident_module.get('nctId', '')
            title = ident_module.get('briefTitle', '')
            official_title = ident_module.get('officialTitle', '')
            
            # Extract conditions
            conditions_module = study.get('conditionsModule', {})
            conditions = conditions_module.get('conditions', [])
            
            # Extract status
            status_module = study.get('statusModule', {})
            status = status_module.get('overallStatus', 'Unknown')
            
            # Extract eligibility
            eligibility_module = study.get('eligibilityModule', {})
            gender = eligibility_module.get('sex', 'ALL')
            min_age_text = eligibility_module.get('minimumAge', '0 Years')
            max_age_text = eligibility_module.get('maximumAge', '100 Years')
            
            # Parse ages
            min_age = self._parse_age_to_years(min_age_text)
            max_age = self._parse_age_to_years(max_age_text)
            
            # Apply filters
            if filter_gender:
                if gender.upper() not in ['ALL', filter_gender.upper()]:
                    continue
            
            if filter_min_age is not None or filter_max_age is not None:
                patient_age = filter_min_age or filter_max_age
                if patient_age:
                    if patient_age < min_age or patient_age > max_age:
                        continue
            
            # Extract locations
            locations = []
            contacts_locations = study.get('contactsLocationsModule', {})
            location_list = contacts_locations.get('locations', [])
            
            for loc in location_list:
                location = {
                    'facility': loc.get('facility'),
                    'city': loc.get('city'),
                    'state': loc.get('state'),
                    'country': loc.get('country'),
                    'zip': loc.get('zip')
                }
                locations.append(location)
            
            # Standardize gender
            if gender.upper() == 'ALL':
                gender_code = 'Both'
            elif gender.upper() == 'MALE':
                gender_code = 'M'
            elif gender.upper() == 'FEMALE':
                gender_code = 'F'
            else:
                gender_code = 'Both'
            
            # Extract description/summary
            desc_module = study.get('descriptionModule', {})
            summary = desc_module.get('briefSummary', '')
            
            # Extract design info
            design_module = study.get('designModule', {})
            phases = design_module.get('phases', ['Not Applicable'])
            phase = ', '.join(phases) if phases else 'Not Applicable'
            study_type = design_module.get('studyType', '')
            
            # Create transformed trial object
            transformed_trial = {
                'nct_id': nct_id,
                'title': title,
                'official_title': official_title,
                'conditions': conditions,
                'status': status,
                'gender': gender_code,
                'age_range': {
                    'min': min_age,
                    'max': max_age,
                    'min_text': min_age_text,
                    'max_text': max_age_text
                },
                'locations': locations,
                'phase': phase,
                'summary': summary,
                'study_type': study_type
            }
            
            transformed_trials.append(transformed_trial)
        
        return transformed_trials
    
    def _parse_age_to_years(self, age_text: str) -> int:
        """Convert age text (e.g., '18 Years') to integer years"""
        try:
            if not age_text or age_text.upper() == 'N/A':
                return 0
            
            # Handle special cases
            if 'child' in age_text.lower():
                return 0
            if 'adult' in age_text.lower():
                return 18
            if 'older adult' in age_text.lower():
                return 65
                
            parts = age_text.split()
            if len(parts) < 2:
                return 0
                
            value = float(parts[0])
            unit = parts[1].lower()
            
            if 'year' in unit:
                return int(value)
            elif 'month' in unit:
                return max(0, int(value / 12))
            elif 'week' in unit:
                return max(0, int(value / 52))
            elif 'day' in unit:
                return max(0, int(value / 365))
            return int(value)
        except:
            return 0
    
    def _generate_cache_key(self, params: Dict) -> str:
        """Generate a unique cache key based on the request parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return f"trials_cache_{hash(param_str)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached response if available and not expired"""
        if not self.cache_enabled:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        # Check if cache is expired
        if time.time() - os.path.getmtime(cache_file) > self.cache_duration:
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict]) -> None:
        """Save API response to cache"""
        if not self.cache_enabled:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    def _get_neighboring_states(self, state: str) -> List[str]:
        """Get a list of neighboring states for geographic expansion (US only)"""
        # Placeholder for actual implementation
        # This should return a list of state abbreviations based on the input state
        return []
    
    def expand_location_search(self, country: str, state: str) -> List[Dict[str, Any]]:
        """
        Expand the location search to include neighboring states (US only)
        
        Args:
            country: Country filter
            state: State filter
            
        Returns:
            List of parameter sets for expanded search
        """
        # Placeholder for actual implementation
        # This should return a list of parameter sets with expanded location filters
        return []
    
    def api_call_with_retry(self, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Make API call with retry mechanism in case of transient errors
        
        Args:
            params: API request parameters
            max_retries: Maximum number of retries (default 3)
            
        Returns:
            API response data
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error calling API (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}
