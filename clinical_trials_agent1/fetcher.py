import json
import time
import requests #type: ignore 
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)


class ClinicalTrialsFetcherAgent:
    """
    Fetcher agent for querying ClinicalTrials.gov API using OpenAI for query understanding.
    Uses the improved prompt and URL generation strategy from clinical (1).py.
    """
    
    def __init__(self, openai_client=None, model="gpt-4"):
        """
        Initialize the Clinical Trials Fetcher Agent.
        
        Args:
            openai_client: OpenAI client instance (required for URL generation)
            model: OpenAI model to use for query processing
        """
        self.base_url = "https://clinicaltrials.gov/api/v2"
        self.client = openai_client
        self.model = model
        self.system_prompt = self._get_system_prompt()
        
        if not self.client:
            logger.warning("OpenAI client not provided. URL generation will not be available.")
    
    def _get_system_prompt(self) -> str:
        """Return the comprehensive system prompt for the ClinicalTrials.gov API agent."""
        return '''
# ClinicalTrials.gov API Agent System Prompt

You are an AI agent that transforms natural language queries into valid ClinicalTrials.gov REST API v2.0.3 endpoints. Your primary goal is to create working API calls that return meaningful, non-empty results with actual information content.

## Base URL
`https://clinicaltrials.gov/api/v2`

## Core Strategy: Guarantee Information Content

**CRITICAL RULE**: Every URL must return studies with information content. Empty responses like `{'totalCount': 0, 'studies': []}` provide no value and must be avoided.

### Information-First Search Strategy

1. **Start Broad**: Begin with terms likely to have many results
2. **Avoid Over-Filtering**: Minimize intersections that could eliminate all results
3. **Diversify Approaches**: Each URL should use different search strategies
4. **Validate Relevance**: Ensure each URL captures different aspects of the query
5. **Prefer Inclusion**: Better to get broader results than no results

CRITICAL SUCCESS RULE: NO EMPTY RESPONSES
ABSOLUTE PRIORITY: Every URL must return studies with totalCount > 0. Empty responses like {'totalCount': 0, 'studies': []} are completely useless and must be avoided at all costs.

## Valid API Endpoints

### 1. Studies Search: `/studies`
**Primary endpoint for most queries**

**Essential Parameters:**
- `query.term`: General search terms
- `query.cond`: Medical conditions/diseases  
- `query.intr`: Interventions/treatments
- `filter.overallStatus`: Study status (use sparingly to avoid empty results)
- `pageSize`: Max 1000, default 10-50 for meaningful samples
- `countTotal`: true (VERY SIGNIFICANT USES THIS PARAMETER IN OUTPUT TO TEST THE URL CORRECTNESS)

**Valid Field Names** (use only these in `fields` parameter):
- `NCTId`, `BriefTitle`, `OfficialTitle`
- `OverallStatus`, `Phase`
- `Condition`, `Intervention`
- `EnrollmentCount`, `EnrollmentType`
- `StartDate`, `CompletionDate`, `StudyFirstPostDate`
- `LocationFacility`, `LocationCity`, `LocationState`, `LocationCountry`
- `PrimaryOutcomeMeasure`, `SecondaryOutcomeMeasure`
- `EligibilityCriteria`, `MinimumAge`, `MaximumAge`, `Gender`
- `LeadSponsorName`, `CollaboratorName`

### 2. Single Study: `/studies/{nctId}`
**For specific NCT numbers**

### 3. Field Statistics: `/stats/field/values`
**For analyzing common values across studies**

## Search Query Construction Principles

### Avoiding Empty Results
**DO:**
- Start with single, common medical terms
- Use OR operators to expand rather than AND to restrict
- Search across multiple fields separately (condition, intervention, general)
- Include broader synonyms and related terms
- Use reasonable page sizes (20-100) to get meaningful samples

**DON'T:**
- Combine multiple restrictive filters simultaneously
- Use highly specific terms as the only search criteria
- Over-constrain with status filters unless specifically requested
- Use complex nested boolean logic initially
- Rely on exact phrase matching for rare terms

### URL Diversification Strategy

Each of the 5 URLs should use a different approach to maximize information content:

1. **URL 1 - Broad Term Search**: Most general relevant term in `query.term`
2. **URL 2 - Condition-Focused**: Key condition in `query.cond`
3. **URL 3 - Intervention-Focused**: Key intervention in `query.intr`
4. **URL 4 - Alternative Terms**: Synonyms or related concepts
5. **URL 5 - Combined Approach**: Thoughtful combination that's still likely to return results

### Term Extraction and Expansion Rules

**From User Query → Multiple API Approaches:**

1. **Identify Core Concept**: Extract primary medical focus
2. **Find Broader Categories**: What larger field does this belong to?
3. **List Synonyms**: Alternative terms for same concept
4. **Consider Related Procedures**: What similar interventions exist?
5. **Think Upstream**: What conditions require this intervention?

### Status Filter Guidelines

**Use status filters cautiously:**
- Only apply when specifically requested by user
- If using status filters, ensure the base search term is broad enough
- Consider that combining rare terms + status filters often yields empty results
- Prefer broader searches that can be filtered post-retrieval

**Valid Status Values:**
- `RECRUITING`
- `ACTIVE_NOT_RECRUITING`
- `COMPLETED`
- `TERMINATED`
- `SUSPENDED`
- `WITHDRAWN`
- `NOT_YET_RECRUITING`

## Content-Rich Response Generation

### URL Design Principles

**Each URL should:**
- Target different aspects of the user's query
- Use different search fields (term vs condition vs intervention)
- Employ varying levels of specificity
- Include adequate page sizes for meaningful samples
- Avoid over-constraining filters

**Information Content Validation:**
- Prefer 50-200 results over 0-5 results
- Better to get broader relevant studies than no studies
- Each URL should contribute unique information to answer the query
- Avoid duplicate search strategies across URLs

## Error Prevention Strategy

### If Anticipating Low Results:
1. **Use broader terms**: "arterial access" not "transpedal arterial access"
2. **Search categories**: "vascular procedures" not specific technique names
3. **Multiple fields**: Spread search across condition/intervention/term
4. **Include related concepts**: Adjacent medical areas
5. **Remove constraints**: Skip status/phase filters initially

## Common Query Patterns - Content-Rich Approach

### Pattern 1: Count Studies (Ensure Non-Zero)
**User**: "How many diabetes studies are recruiting?"
**Strategy**: Start broad, then get specific
```
URL1: query.cond=diabetes&pageSize=100 (broad baseline)
URL2: query.cond=diabetes&filter.overallStatus=RECRUITING&pageSize=50
URL3: query.term=diabetes&pageSize=100 (alternative field)
URL4: query.cond=diabetes%20mellitus&pageSize=50 (formal term)
URL5: query.intr=diabetes&pageSize=50 (intervention perspective)
```

### Pattern 2: Find Studies (Multiple Angles)
**User**: "Find cancer immunotherapy trials"
**Strategy**: Different aspects of the same query
```
URL1: query.term=cancer%20immunotherapy&pageSize=100
URL2: query.cond=cancer&pageSize=100
URL3: query.intr=immunotherapy&pageSize=100
URL4: query.term=oncology%20immunology&pageSize=50
URL5: query.intr=checkpoint%20inhibitor&pageSize=50
```

### Pattern 3: Rare/Specific Terms
**User**: "Studies on transpedal arterial access"
**Strategy**: Pyramid from specific to broad, ensuring content
```
URL1: query.term=arterial%20access&pageSize=100 (likely to have results)
URL2: query.intr=vascular%20access&pageSize=100 (related, broad)
URL3: query.term=pedal%20access&pageSize=50 (more specific)
URL4: query.cond=peripheral%20artery%20disease&pageSize=100 (related condition)
URL5: query.term=percutaneous%20intervention&pageSize=100 (procedure category)
```

## CRITICAL: JSON Response Format

**YOU MUST ALWAYS RETURN VALID JSON. NO EXPLANATORY TEXT OUTSIDE THE JSON.**

For every query, analyze the user's request and return exactly this JSON structure:

```json
{
  "urls": [
    "url1",
    "url2",
    "url3",
    "url4",
    "url5"
  ]
}
```
SOME EXAMPLES OF INVALID RESPONSES:
```
400 Client Error for the ClinicalTrials.gov URLs that contain filter.locationCountry=India (or use query.cond=…)
The URL your Fetcher builds is not valid for v2 of the CTG API.
• query.cond= is not a legal parameter (use query.term=).
• filter.locationCountry= must be written exactly as the spec says (filter.locationCountry=, but the country has to be an ISO-3166-1 Alpha-2 code such as IN, not the full name).
```

**RESPONSE RULES:**
- Return ONLY the JSON object, nothing else
- No text before or after the JSON
- No markdown code blocks
- No explanations or comments
- Exactly 5 URLs in the array
- Each URL must be a complete, valid ClinicalTrials.gov API endpoint
- Each URL must be designed to return meaningful, non-empty results
- URLs should provide diverse perspectives on the query
- Prioritize information content over precision

## Success Metrics

**A successful response provides:**
- 5 URLs that each return studies (totalCount > 0)
- Diverse information covering different aspects of the query
- Sufficient data volume for meaningful analysis
- Multiple perspectives on the same medical topic
- Actionable information content for query answering

**Golden Rule**: Information-rich broad results are infinitely more valuable than precise empty results. Every URL must contribute meaningful content to answer the user's question.
'''
    
    def generate_api_urls(self, user_query: str, max_retries: int = 5, wait_seconds: int = 2) -> Optional[Dict[str, List[str]]]:
        """
        Generate API URLs using OpenAI to process the user query.
        
        Args:
            user_query: Natural language query about clinical trials
            max_retries: Maximum number of retry attempts
            wait_seconds: Wait time between retries
            
        Returns:
            Dictionary with 'urls' key containing list of API URLs, or None if failed
        """
        if not self.client:
            raise ValueError("OpenAI client not provided. Cannot generate URLs automatically.")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_query}
                    ]
                )
                response_content = response.choices[0].message.content
                
                # Clean the response content - remove any markdown or extra text
                # Try to find JSON content between curly braces
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    json_data = json.loads(json_content)
                    
                    # Check if the expected key is in the JSON
                    if 'urls' in json_data and isinstance(json_data['urls'], list):
                        logger.info(f"Successfully generated {len(json_data['urls'])} URLs for query")
                        return json_data
                
                logger.warning(f"Attempt {attempt + 1}: API response was not in the expected JSON format. Retrying...")
                
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to decode JSON: {e}. Retrying...")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: An unexpected error occurred: {e}. Retrying...")
            
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
        
        logger.error("Maximum retries reached. Could not get a valid JSON response.")
        return None
    
    def fetch_clinical_trials_data(self, urls: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Fetch data from multiple ClinicalTrials.gov API URLs.
        
        Args:
            urls: List of API URLs to fetch data from
            
        Returns:
            Tuple of (accessible_urls_content, inaccessible_urls)
        """
        accessible_urls_content = {}
        inaccessible_urls = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Check if the content is likely JSON
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    try:
                        # Attempt to parse as JSON to ensure validity
                        json_content = response.json()
                        
                        # Check if the response has actual studies
                        total_count = json_content.get('totalCount', 0)
                        studies = json_content.get('studies', [])
                        
                        if total_count > 0 and studies:
                            accessible_urls_content[url] = json_content
                            logger.info(f"✓ Successfully fetched {total_count} studies from: {url}")
                        else:
                            logger.warning(f"✗ Empty result set from: {url} (totalCount: {total_count})")
                            inaccessible_urls.append(url)
                            
                    except json.JSONDecodeError:
                        logger.error(f"✗ Could not parse JSON from: {url} (Invalid JSON format)")
                        inaccessible_urls.append(url)
                else:
                    logger.warning(f"✗ Skipping non-JSON content from: {url} (Content-Type: {content_type})")
                    inaccessible_urls.append(url)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Could not access URL: {url} - Error: {e}")
                inaccessible_urls.append(url)
            except Exception as e:
                logger.error(f"✗ An unexpected error occurred while processing URL: {url} - Error: {e}")
                inaccessible_urls.append(url)
        
        return accessible_urls_content, inaccessible_urls
    
    def collate_studies_data(self, accessible_urls_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collate data from multiple URLs into a single structure.
        
        Args:
            accessible_urls_content: Dictionary of URL -> JSON content
            
        Returns:
            Dictionary with collated studies data
        """
        all_studies = []
        total_count = 0
        source_urls = []
        
        for url, content in accessible_urls_content.items():
            studies = content.get('studies', [])
            all_studies.extend(studies)
            total_count += content.get('totalCount', 0)
            source_urls.append(url)
        
        # Remove duplicate studies based on NCT ID
        unique_studies = {}
        for study in all_studies:
            try:
                nct_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                if nct_id and nct_id not in unique_studies:
                    unique_studies[nct_id] = study
            except Exception as e:
                logger.warning(f"Error processing study: {e}")
        
        collated_data = {
            'studies': list(unique_studies.values()),
            'totalCount': len(unique_studies),
            'originalTotalCount': total_count,
            'sourceUrls': source_urls
        }
        
        logger.info(f"Collated {len(unique_studies)} unique studies from {len(source_urls)} sources")
        return collated_data
    
    def analyze_user_query(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to analyze user input and fetch clinical trials data.
        This is the primary method called by the pipeline.
        
        Args:
            user_input: Natural language query from user
            
        Returns:
            Dictionary containing analysis results and data
        """
        logger.info(f"Analyzing user query: {user_input}")
        
        try:
            # Step 1: Generate API URLs using OpenAI
            json_response = self.generate_api_urls(user_input)
            
            if not json_response or 'urls' not in json_response:
                return {
                    'success': False,
                    'error': 'Failed to generate API URLs from query',
                    'data': None,
                    'total_count': 0,
                    'source_url': ''
                }
            
            urls = json_response['urls']
            logger.info(f"Generated {len(urls)} URLs for query")
            
            # Step 2: Fetch data from URLs
            accessible_urls_content, failed_urls = self.fetch_clinical_trials_data(urls)
            
            if not accessible_urls_content:
                return {
                    'success': False,
                    'error': 'No data could be fetched from any generated URLs',
                    'data': None,
                    'total_count': 0,
                    'failed_urls': failed_urls,
                    'attempted_urls': urls,
                    'source_url': ''
                }
            
            # Step 3: Collate the data
            collated_data = self.collate_studies_data(accessible_urls_content)
            
            # Step 4: Prepare response
            return {
                'success': True,
                'data': collated_data,
                'total_count': collated_data['totalCount'],
                'studies_returned': len(collated_data['studies']),
                'source_url': collated_data['sourceUrls'][0] if collated_data['sourceUrls'] else '',
                'all_source_urls': collated_data['sourceUrls'],
                'failed_urls': failed_urls,
                'attempted_urls': urls,
                'query_analysis': {
                    'original_query': user_input,
                    'url_generation_strategy': 'content-rich-diversified',
                    'urls_attempted': len(urls),
                    'urls_successful': len(accessible_urls_content),
                    'unique_studies_found': collated_data['totalCount']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_user_query: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'total_count': 0,
                'source_url': ''
            }


# Backward compatibility functions
def create_clinical_trials_agent(openai_client=None, model="gpt-4") -> ClinicalTrialsFetcherAgent:
    """
    Factory function to create a ClinicalTrialsAgent instance.
    Maintained for backward compatibility.
    
    Args:
        openai_client: OpenAI client instance (required)
        model: Model to use for URL generation
        
    Returns:
        ClinicalTrialsAgent instance
    """
    return ClinicalTrialsFetcherAgent(openai_client=openai_client, model=model)