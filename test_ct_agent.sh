# Create test file
echo "
import os
os.environ['DEFAULT_LLM_MODEL'] = 'ollama/llama3'
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'

from agents.clinical_trials_agent.data_fetcher import ClinicalTrialsFetcher
f = ClinicalTrialsFetcher()
result = f.analyze_user_query('diabetes type 2 SGLT2 inhibitor trials')
print('success=', result['success'], 'studies=', result['total_count'])
" > test_ct.py

python test_ct.py