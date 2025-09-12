from .clinical_trials_context_extractor import ClinicalTrialsContextExtractor

# Simply re-export with a new name so imports stay clear.
class FdaContextExtractor(ClinicalTrialsContextExtractor):
    """No changes needed; FDA chunks use the same logic as clinical trial contexts."""
    pass
