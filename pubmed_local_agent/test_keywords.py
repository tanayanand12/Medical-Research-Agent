"""
test_keyword_agent.py
~~~~~~~~~~~~~~~~~~~~~

Test script for keyword processing agents.
"""
import json
import logging
from keyword_agents import OpenAIClusteringAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("test_keyword_agent")

def test_keyword_agent():
    """Test the OpenAI clustering agent."""
    # Sample keywords
    keywords = [
    "Transradial access",
    "Transradial procedure",
    "Radial artery hemostasis",
    "Patent hemostasis",
    "Suction-actuated hemostasis",
    "Vascular closure device",
    "Radial artery patency",
    "Radial artery occlusion",
    "Hemostasis device",
    "Transradial catheterization",
    "Coronary catheterization",
    "Peripheral catheterization",
    "Reverse Barbeau test",
    "Digital plethysmography",
    "Duplex ultrasound",
    "Radial artery complications",
    "Access site complications",
    "Major access site complications",
    "Minor access site complications",
    "Hematoma",
    "Pseudoaneurysm",
    "Arteriovenous fistula",
    "Nerve compression",
    "Patient comfort",
    "Ease of use",
    "Device-related adverse effects",
    "Serious adverse effects",
    "Unanticipated adverse device effect",
    "Good Clinical Practice",
    "Clinical trial protocol",
    "Multicenter clinical study",
    "Nonrandomized clinical trial",
    "Interventional cardiology",
    "Anticoagulation therapy",
    "Heparin administration",
    "Medical adhesives",
    "Biocompatibility testing",
    "ISO 10993",
    "Informed consent process",
    "Institutional Review Board",
    "Ethics Committee",
    "MedDRA coding",
    "CTCAE grading",
    "Radial artery flow",
    "Ulnar artery compression"
]

    
    
    # [
    #     "glioblastoma", 
    #     "temozolomide", 
    #     "IDH1 mutation",
    #     "MGMT methylation",
    #     "radiation therapy",
    #     "microRNA",
    #     "immunotherapy",
    #     "blood-brain barrier",
    #     "tumor microenvironment",
    #     "CAR-T therapy",
    #     "checkpoint inhibitors",
    #     "VEGF inhibitors",
    #     "EGFR amplification",
    #     "PTEN loss",
    #     "p53 mutation",
    #     "exosomes",
    #     "tumor-treating fields",
    #     "bevacizumab",
    #     "recurrence",
    #     "survival rate"
    # ]
    
    # Initialize agent
    agent = OpenAIClusteringAgent(model="gpt-4o", max_batch_size=50)
    
    # Test clustering
    logger.info("Testing keyword clustering...")
    clusters = agent.cluster_keywords(keywords, max_clusters=5)
    logger.info(f"Clusters created: {len(clusters)}")
    logger.info(f"Clusters: {clusters}")
    
    # Print clusters
    logger.info("Clusters created:")
    for name, terms in clusters.items():
        logger.info(f"  {name}: {', '.join(terms)}")
    
    # Test URL generation
    logger.info("Testing URL generation...")
    urls = agent.format_pubmed_url(clusters, optimize_boolean=True)
    
    # Print URLs
    logger.info(f"Generated {len(urls)} URLs:")
    for i, url in enumerate(urls):
        logger.info(f"  URL {i+1}: {url}")
    
    # Test full processing
    logger.info("Testing full keyword processing...")
    all_urls = agent.process_keywords(
        keywords, 
        max_clusters=5, 
        optimize_boolean=True
    )
    
    logger.info(f"Process completed with {len(all_urls)} URLs")
    
    # Save output
    with open("agent_test_output(1).json", "w") as f:
        json.dump({
            "clusters": clusters,
            "urls": urls,
            "all_urls": all_urls
        }, f, indent=4)
    
    logger.info("Test output saved to agent_test_output.json")
    
if __name__ == "__main__":
    test_keyword_agent()
