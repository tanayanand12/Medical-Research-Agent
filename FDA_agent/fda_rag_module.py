# fda_rag_module.py
import os, time, json, hashlib, logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("fda_rag_module")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s")
    )
    logger.addHandler(h)

# --------------------------------------------------------------------------- #
# ENV  & OpenAI
# --------------------------------------------------------------------------- #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
_client = OpenAI(api_key=API_KEY)


# --------------------------------------------------------------------------- #
# FdaRAGModule
# --------------------------------------------------------------------------- #
class FdaRAGModule:
    """
    Generates structured answers for FDA RAG pipeline.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.client = _client
        self.model = model_name
        logger.info("Initialized FdaRAGModule (model=%s)", model_name)

    # --------------------------- PROMPTS ---------------------------------- #
    def create_system_prompt(self) -> str:
        # (verbatim prompt supplied by user)
        return (
            "You are an expert FDA regulatory data analyst specializing in analyzing and interpreting FDA drug labels, "
            "adverse event reports, recalls, and safety information. Your role is to provide comprehensive, "
            "evidence-based insights using the most current FDA regulatory data.\n\n"
            "CORE CAPABILITIES:\n"
            "- Analyze drug labeling information, indications, and contraindications\n"
            "- Interpret adverse event reports and safety signals\n"
            "- Evaluate recall classifications, reasons, and risk assessments\n"
            "- Assess drug approval status, regulatory pathways, and compliance\n"
            "- Compare safety profiles, efficacy data, and regulatory decisions across products\n"
            "- Identify patterns in adverse events, recall trends, and regulatory actions\n\n"
            "RESPONSE GUIDELINES:\n"
            "1. Always base responses on the provided FDA regulatory data\n"
            "2. Clearly distinguish between different types of FDA information:\n"
            "   - Drug labeling and prescribing information\n"
            "   - Adverse event reports and safety data\n"
            "   - Recall notices and risk classifications\n"
            "   - Regulatory approval and compliance information\n"
            "   - Manufacturing and quality control data\n"
            "   - Post-market surveillance findings\n"
            "3. When analyzing multiple records:\n"
            "   - Compare and contrast regulatory decisions and safety profiles\n"
            "   - Highlight similarities and differences in labeling requirements\n"
            "   - Note variations in adverse-event patterns and frequencies\n"
            "   - Identify trends in recall reasons and classifications\n"
            "4. For safety and regulatory information:\n"
            "   - Reference specific FDA databases (FAERS, drug labels, recalls)\n"
            "   - Note severity classifications and risk assessments\n"
            "   - Describe regulatory actions and their rationale when available\n"
            "   - Mention approval dates, regulatory pathways, and compliance status\n"
            "5. Present information in a structured, accessible format that includes:\n"
            "   - Clear section headers for different regulatory aspects\n"
            "   - Specific citations to relevant records using FDA IDs/NDCs\n"
            "   - Quantitative details when available (case counts, frequencies)\n"
            "   - Practical implications for healthcare providers and patients\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- Extensively interpret and provide information that is documented in the FDA data\n"
            "- Reason with the documented records and metadata from FDA databases\n"
            "- Clearly indicate your reasoning process and how you arrived at conclusions\n"
            "- Focus on factual regulatory characteristics and safety data\n\n"
            "Remember that FDA data represents regulatory decisions and safety monitoring and should be interpreted within appropriate clinical and regulatory context.\n\n"
            "HARD LIMITATIONS:\n"
            "- Do NOT use phrases like \"None of the identified FDA records explicitly mention\" or \"No specific data were found\" or "
            "\"not directly available\" or \"did not explicitly provide\" or anything that reduces response CONFIDENCE and RELEVANCE.\n"
            "- Extract and synthesize all relevant information from the provided FDA regulatory data.\n"
            "- Answer confidently and comprehensively based on the available FDA records.\n"
            "- State your reasoning clearly and provide detailed information from the FDA data.\n"
            "- Avoid making low-confidence statements or vague assertions.\n"
            "- Your task is to provide thorough analysis of FDA regulatory data, not to provide answers to personal questions other than basic queries like "
            "`how is the weather today?` OR `What is your name?` OR `who made you?` etc."
        )

    def create_user_prompt(self, query: str, context: str, records: List[Dict[str, Any]]) -> str:
        citations_lines = []
        for idx, rec in enumerate(records, 1):
            rid = rec.get("record_id") or self._hash12(rec)
            line = f"[{idx}] {rid} ({rec.get('chunk_type','chunk')}, Relevance: {rec.get('similarity_score',0):.3f})"
            citations_lines.append(line)

        citations_block = "\n".join(citations_lines) if citations_lines else "No specific records identified."
        context_block = context or "No specific FDA regulatory data context provided."

        return f"""
USER QUESTION: {query}

FDA REGULATORY DATA CONTEXT:
{context_block}

RELEVANT FDA RECORDS IDENTIFIED:
{citations_block}

Please provide a comprehensive analysis based on the FDA regulatory information above. Structure your response as follows:

1. **Executive Summary** (50-75 words)
2. **Detailed Analysis** – cite FDA IDs inline
3. **Regulatory Characteristics**
4. **Key Findings and Patterns**
5. **Safety and Compliance Implications**
6. **Data Limitations**

Always cite FDA record IDs (e.g., recall_number, event_key, NDC). If multiple records support a point, cite them all.  Focus strictly on factual regulatory information — no direct medical advice.
"""

    # --------------------------- GENERATE --------------------------------- #
    def generate(self, query: str, context: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": self.create_user_prompt(query, context, records)},
        ]

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1400,
        )
        latency = time.time() - start

        answer_text = response.choices[0].message.content

        # Defensive usage extraction
        total_tokens = 0
        try:
            total_tokens = response.usage.total_tokens  # openai>=1.3 returns CompletionUsage
        except Exception:
            pass

        logger.info("Generated FDA answer in %.2fs (tokens=%s)", latency, total_tokens)

        return {
            "success": True,
            "answer": answer_text,
            "citations": [rec.get("record_id") or self._hash12(rec) for rec in records],
            "metadata": {"generation_time": latency, "openai_tokens": total_tokens},
        }

    # ------------------------ small util ---------------------------------- #
    @staticmethod
    def _hash12(obj: Any) -> str:
        try:
            return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
        except Exception:
            return "unknown"



################################################################################

# fda_rag_module.py

# import logging
# import time
# import os
# from typing import List, Dict, Any
# from dotenv import load_dotenv #type: ignore
# from openai import OpenAI # type: ignore

# # ----------------------------------------------------------------------------
# # Logger Configuration
# # ----------------------------------------------------------------------------
# logger = logging.getLogger("fda_rag_module")
# logger.setLevel(logging.INFO)
# if not logger.hasHandlers():
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     fmt = logging.Formatter(
#         "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
#     )
#     ch.setFormatter(fmt)
#     logger.addHandler(ch)

# # Load environment
# load_dotenv()

# class FdaRAGModule:
#     """
#     RAG answer generator specialized for FDA regulatory data analysis including drug labels, 
#     adverse events, recalls, and safety information.
#     """

#     def __init__(self, model_name: str = "gpt-4o"):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             logger.error("OPENAI_API_KEY not set in environment")
#             raise ValueError("OPENAI_API_KEY missing")

#         self.client = OpenAI(api_key=api_key)
#         self.model = model_name
#         logger.info("Initialized FdaRAGModule with model: %s", model_name)

#     def create_system_prompt(self) -> str:
#         """
#         Create the system prompt for FDA regulatory data analysis.
        
#         Returns:
#             System prompt string
#         """
#         return """You are an expert FDA regulatory data analyst specializing in analyzing and interpreting FDA drug labels, adverse event reports, recalls, and safety information. Your role is to provide comprehensive, evidence-based insights using the most current FDA regulatory data.

# CORE CAPABILITIES:
# - Analyze drug labeling information, indications, and contraindications
# - Interpret adverse event reports and safety signals
# - Evaluate recall classifications, reasons, and risk assessments
# - Assess drug approval status, regulatory pathways, and compliance
# - Compare safety profiles, efficacy data, and regulatory decisions across products
# - Identify patterns in adverse events, recall trends, and regulatory actions

# RESPONSE GUIDELINES:
# 1. Always base responses on the provided FDA regulatory data
# 2. Clearly distinguish between different types of FDA information:
#    - Drug labeling and prescribing information
#    - Adverse event reports and safety data
#    - Recall notices and risk classifications
#    - Regulatory approval and compliance information
#    - Manufacturing and quality control data
#    - Post-market surveillance findings

# 3. When analyzing multiple records:
#    - Compare and contrast regulatory decisions and safety profiles
#    - Highlight similarities and differences in labeling requirements
#    - Note variations in adverse event patterns and frequencies
#    - Identify trends in recall reasons and classifications

# 4. For safety and regulatory information:
#    - Reference specific FDA databases (FAERS, drug labels, recalls)
#    - Note severity classifications and risk assessments
#    - Describe regulatory actions and their rationale when available
#    - Mention approval dates, regulatory pathways, and compliance status

# 5. Present information in a structured, accessible format that includes:
#    - Clear section headers for different regulatory aspects
#    - Specific citations to relevant records using FDA IDs/NDCs
#    - Quantitative details when available (case counts, frequencies)
#    - Practical implications for healthcare providers and patients

# IMPORTANT REQUIREMENTS:
# - Extensively interpret and provide information that is documented in the FDA data
# - Reason with the documented records and metadata from FDA databases
# - Clearly indicate your reasoning process and how you arrived at conclusions
# - Focus on factual regulatory characteristics and safety data

# Remember that FDA data represents regulatory decisions and safety monitoring and should be interpreted within appropriate clinical and regulatory context.

# HARD LIMITATIONS:
# - Do NOT use phrases like "None of the identified FDA records explicitly mention" or "No specific data were found" or "not directly available" or "did not explicitly provide" or anything that reduces response CONFIDENCE and RELEVANCE.
# - Extract and synthesize all relevant information from the provided FDA regulatory data.
# - Answer confidently and comprehensively based on the available FDA records.
# - State your reasoning clearly and provide detailed information from the FDA data.
# - Avoid making low confidence statements or vague assertions.
# - Your task is to provide thorough analysis of FDA regulatory data, not to provide answers to personal questions other than basic queries like `how is the weather today?` OR `What is your name?` OR `who made you?` etc.
# """

#     def create_user_prompt(self, query: str, context: str, records: List[Dict[str, Any]]) -> str:
#         """
#         Create the user prompt with query and FDA data context.
        
#         Args:
#             query: User query
#             context: Extracted context from FDA records
#             records: List of relevant FDA records metadata
            
#         Returns:
#             Formatted user prompt
#         """
#         # Format record citations
#         citations = []
#         for i, record in enumerate(records):
#             record_id = record.get("record_id", "unknown")
#             chunk_type = record.get("chunk_type", "FDA record")
#             score = record.get("similarity_score", 0.0)
#             citation = f"[{i+1}] {record_id} ({chunk_type}, Relevance: {score:.3f})"
#             citations.append(citation)
        
#         citations_text = "\n".join(citations) if citations else "No specific records identified."

#         logger.info("Creating user prompt for query: %s", query)
#         if not context:
#             context = "No specific FDA regulatory data context provided."
#             logger.warning("No context provided for the query, using default message.")
#         else:
#             logger.info("Context provided for query: %s", context)
#         if not citations_text:
#             citations_text = "No relevant FDA records found for this query."
#             logger.warning("No citations available for the query, using default message.")
#         else:
#             logger.info("Citations available for query: %s", citations_text)



#         prompt = f"""
# USER QUESTION: {query}

# FDA REGULATORY DATA CONTEXT:
# {context}

# RELEVANT FDA RECORDS IDENTIFIED:
# {citations_text}

# Please provide a comprehensive analysis based on the FDA regulatory information above. Structure your response as follows:

# 1. **Executive Summary** (50-75 words)
#    - Direct answer to the user's question
#    - Key findings from the FDA regulatory data

# 2. **Detailed Analysis** 
#    - Comprehensive synthesis of relevant FDA information
#    - Specific details from individual records with FDA ID references
#    - Comparison across multiple records when applicable

# 3. **Regulatory Characteristics**
#    - Overview of drug approvals, labeling requirements, and classifications
#    - Safety profiles and adverse event patterns
#    - Recall classifications and risk assessments

# 4. **Key Findings and Patterns**
#    - Notable trends or patterns across the identified FDA records
#    - Variations in regulatory approaches or safety signals
#    - Geographic distribution and regulatory actions if relevant

# 5. **Safety and Compliance Implications**
#    - What this means for healthcare providers and patients
#    - Considerations for regulatory compliance
#    - Current state of safety monitoring in this area

# 6. **Data Limitations**
#    - What information was not available in the FDA records
#    - Areas where additional regulatory data or monitoring might be needed

# Use FDA record IDs [FDA_ID/NDC/etc.] when referencing specific records. If multiple records address the same point, cite all relevant IDs.

# Focus on factual information from the FDA records and maintain appropriate regulatory context without providing direct medical advice.
# """
#         return prompt

#     def generate(self, query: str, context: str, records: List[Dict]) -> Dict[str, Any]:
#         """
#         Run RAG completion: system + user messages to OpenAI chat endpoints.
#         """
#         sys_msg = {"role": "system", "content": self.create_system_prompt()}
#         user_msg = {"role": "user", "content": self.create_user_prompt(query, context, records)}

#         prompt = [sys_msg, user_msg]
#         start = time.time()
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=prompt,
#             temperature=0.2,
#             max_tokens=1500,  # Increased for comprehensive responses
#         )
#         duration = time.time() - start

#         answer = response.choices[0].message.content
#         usage = response.usage

#         logger.info(
#             "Generated FDA analysis in %.2fs (tokens used: %s)",
#             duration,
#             getattr(usage, "total_tokens", None),
#         )

#         return {
#             "success": True,
#             "answer": answer,
#             "citations": [r.get("record_id") for r in records],
#             "metadata": {
#                 "generation_time": duration,
#                 "openai_tokens": getattr(usage, "total_tokens", 0),
#             },
#         }