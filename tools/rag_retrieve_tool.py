"""
rag_retrieve_tool.py — Auto-discovery shim for MCPToolRegistry.

Re-exports :class:`RAGTool` from ``rag_engine.mcp_rag_tool`` so that
:class:`MCPToolRegistry`'s ``tools`` package scan finds and registers it.
"""

from rag_engine.mcp_rag_tool import RAGTool  # noqa: F401

__all__ = ["RAGTool"]
