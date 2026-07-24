"""
mcp_registry.py — Phase 2: Module-level MCP tool registry instance.

Provides a singleton ``mcp_registry`` object that auto-discovers all
MCPToolBase subclasses in the ``tools`` package.

Usage::

    from mcp_registry import mcp_registry

    tool = mcp_registry.get_tool("search_pubmed")
    result = tool.invoke(query="SGLT2 inhibitors", context={})
"""

from tools.mcp_base import MCPToolRegistry

# Module-level singleton — auto-discovers tools on first import.
mcp_registry = MCPToolRegistry()
