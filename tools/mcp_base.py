"""
mcp_base.py — Phase 2: MCP tool base class and auto-discovery registry.

MCPToolBase defines the contract for all retrieval tools.
MCPToolRegistry auto-discovers MCPToolBase subclasses in the ``tools`` package.
"""

import importlib
import logging
import pkgutil
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPToolBase(ABC):
    """Abstract base for MCP retrieval tools.

    Subclasses must set ``name`` and ``description`` as class attributes
    and implement :meth:`call`.

    The ``invoke`` convenience method translates the
    ``(query, context)`` signature used by the Phase-4 graph node
    ``parallel_retrieve`` into a single ``input_dict`` for ``call``.
    """

    name: str = ""
    description: str = ""

    # JSON Schema (Draft 7) — subclasses should override for validation.
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }

    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "tokens_used": {"type": "integer"},
            "cost": {"type": "number"},
            "retrieval_time_sec": {"type": "number"},
            "error": {},
        },
    }

    # Keywords that hint when this tool should be selected.
    triggers: List[str] = []

    # ------------------------------------------------------------------ #

    @abstractmethod
    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool.

        Parameters
        ----------
        input_dict : dict
            Must contain at least ``"query"`` (str).
            May contain ``top_k``, ``db_name``, and other tool-specific keys.

        Returns
        -------
        dict
            ``{"results": [...], "tokens_used": int, "cost": float,
              "retrieval_time_sec": float, "error": None | str}``
        """

    # ------------------------------------------------------------------ #
    # Convenience wrapper used by parallel_retrieve node
    # ------------------------------------------------------------------ #

    def invoke(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Translate ``(query, context)`` to ``call(input_dict)``.

        This is the interface consumed by
        ``nodes/parallel_retrieve.py::invoke_tool_with_timeout``.
        """
        input_dict: Dict[str, Any] = {"query": query}
        if context:
            input_dict.update(context)
        return self.call(input_dict)

    # ------------------------------------------------------------------ #
    # Helpers available to subclasses
    # ------------------------------------------------------------------ #

    @staticmethod
    def _success(
        results: list,
        tokens_used: int = 0,
        cost: float = 0.0,
        retrieval_time_sec: float = 0.0,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build a standard success response."""
        resp: Dict[str, Any] = {
            "results": results,
            "tokens_used": tokens_used,
            "cost": cost,
            "retrieval_time_sec": retrieval_time_sec,
            "error": None,
        }
        resp.update(extra)
        return resp

    @staticmethod
    def _error(
        message: str,
        retrieval_time_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Build a standard error response."""
        return {
            "results": [],
            "tokens_used": 0,
            "cost": 0.0,
            "retrieval_time_sec": retrieval_time_sec,
            "error": message,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ====================================================================== #
# Registry
# ====================================================================== #


class MCPToolRegistry:
    """Auto-discovering registry for MCP tools.

    On construction the registry imports every module in the ``tools``
    package (except ``mcp_base`` itself), finds concrete
    :class:`MCPToolBase` subclasses, instantiates each one, and registers
    it under its ``name``.

    Usage::

        registry = MCPToolRegistry()
        tool = registry.get_tool("search_pubmed")
        result = tool.invoke(query="SGLT2 inhibitors", context={})
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MCPToolBase] = {}
        self._auto_discover()

    # ------------------------------------------------------------------ #

    def _auto_discover(self) -> None:
        """Scan the ``tools`` package for MCPToolBase subclasses."""
        import tools as _pkg  # the package this module belongs to

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname == "mcp_base":
                continue
            fqn = f"tools.{modname}"
            try:
                module = importlib.import_module(fqn)
            except Exception:
                logger.warning(
                    "MCPToolRegistry: failed to import %s", fqn, exc_info=True
                )
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, MCPToolBase)
                    and attr is not MCPToolBase
                    and attr.name  # skip classes without a name
                ):
                    try:
                        instance = attr()
                        self._tools[instance.name] = instance
                        logger.info(
                            "MCPToolRegistry: registered tool %r from %s",
                            instance.name,
                            fqn,
                        )
                    except Exception:
                        logger.warning(
                            "MCPToolRegistry: failed to instantiate %s.%s",
                            fqn,
                            attr_name,
                            exc_info=True,
                        )

        logger.info(
            "MCPToolRegistry: discovered %d tools: %s",
            len(self._tools),
            list(self._tools.keys()),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_tool(self, name: str) -> MCPToolBase:
        """Return a registered tool by name.

        Raises
        ------
        KeyError
            If no tool is registered under *name*.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} not found. Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def register(self, tool: MCPToolBase) -> None:
        """Manually register a tool instance."""
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<MCPToolRegistry tools={self.list_tools()}>"
