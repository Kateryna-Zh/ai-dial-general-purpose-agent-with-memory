import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        # Provide self-descriptive name
        return "search_memory"

    @property
    def description(self) -> str:
        # Provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "Tool for searching long-term memories about the user."

    @property
    def parameters(self) -> dict[str, Any]:
        # Provide tool parameters JSON Schema:
        #  - query is string, description: "The search query. Can be a question or keywords to find relevant memories", required
        #  - top_k is integer, description: "Number of most relevant memories to return.", minimum is 1, maximum is 20, default is 5
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        #TODO:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `query` from arguments
        query = arguments["query"]
        # 3. Get `top_k` from arguments, default is 5
        top_k = arguments.get("top_k", 5)
        # 4. Call `memory_store` `search_memories` (we will implement logic in `memory_store` later)
        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        # 5. If results are empty then set `final_result` as "No memories found.",
        #    otherwise iterate through results and collect content, category and topics (if preset) in markdown format
        if not results:
            final_result = "No memories found."
        else:
            final_result = ""
            for memory in results:
                final_result += f"- **Content:** {memory.content}\n"
                final_result += f"  **Category:** {memory.category}\n"
                if memory.topics:
                    final_result += f"  **Topics:** {', '.join(memory.topics)}\n"
                final_result += "\n"

        # 6. Add result to stage as markdown text
        tool_call_params.stage.append_content(final_result)
        # 7. Return result
        return final_result
