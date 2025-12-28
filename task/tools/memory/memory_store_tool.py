import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class StoreMemoryTool(BaseTool):
    """
    Tool for storing long-term memories about the user.

    The orchestration LLM should extract important, novel facts about the user
    and store them using this tool. Examples:
    - User preferences (likes Python, prefers morning meetings)
    - Personal information (lives in Paris, works at Google)
    - Goals and plans (learning Spanish, traveling to Japan)
    - Important context (has a cat named Mittens)
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        # Provide self-descriptive name
        return "store_memory"

    @property
    def description(self) -> str:
        # Provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "Tool for storing long-term memories about the user."

    @property
    def parameters(self) -> dict[str, Any]:
        # Provide tool parameters JSON Schema:
        #  - content is string, description: "The memory content to store. Should be a clear, concise fact about the user.", required
        #  - category is string, description: "Category of the info (e.g., 'preferences', 'personal_info', 'goals', 'plans', 'context')", default is 'general' required
        #  - importance is number, description: "Importance score between 0 and 1. Higher means more important to remember.", minimum is 0, maximum is 1, default is 0.5
        #  - topics is array of strings, description: "Related topics or tags for the memory", default is empty array
        return {
            "type": "object",
            "properties": {     
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Should be a clear, concise fact about the user."
                },
                "category": {
                    "type": "string",
                    "description": "Category of the info (e.g., 'preferences', 'personal_info', 'goals', 'plans', 'context')",
                    "default": "general",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score between 0 and 1. Higher means more important to remember.",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Related topics or tags for the memory",
                    "default": []
                }
            },
            "required": ["content", "category"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        #TODO:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `content` from arguments
        content = arguments["content"]
        # 3. Get `category` from arguments
        category = arguments["category"]
        # 4. Get `importance` from arguments, default is 0.5
        importance = arguments.get("importance", 0.5)
        # 5. Get `topics` from arguments, default is empty array
        topics = arguments.get("topics", [])
        # 6. Call `memory_store` `add_memory` (we will implement logic in `memory_store` later)
        result = await self.memory_store.add_memory(
            api_key=tool_call_params.api_key,
            content=content, 
            category=category, 
            importance=importance, 
            topics=topics)

        # 7. Add result to stage
        tool_call_params.stage.append_content(result)
        # 8. Return result
        return result
