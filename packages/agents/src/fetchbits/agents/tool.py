from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from typing_extensions import Self



@dataclass

class ToolCallResult:

    id:str
    name:str
    arguments : dict[str, Any]
    result : Any


@dataclass

class Tool:
    name : str
    description : str
    parameters: dict[str, Any]
    on_tool_call : Callable

    @classmethod
    def from_callable(cls,callable : Callable) -> Self:
        schema =  convert_function_to_function_schema(callable)

        return cls(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parameters=schema["function"]["parameters"],
            on_tool_call=callable,
        )
    
    def to_function_schema(self) -> dict[str,Any]:
         """
        Convert the Tool to a standardized function schema format.

        Returns:
            Function schema dictionary with 'type' and 'function' keys.
        """
         return {
              "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
         }

