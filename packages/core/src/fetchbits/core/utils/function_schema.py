import contextlib
import inspect
import logging
from collections.abc import Callable, Generator

from typing import Any, get_args, get_origin, get_type_hints
from pydantic import BaseModel, create_model,Field


@contextlib.contextmanager

def _suppress_griffe_logging() -> Generator:
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def _generate_func_documentation(func : Callable[...,Any]) -> dict:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.

    Returns:
        A dict containing the function's name, description, and parameter
        descriptions.
    """

    name = func.__name__
    doc = inspect.getdoc(func)

    if not doc:
        return {"name" : name,"description": None, "param_descriptions": None}
    

    with _suppress_griffe_logging():
        docstring = Docstring(doc,lineno = 1,parser="google")
        parsed = docstring.parse()


    description : str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text),None
    )


    param_descriptions  : dict[str,str] = {
        param.name : param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return {
        "name" : func.__name__,
         "description": description,
        "param_descriptions": param_descriptions or None,
    }

def convert_function_to_function_schema(func : Callable[...,Any]) -> dict:
    """
    Given a python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.

    Returns:
        A dict containing the function's name, description, parameter descriptions,
        and other metadata.
    """

    #Grab docstring info

    doc_info = _generate_func_documentation(func)
    param_desc = doc_info["param_descriptions"] or {}

    func_name = doc_info["name"] if doc_info else func.__name__

    #Inspect function signature and get type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = list(sig.parameters.items())
    filtered_params = []

    if params:
        first_name,first_param = params[0]
         # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name,first_param.annotation)
        filtered_params.append((first_name,first_param))

    
    for name,param in params[1:]:
         ann = type_hints.get(name, param.annotation)
         filtered_params.append((name,param))


    fields : dict[str,Any] = {}


    for name,param in filtered_params:
        ann = type_hints.get(name,param.annotation)
        default = param.default

        if ann == inspect._empty:
            ann = Any
            

        # If a docstring param description exists, use it
        field_description = param_desc.get(name,None)

        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
             # e.g. *args: extend positional args
             if get_origin(ann) is tuple:
                 
                 args_of_tuple = get_args(ann)
                 args_of_tuple_with_ellipsis_length = 2
                 ann = (
                     list[args_of_tuple[0]]
                     if len(args_of_tuple) == args_of_tuple_with_ellipsis_length and args_of_tuple[1] is Ellipsis
                     else list[Any]
                 )
             else:
                 ann = list[ann]


             fields[name] = (
                 ann,
                Field(default_factory=list, description=field_description),  # type: ignore
             )


        elif param.kind == param.VAR_KEYWORD:
             # **kwargs handling
             if get_origin(ann) is dict:
                 dict_args = get_args(ann)
                 dict_args_to_check_length = 2

                 ann = (
                      dict[dict_args[0], dict_args[1]]
                      if len(dict_args) == dict_args_to_check_length
                      else dict[str, Any]
                 )
             else:
                 ann = dict[str, ann]  # type: ignore


             fields[name] = (
                  ann,
                Field(default_factory=dict, description=field_description), 
             )

        elif default == inspect._empty:
            fields[name] = (
                 ann,
                Field(..., description=field_description),
            )

        else:
            fields[name] = (
                ann,
                Field(default=default, description=field_description),
            )


    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)

    # 4. Build JSON schema from that model

    json_schema = dynamic_model.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": doc_info["description"] if doc_info else None,
            "parameters": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            },
        },
    }

    




