import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, get_origin

from ragbits.core.audit.traces import trace
from ragbits.core.config import core_config
from fetchbits.core.prompt import Prompt


class PromptDiscovery:
    """
     Discovers Prompt objects within Python modules.
     Args:
        file_pattern (str): The file pattern to search for Prompt objects. Defaults to "**/prompt_*.py"
        root_path (Path): The root path to search for Prompt objects. Defaults to the directory where the script is run.
    """

    def __init__(self, file_pattern: str = core_config.prompt_path_pattern, root_path: Path | None = None):
        self.file_pattern = file_pattern
        self.root_path = root_path or Path.cwd()

    
    @staticmethod
    def is_prompt_subclass(obj : Any) -> bool:

        return inspect.isclass(obj) and  not get_origin(obj) and issubclass(obj, Prompt) and obj != Prompt
    

    def discover(self) -> set[type[Prompt]]:
        """
        Discovers Prompt objects within the specified file paths.

        Returns:
            set[Prompt]: The discovered Prompt objects.
        """
        with trace(file_patern = self.file_pattern,path = self.root_path) as outputs:
            result_set: set[type[Prompt]] = set()
            for file_path in self.root_path.glob(self.file_pattern):

                  module_name = str(file_path).rsplit(".", 1)[0].replace(os.sep, ".")

                  spec = importlib.util.spec_from_file_location(module_name, file_path)

                  if spec is None:
                      print(f"Skipping {file_path}, not a Python module")
                      continue
                  

                  module = importlib.util.module_from_spec(spec)

                  assert spec.loader is not None


                  try:
                       spec.loader.exec_module(module)

                  except Exception as e:
                       print(f"Skipping {file_path}, loading failed: {e}")
                       continue
                  

                  for _,obj in inspect.getmembers(module):
                      if self.is_prompt_subclass(obj):
                          result_set.add(obj)


                  outputs.results_set = result_set


        return result_set