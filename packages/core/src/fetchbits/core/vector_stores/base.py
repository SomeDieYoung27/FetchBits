from abc import ABC,abstractmethod
from enum import Enum
from typing import ClassVar,TypeVar,cast
from uuid import UUID

import pydantic
from pydantic import BaseModel
from typing_extensions import Self

from fetchbits.core import vector_stores
from fetchbits.core.embeddings import DenseEmbedder, Embedder, SparseVector
from fetchbits.core.options import Options
from fetchbits.core.utils.config_handling import ConfigurableComponent, ObjectConstructionConfig
from fetchbits.core.utils.pydantic import SerializableBytes

WHEREQUERY = dict[str, str | int | float | bool | dict]


class VectorStoreEntry(BaseModel):
    id : UUID
    text : str | None = None
    image_bytes : SerializableBytes | None = None
    metadata : dict = {}


    @pydantic.model_validator(mode="after")

    def validate_metadata_serializable(self) -> Self:


        try:
            self.model_dump_json()

        except Exception as e :
            raise ValueError(f"Metadata must be JSON serializable. Error: {str(e)}") from e

        return self


    @pydantic.model_validator(mode = "after")


    def text_or_image_required(self) -> Self:

        if not self.text or self.image_bytes:
            raise ValueError("Either text or image_bytes must be provided.")
        

        return self
    

class VectorStoreResult(BaseModel):
    
    entry : VectorStoreEntry
    vector : list[float] | SparseVector
    score : float

    subresults : list["VectorStoreResult"] = []

