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


class VectorStoreOptions(Options):

    k : int = 5
    score_threshold : float | None = None
    where : WHEREQUERY | None = None


VectorStoreOptionsT = TypeVar("VectorStoreOptionsT", bound = "VectorStoreOptions")


class VectorStore(ConfigurableComponent[VectorStoreOptionsT],ABC):

    options_cls : type[VectorStoreOptionsT]
    default_module : ClassVar = vector_stores
    configuration_key : ClassVar = "vector_store"


    @abstractmethod

    async def store(self,entries : list[VectorStoreEntry]) -> None:
        """
        Add an entry to the vector store.
        """

    @abstractmethod

    async def retreive(self,text : str,options: VectorStoreOptionsT | None = None) -> list[VectorStoreResult]:
         """
        Retrieve entries from the vector store most similar to the provided text.

        Args:
            text: The text to query the vector store with.
            options: The options for querying the vector store.

        Returns:
            The entries.
        """


    @abstractmethod

    async def remove(self,ids : list[UUID]) -> None:
        """
        Remove entries from the vector store.

        Args:
            ids: The list of entries' IDs to remove.
        """

    async def list(self,where : WHEREQUERY | None = None,limit : int | None = None,offset : int = 0) -> list[VectorStoreEntry]:
        """
        List entries from the vector store. The entries can be filtered, limited and offset.

        Args:
            where: The filter dictionary - the keys are the field names and the values are the values to filter by.
                Not specifying the key means no filtering.
            limit: The maximum number of entries to return.
            offset: The number of entries to skip.

        Returns:
            The entries.
        """


class EmbeddingType(Enum):

    TEXT = "text"
    IMAGE = "image"

class VectorStoreWithDenseEmbedder(VectorStore[VectorStoreOptionsT]):
    """
    Base class for vector stores that takes a dense embedder as an argument.
    """

    def __init__(self,embedder : DenseEmbedder,embedding_type : EmbeddingType = EmbeddingType.TEXT,default_options : VectorStoreOptionsT | None = None) -> None:

        super().__init__(default_options)

        self.embedder = embedder
        self.embedding_type = embedding_type

        if self.embedding_type == EmbeddingType.IMAGE and not self._embedder.supports_image_embeddings:
            raise ValueError("The embedder does not support image embeddings.")
        

    async def _create_embeddings(self,entries : list[VectorStoreEntry]) -> dict[UUID,list[float]]:
        """
        Create embeddings for the given entry, using the provided embedder and embedding type.

        Args:
            entries: The entries to create embeddings for.

        Returns:
            The embeddings mapped by entry ID
        """
        if self.embedding_type == EmbeddingType.TEXT:
            entries = [e for e in entries if e.text is not None]

            embeddings = await self._embedder.embed_text([e.text for e in entries if e.text is not None])
            return {e.id : v for e,v in zip(entries,embeddings,strict = True)}
        
        elif self.embedding_type == EmbeddingType.IMAGE:
             entries = [e for e in entries if e.image_bytes is not None]
             embeddings = await self._embedder.embed_image([e.image_bytes for e in entries if e.image_bytes is not None])
             return {e.id: v for e, v in zip(entries, embeddings, strict=True)}
        
        else:
            raise ValueError(f"Unsupported embedding type: {self._embedding_type}")
        

    @classmethod

    def from_config(cls,config : dict) -> Self:
        """
        Initializes the class with the provided configuration.

        Args:
            config: A dictionary containing configuration details for the class.

        Returns:
            An instance of the class initialized with the provided configuration.
        """

        default_options = config.pop("default_options",None)
        options = cls.options_cls(**default_options) if default_options else None

        embedder_config = config.pop("embedder")

        embedder : DenseEmbedder = DenseEmbedder.subclass_from_config(
            ObjectConstructionConfig.model_validate(embedder_config)
        )

        return cls(**config,default_options = options,embedder = embedder)
    
class VectorStoreWithEmbedder(VectorStore[VectorStoreOptionsT]):
    """
    Base class for vector stores that take either a dense or sparse embedder as an argument.
    This class is used for vector stores that can handle both types of embeddings.
    """

    def __init__(
        self,
        embedder: Embedder,
        embedding_type: EmbeddingType = EmbeddingType.TEXT,
        default_options: VectorStoreOptionsT | None = None,
    ) -> None:
        """
        Constructs a new VectorStore instance.

        Args:
            embedder: The embedder to use for converting entries to vectors. Can be a regular Embedder for dense vectors
                     or a SparseEmbedder for sparse vectors.
            embedding_type: Which part of the entry to embed, either text or image. The other part will be ignored.
            default_options: The default options for querying the vector store.
        """
        super().__init__(default_options=default_options)
        self._embedder = embedder
        self._embedding_type = embedding_type

        if self._embedding_type == EmbeddingType.IMAGE and not self._embedder.image_support():
            raise ValueError("Embedder does not support image embeddings")

    async def _create_embeddings(self, entries: list[VectorStoreEntry]) -> dict[UUID, list[float] | SparseVector]:
        """
        Create embeddings for the given entry, using the provided embedder and embedding type.

        Args:
            entries: The entries to create embeddings for.

        Returns:
            The embeddings mapped by entry ID. Returns either dense vectors as list[float] or
            sparse vectors as SparseVector depending on the type of embedder used.
        """
        if self._embedding_type == EmbeddingType.TEXT:
            entries = [e for e in entries if e.text is not None]
            embeddings = await self._embedder.embed_text([e.text for e in entries if e.text is not None])
            return {e.id: cast(SparseVector | list[float], v) for e, v in zip(entries, embeddings, strict=True)}
        elif self._embedding_type == EmbeddingType.IMAGE:
            entries = [e for e in entries if e.image_bytes is not None]
            embeddings = await self._embedder.embed_image([e.image_bytes for e in entries if e.image_bytes is not None])
            return {e.id: cast(SparseVector | list[float], v) for e, v in zip(entries, embeddings, strict=True)}
        else:
            raise ValueError(f"Unsupported embedding type: {self._embedding_type}")

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Initializes the class with the provided configuration.

        Args:
            config: A dictionary containing configuration details for the class.

        Returns:
            An instance of the class initialized with the provided configuration.
        """
        default_options = config.pop("default_options", None)
        options = cls.options_cls(**default_options) if default_options else None

        embedder_config = config.pop("embedder")
        embedder: Embedder = Embedder.subclass_from_config(ObjectConstructionConfig.model_validate(embedder_config))

        return cls(**config, default_options=options, embedder=embedder)






