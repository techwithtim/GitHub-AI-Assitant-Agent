# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from astrapy.constants import DocumentType, SortType, VectorType
from astrapy.results import (
    BulkWriteResult,
    DeleteResult,
    InsertManyResult,
    InsertOneResult,
    UpdateResult,
)
from astrapy.collection import AsyncCollection, Collection


def reduce_bulk_write_results(results: List[BulkWriteResult]) -> BulkWriteResult:
    """
    Reduce a list of bulk write results into a single one.

    Args:
        results: a list of BulkWriteResult instances.

    Returns:
        A new BulkWRiteResult object which summarized the whole input list.
    """

    zero = BulkWriteResult.zero()

    def _sum_results(r1: BulkWriteResult, r2: BulkWriteResult) -> BulkWriteResult:
        bulk_api_results = {**r1.bulk_api_results, **r2.bulk_api_results}
        if r1.deleted_count is None or r2.deleted_count is None:
            deleted_count = None
        else:
            deleted_count = r1.deleted_count + r2.deleted_count
        inserted_count = r1.inserted_count + r2.inserted_count
        matched_count = r1.matched_count + r2.matched_count
        modified_count = r1.modified_count + r2.modified_count
        upserted_count = r1.upserted_count + r2.upserted_count
        upserted_ids = {**r1.upserted_ids, **r2.upserted_ids}
        return BulkWriteResult(
            bulk_api_results=bulk_api_results,
            deleted_count=deleted_count,
            inserted_count=inserted_count,
            matched_count=matched_count,
            modified_count=modified_count,
            upserted_count=upserted_count,
            upserted_ids=upserted_ids,
        )

    return reduce(_sum_results, results, zero)


class BaseOperation(ABC):
    """
    Base class for all operations amenable to be used
    in bulk writes on (sync) collections.
    """

    @abstractmethod
    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult: ...


@dataclass
class InsertOne(BaseOperation):
    """
    Represents an `insert_one` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        document: the document to insert.
        vector: an optional suitable vector to enrich the document at insertion.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
    """

    document: DocumentType
    vector: Optional[VectorType]
    vectorize: Optional[str]

    def __init__(
        self,
        document: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
    ) -> None:
        self.document = document
        self.vector = vector
        self.vectorize = vectorize

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: InsertOneResult = collection.insert_one(
            document=self.document,
            vector=self.vector,
            vectorize=self.vectorize,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class InsertMany(BaseOperation):
    """
    Represents an `insert_many` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        documents: the list document to insert.
        vectors: an optional list of vectors to enrich the documents at insertion.
        vectorize: an optional list of texts achieving the same effect as `vectors`
            except through an embedding service, if one is configured for the collection.
            NOTE: This feature is under current development.
        ordered: whether the inserts should be done in sequence.
        chunk_size: how many documents to include in a single API request.
            Exceeding the server maximum allowed value results in an error.
            Leave it unspecified (recommended) to use the system default.
        concurrency: maximum number of concurrent requests to the API at
            a given time. It cannot be more than one for ordered insertions.
    """

    documents: Iterable[DocumentType]
    vectors: Optional[Iterable[Optional[VectorType]]]
    vectorize: Optional[Iterable[Optional[str]]]
    ordered: bool
    chunk_size: Optional[int]
    concurrency: Optional[int]

    def __init__(
        self,
        documents: Iterable[DocumentType],
        *,
        vectors: Optional[Iterable[Optional[VectorType]]] = None,
        vectorize: Optional[Iterable[Optional[str]]] = None,
        ordered: bool = True,
        chunk_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> None:
        self.documents = documents
        self.ordered = ordered
        self.vectors = vectors
        self.vectorize = vectorize
        self.chunk_size = chunk_size
        self.concurrency = concurrency

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: InsertManyResult = collection.insert_many(
            documents=self.documents,
            vectors=self.vectors,
            vectorize=self.vectorize,
            ordered=self.ordered,
            chunk_size=self.chunk_size,
            concurrency=self.concurrency,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class UpdateOne(BaseOperation):
    """
    Represents an `update_one` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        update: an update prescription to apply to the document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    update: Dict[str, Any]
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.update = update
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort
        self.upsert = upsert

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = collection.update_one(
            filter=self.filter,
            update=self.update,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class UpdateMany(BaseOperation):
    """
    Represents an `update_many` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select target documents.
        update: an update prescription to apply to the documents.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    update: Dict[str, Any]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.update = update
        self.upsert = upsert

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = collection.update_many(
            filter=self.filter,
            update=self.update,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class ReplaceOne(BaseOperation):
    """
    Represents a `replace_one` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        replacement: the replacement document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    replacement: DocumentType
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.replacement = replacement
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort
        self.upsert = upsert

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = collection.replace_one(
            filter=self.filter,
            replacement=self.replacement,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class DeleteOne(BaseOperation):
    """
    Represents a `delete_one` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
    """

    filter: Dict[str, Any]
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]

    def __init__(
        self,
        filter: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
    ) -> None:
        self.filter = filter
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: DeleteResult = collection.delete_one(
            filter=self.filter,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class DeleteMany(BaseOperation):
    """
    Represents a `delete_many` operation on a (sync) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select target documents.
    """

    filter: Dict[str, Any]

    def __init__(
        self,
        filter: Dict[str, Any],
    ) -> None:
        self.filter = filter

    def execute(
        self,
        collection: Collection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: DeleteResult = collection.delete_many(
            filter=self.filter, max_time_ms=bulk_write_timeout_ms
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


class AsyncBaseOperation(ABC):
    """
    Base class for all operations amenable to be used
    in bulk writes on (async) collections.
    """

    @abstractmethod
    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult: ...


@dataclass
class AsyncInsertOne(AsyncBaseOperation):
    """
    Represents an `insert_one` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        document: the document to insert.
        vector: an optional suitable vector to enrich the document at insertion.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
    """

    document: DocumentType
    vector: Optional[VectorType]
    vectorize: Optional[str]

    def __init__(
        self,
        document: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
    ) -> None:
        self.document = document
        self.vector = vector
        self.vectorize = vectorize

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: InsertOneResult = await collection.insert_one(
            document=self.document,
            vector=self.vector,
            vectorize=self.vectorize,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncInsertMany(AsyncBaseOperation):
    """
    Represents an `insert_many` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        documents: the list document to insert.
        vectors: an optional list of vectors to enrich the documents at insertion.
        vectorize: an optional list of texts achieving the same effect as `vectors`
            except through an embedding service, if one is configured for the collection.
            NOTE: This feature is under current development.
        ordered: whether the inserts should be done in sequence.
        chunk_size: how many documents to include in a single API request.
            Exceeding the server maximum allowed value results in an error.
            Leave it unspecified (recommended) to use the system default.
        concurrency: maximum number of concurrent requests to the API at
            a given time. It cannot be more than one for ordered insertions.
    """

    documents: Iterable[DocumentType]
    vectors: Optional[Iterable[Optional[VectorType]]]
    vectorize: Optional[Iterable[Optional[str]]]
    ordered: bool
    chunk_size: Optional[int]
    concurrency: Optional[int]

    def __init__(
        self,
        documents: Iterable[DocumentType],
        *,
        vectors: Optional[Iterable[Optional[VectorType]]] = None,
        vectorize: Optional[Iterable[Optional[str]]] = None,
        ordered: bool = True,
        chunk_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self.vectorize = vectorize
        self.ordered = ordered
        self.chunk_size = chunk_size
        self.concurrency = concurrency

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: InsertManyResult = await collection.insert_many(
            documents=self.documents,
            vectors=self.vectors,
            vectorize=self.vectorize,
            ordered=self.ordered,
            chunk_size=self.chunk_size,
            concurrency=self.concurrency,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncUpdateOne(AsyncBaseOperation):
    """
    Represents an `update_one` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        update: an update prescription to apply to the document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    update: Dict[str, Any]
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.update = update
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort
        self.upsert = upsert

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = await collection.update_one(
            filter=self.filter,
            update=self.update,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncUpdateMany(AsyncBaseOperation):
    """
    Represents an `update_many` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select target documents.
        update: an update prescription to apply to the documents.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    update: Dict[str, Any]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.update = update
        self.upsert = upsert

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = await collection.update_many(
            filter=self.filter,
            update=self.update,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncReplaceOne(AsyncBaseOperation):
    """
    Represents a `replace_one` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        replacement: the replacement document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
        upsert: controls what to do when no documents are found.
    """

    filter: Dict[str, Any]
    replacement: DocumentType
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]
    upsert: bool

    def __init__(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
    ) -> None:
        self.filter = filter
        self.replacement = replacement
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort
        self.upsert = upsert

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: UpdateResult = await collection.replace_one(
            filter=self.filter,
            replacement=self.replacement,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            upsert=self.upsert,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncDeleteOne(AsyncBaseOperation):
    """
    Represents a `delete_one` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select a target document.
        vector: a vector of numbers to use for ANN (vector-search) sorting.
        vectorize: a string to be made into a vector, with the same result as the
            `vector` attribute, through an embedding service, assuming one is
            configured for the collection.
            NOTE: This feature is under current development.
        sort: controls ordering of results, hence which document is affected.
    """

    filter: Dict[str, Any]
    vector: Optional[VectorType]
    vectorize: Optional[str]
    sort: Optional[SortType]

    def __init__(
        self,
        filter: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
    ) -> None:
        self.filter = filter
        self.vector = vector
        self.vectorize = vectorize
        self.sort = sort

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: DeleteResult = await collection.delete_one(
            filter=self.filter,
            vector=self.vector,
            vectorize=self.vectorize,
            sort=self.sort,
            max_time_ms=bulk_write_timeout_ms,
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)


@dataclass
class AsyncDeleteMany(AsyncBaseOperation):
    """
    Represents a `delete_many` operation on a (async) collection.
    See the documentation on the collection method for more information.

    Attributes:
        filter: a filter condition to select target documents.
    """

    filter: Dict[str, Any]

    def __init__(
        self,
        filter: Dict[str, Any],
    ) -> None:
        self.filter = filter

    async def execute(
        self,
        collection: AsyncCollection,
        index_in_bulk_write: int,
        bulk_write_timeout_ms: Optional[int],
    ) -> BulkWriteResult:
        """
        Execute this operation against a collection as part of a bulk write.

        Args:
            collection: the collection this write targets.
            insert_in_bulk_write: the index in the list of bulkoperations
        """

        op_result: DeleteResult = await collection.delete_many(
            filter=self.filter, max_time_ms=bulk_write_timeout_ms
        )
        return op_result.to_bulk_write_result(index_in_bulk_write=index_in_bulk_write)
