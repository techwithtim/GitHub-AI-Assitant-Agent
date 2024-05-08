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

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

from astrapy.core.db import (
    AstraDBCollection,
    AsyncAstraDBCollection,
)
from astrapy.core.defaults import MAX_INSERT_NUM_DOCUMENTS
from astrapy.exceptions import (
    BulkWriteException,
    CollectionNotFoundException,
    CumulativeOperationException,
    DataAPIFaultyResponseException,
    DataAPIResponseException,
    DeleteManyException,
    InsertManyException,
    MultiCallTimeoutManager,
    TooManyDocumentsToCountException,
    UpdateManyException,
    recast_method_sync,
    recast_method_async,
    base_timeout_info,
)
from astrapy.constants import (
    DocumentType,
    FilterType,
    ProjectionType,
    ReturnDocument,
    SortType,
    VectorType,
    normalize_optional_projection,
)
from astrapy.database import AsyncDatabase, Database
from astrapy.results import (
    DeleteResult,
    InsertManyResult,
    InsertOneResult,
    UpdateResult,
    BulkWriteResult,
)
from astrapy.cursors import AsyncCursor, Cursor
from astrapy.info import CollectionInfo, CollectionOptions


if TYPE_CHECKING:
    from astrapy.operations import AsyncBaseOperation, BaseOperation


logger = logging.getLogger(__name__)


DEFAULT_INSERT_MANY_CONCURRENCY = 20
DEFAULT_BULK_WRITE_CONCURRENCY = 10


def _prepare_update_info(statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
    reduced_status = {
        "matchedCount": sum(
            status["matchedCount"] for status in statuses if "matchedCount" in status
        ),
        "modifiedCount": sum(
            status["modifiedCount"] for status in statuses if "modifiedCount" in status
        ),
        "upsertedId": [
            status["upsertedId"] for status in statuses if "upsertedId" in status
        ],
    }
    if reduced_status["upsertedId"]:
        if len(reduced_status["upsertedId"]) == 1:
            ups_dict = {"upserted": reduced_status["upsertedId"][0]}
        else:
            ups_dict = {"upserteds": reduced_status["upsertedId"]}
    else:
        ups_dict = {}
    return {
        **{
            "n": reduced_status["matchedCount"] + len(reduced_status["upsertedId"]),
            "updatedExisting": reduced_status["modifiedCount"] > 0,
            "ok": 1.0,
            "nModified": reduced_status["modifiedCount"],
        },
        **ups_dict,
    }


def _collate_vector_to_sort(
    sort: Optional[SortType],
    vector: Optional[VectorType],
    vectorize: Optional[str],
) -> Optional[SortType]:
    _vsort: Dict[str, Any]
    if vector is None:
        if vectorize is None:
            return sort
        else:
            _vsort = {"$vectorize": vectorize}
            if sort is None:
                return _vsort
            else:
                raise ValueError(
                    "The `vectorize` and `sort` clauses are mutually exclusive."
                )
    else:
        if vectorize is None:
            _vsort = {"$vector": vector}
            if sort is None:
                return _vsort
            else:
                raise ValueError(
                    "The `vector` and `sort` clauses are mutually exclusive."
                )
        else:
            raise ValueError(
                "The `vector` and `vectorize` parameters cannot be passed at the same time."
            )


def _is_vector_sort(sort: Optional[SortType]) -> bool:
    if sort is None:
        return False
    else:
        return "$vector" in sort or "$vectorize" in sort


def _collate_vector_to_document(
    document0: DocumentType, vector: Optional[VectorType], vectorize: Optional[str]
) -> DocumentType:
    if vector is None:
        if vectorize is None:
            return document0
        else:
            if "$vectorize" in document0:
                raise ValueError(
                    "Cannot specify the `vectorize` separately for a document with "
                    "its '$vectorize' field already."
                )
            else:
                return {
                    **document0,
                    **{"$vectorize": vectorize},
                }
    else:
        if vectorize is None:
            if "$vector" in document0:
                raise ValueError(
                    "Cannot specify the `vector` separately for a document with "
                    "its '$vector' field already."
                )
            else:
                return {
                    **document0,
                    **{"$vector": vector},
                }
        else:
            raise ValueError(
                "The `vector` and `vectorize` parameters cannot be passed at the same time."
            )


def _collate_vectors_to_documents(
    documents: Iterable[DocumentType],
    vectors: Optional[Iterable[Optional[VectorType]]],
    vectorize: Optional[Iterable[Optional[str]]],
) -> List[DocumentType]:
    if vectors is None and vectorize is None:
        return list(documents)
    else:
        _documents = list(documents)
        _ndocs = len(_documents)
        _vectors = list(vectors) if vectors else [None] * _ndocs
        _vectorize = list(vectorize) if vectorize else [None] * _ndocs
        if _ndocs != len(_vectors):
            raise ValueError(
                "The `documents` and `vectors` parameters must have the same length"
            )
        elif _ndocs != len(_vectorize):
            raise ValueError(
                "The `documents` and `vectorize` parameters must have the same length"
            )
        return [
            _collate_vector_to_document(_doc, _vec, _vecize)
            for _doc, _vec, _vecize in zip(_documents, _vectors, _vectorize)
        ]


class Collection:
    """
    A Data API collection, the main object to interact with the Data API,
    especially for DDL operations.
    This class has a synchronous interface.

    A Collection is spawned from a Database object, from which it inherits
    the details on how to reach the API server (endpoint, authentication token).

    Args:
        database: a Database object, instantiated earlier. This represents
            the database the collection belongs to.
        name: the collection name. This parameter should match an existing
            collection on the database.
        namespace: this is the namespace to which the collection belongs.
            If not specified, the database's working namespace is used.
        caller_name: name of the application, or framework, on behalf of which
            the Data API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.

    Examples:
        >>> from astrapy import DataAPIClient, Collection
        >>> my_client = astrapy.DataAPIClient("AstraCS:...")
        >>> my_db = my_client.get_database_by_api_endpoint(
        ...    "https://01234567-....apps.astra.datastax.com"
        ... )
        >>> my_coll_1 = Collection(database=my_db, name="my_collection")
        >>> my_coll_2 = my_db.create_collection(
        ...     "my_v_collection",
        ...     dimension=3,
        ...     metric="cosine",
        ... )
        >>> my_coll_3a = my_db.get_collection("my_already_existing_collection")
        >>> my_coll_3b = my_db.my_already_existing_collection
        >>> my_coll_3c = my_db["my_already_existing_collection"]

    Note:
        creating an instance of Collection does not trigger actual creation
        of the collection on the database. The latter should have been created
        beforehand, e.g. through the `create_collection` method of a Database.
    """

    def __init__(
        self,
        database: Database,
        name: str,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self._astra_db_collection: AstraDBCollection = AstraDBCollection(
            collection_name=name,
            astra_db=database._astra_db,
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )
        # this comes after the above, lets AstraDBCollection resolve namespace
        self._database = database._copy(
            namespace=self._astra_db_collection.astra_db.namespace
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(name="{self.name}", '
            f'namespace="{self.namespace}", database={self.database})'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Collection):
            return self._astra_db_collection == other._astra_db_collection
        else:
            return False

    def __call__(self, *pargs: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"'{self.__class__.__name__}' object is not callable. If you "
            f"meant to call the '{self.name}' method on a "
            f"'{self.database.__class__.__name__}' object "
            "it is failing because no such method exists."
        )

    def _copy(
        self,
        *,
        database: Optional[Database] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> Collection:
        return Collection(
            database=database or self.database._copy(),
            name=name or self.name,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self._astra_db_collection.caller_name,
            caller_version=caller_version or self._astra_db_collection.caller_version,
        )

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> Collection:
        """
        Create a clone of this collection with some changed attributes.

        Args:
            name: the name of the collection. This parameter is useful to
                quickly spawn Collection instances each pointing to a different
                collection existing in the same namespace.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new Collection instance.

        Example:
            >>> my_other_coll = my_coll.with_options(
            ...     name="the_other_coll",
            ...     caller_name="caller_identity",
            ... )
        """

        return self._copy(
            name=name,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def to_async(
        self,
        *,
        database: Optional[AsyncDatabase] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncCollection:
        """
        Create an AsyncCollection from this one. Save for the arguments
        explicitly provided as overrides, everything else is kept identical
        to this collection in the copy (the database is converted into
        an async object).

        Args:
            database: an AsyncDatabase object, instantiated earlier.
                This represents the database the new collection belongs to.
            name: the collection name. This parameter should match an existing
                collection on the database.
            namespace: this is the namespace to which the collection belongs.
                If not specified, the database's working namespace is used.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            the new copy, an AsyncCollection instance.

        Example:
            >>> asyncio.run(my_coll.to_async().count_documents({},upper_bound=100))
            77
        """

        return AsyncCollection(
            database=database or self.database.to_async(),
            name=name or self.name,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self._astra_db_collection.caller_name,
            caller_version=caller_version or self._astra_db_collection.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Set a new identity for the application/framework on behalf of which
        the Data API calls are performed (the "caller").

        Args:
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Example:
            >>> my_coll.set_caller(caller_name="the_caller", caller_version="0.1.0")
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._astra_db_collection.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def options(self, *, max_time_ms: Optional[int] = None) -> CollectionOptions:
        """
        Get the collection options, i.e. its configuration as read from the database.

        The method issues a request to the Data API each time is invoked,
        without caching mechanisms: this ensures up-to-date information
        for usages such as real-time collection validation by the application.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a CollectionOptions instance describing the collection.
            (See also the database `list_collections` method.)

        Example:
            >>> my_coll.options()
            CollectionOptions(vector=CollectionVectorOptions(dimension=3, metric='cosine'))
        """

        logger.info(f"getting collections in search of '{self.name}'")
        self_descriptors = [
            coll_desc
            for coll_desc in self.database.list_collections(max_time_ms=max_time_ms)
            if coll_desc.name == self.name
        ]
        logger.info(f"finished getting collections in search of '{self.name}'")
        if self_descriptors:
            return self_descriptors[0].options  # type: ignore[no-any-return]
        else:
            raise CollectionNotFoundException(
                text=f"Collection {self.namespace}.{self.name} not found.",
                namespace=self.namespace,
                collection_name=self.name,
            )

    def info(self) -> CollectionInfo:
        """
        Information on the collection (name, location, database), in the
        form of a CollectionInfo object.

        Not to be confused with the collection `options` method (related
        to the collection internal configuration).

        Example:
            >>> my_coll.info().database_info.region
            'eu-west-1'
            >>> my_coll.info().full_name
            'default_keyspace.my_v_collection'

        Note:
            the returned CollectionInfo wraps, among other things,
            the database information: as such, calling this method
            triggers the same-named method of a Database object (which, in turn,
            performs a HTTP request to the DevOps API).
            See the documentation for `Database.info()` for more details.
        """

        return CollectionInfo(
            database_info=self.database.info(),
            namespace=self.namespace,
            name=self.name,
            full_name=self.full_name,
        )

    @property
    def database(self) -> Database:
        """
        a Database object, the database this collection belongs to.

        Example:
            >>> my_coll.database.name
            'the_application_database'
        """

        return self._database

    @property
    def namespace(self) -> str:
        """
        The namespace this collection is in.

        Example:
            >>> my_coll.namespace
            'default_keyspace'
        """

        return self.database.namespace

    @property
    def name(self) -> str:
        """
        The name of this collection.

        Example:
            >>> my_coll.name
            'my_v_collection'
        """

        # type hint added as for some reason the typechecker gets lost
        return self._astra_db_collection.collection_name  # type: ignore[no-any-return, has-type]

    @property
    def full_name(self) -> str:
        """
        The fully-qualified collection name within the database,
        in the form "namespace.collection_name".

        Example:
            >>> my_coll.full_name
            'default_keyspace.my_v_collection'
        """

        return f"{self.namespace}.{self.name}"

    @recast_method_sync
    def insert_one(
        self,
        document: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> InsertOneResult:
        """
        Insert a single document in the collection in an atomic operation.

        Args:
            document: the dictionary expressing the document to insert.
                The `_id` field of the document can be left out, in which
                case it will be created automatically.
            vector: a vector (a list of numbers appropriate for the collection)
                for the document. Passing this parameter is equivalent to
                providing the vector in the "$vector" field of the document itself,
                however the two are mutually exclusive.
            vectorize: a string to be made into a vector, if such a service
                is configured for the collection. Passing this parameter is
                equivalent to providing a `$vectorize` field in the document itself,
                however the two are mutually exclusive.
                Moreover, this parameter cannot coexist with `vector`.
                NOTE: This feature is under current development.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an InsertOneResult object.

        Examples:
            >>> my_coll.count_documents({}, upper_bound=10)
            0
            >>> my_coll.insert_one(
            ...     {
            ...         "age": 30,
            ...         "name": "Smith",
            ...         "food": ["pear", "peach"],
            ...         "likes_fruit": True,
            ...     },
            ... )
            InsertOneResult(raw_results=..., inserted_id='ed4587a4-...-...-...')
            >>> my_coll.insert_one({"_id": "user-123", "age": 50, "name": "Maccio"})
            InsertOneResult(raw_results=..., inserted_id='user-123')
            >>> my_coll.count_documents({}, upper_bound=10)
            2

            >>> my_coll.insert_one({"tag": v"}, vector=[10, 11])
            InsertOneResult(...)

        Note:
            If an `_id` is explicitly provided, which corresponds to a document
            that exists already in the collection, an error is raised and
            the insertion fails.
        """

        _document = _collate_vector_to_document(document, vector, vectorize)
        logger.info(f"inserting one document in '{self.name}'")
        io_response = self._astra_db_collection.insert_one(
            _document,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished inserting one document in '{self.name}'")
        if "insertedIds" in io_response.get("status", {}):
            if io_response["status"]["insertedIds"]:
                inserted_id = io_response["status"]["insertedIds"][0]
                return InsertOneResult(
                    raw_results=[io_response],
                    inserted_id=inserted_id,
                )
            else:
                raise DataAPIFaultyResponseException(
                    text="Faulty response from insert_one API command.",
                    raw_response=io_response,
                )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from insert_one API command.",
                raw_response=io_response,
            )

    @recast_method_sync
    def insert_many(
        self,
        documents: Iterable[DocumentType],
        *,
        vectors: Optional[Iterable[Optional[VectorType]]] = None,
        vectorize: Optional[Iterable[Optional[str]]] = None,
        ordered: bool = True,
        chunk_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        max_time_ms: Optional[int] = None,
    ) -> InsertManyResult:
        """
        Insert a list of documents into the collection.
        This is not an atomic operation.

        Args:
            documents: an iterable of dictionaries, each a document to insert.
                Documents may specify their `_id` field or leave it out, in which
                case it will be added automatically.
            vectors: an optional list of vectors (as many vectors as the provided
                documents) to associate to the documents when inserting.
                Each vector is added to the corresponding document prior to
                insertion on database. The list can be a mixture of None and vectors,
                in which case some documents will not have a vector, unless it is
                specified in their "$vector" field already.
                Passing vectors this way is indeed equivalent to the "$vector" field
                of the documents, however the two are mutually exclusive.
            vectorize: an optional list of strings to be made into as many vectors
                (one per document), if such a service is configured for the collection.
                Passing this parameter is equivalent to providing a `$vectorize`
                field in the documents themselves, however the two are mutually exclusive.
                For any given document, this parameter cannot coexist with the
                corresponding `vector` entry.
                NOTE: This feature is under current development.
            ordered: if True (default), the insertions are processed sequentially.
                If False, they can occur in arbitrary order and possibly concurrently.
            chunk_size: how many documents to include in a single API request.
                Exceeding the server maximum allowed value results in an error.
                Leave it unspecified (recommended) to use the system default.
            concurrency: maximum number of concurrent requests to the API at
                a given time. It cannot be more than one for ordered insertions.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            an InsertManyResult object.

        Examples:
            >>> my_coll.count_documents({}, upper_bound=10)
            0
            >>> my_coll.insert_many([{"a": 10}, {"a": 5}, {"b": [True, False, False]}])
            InsertManyResult(raw_results=..., inserted_ids=['184bb06f-...', '...', '...'])
            >>> my_coll.count_documents({}, upper_bound=100)
            3
            >>> my_coll.insert_many(
            ...     [{"seq": i} for i in range(50)],
            ...     ordered=False,
            ...     concurrency=5,
            ... )
            InsertManyResult(raw_results=..., inserted_ids=[... ...])
            >>> my_coll.count_documents({}, upper_bound=100)
            53

            # The following are three equivalent statements:
            >>> my_coll.insert_many(
            ...     [{"tag": "a"}, {"tag": "b"}],
            ...     vectors=[[1, 2], [3, 4]],
            ... )
            InsertManyResult(...)
            >>> my_coll.insert_many(
            ...     [{"tag": "a", "$vector": [1, 2]}, {"tag": "b"}],
            ...     vectors=[None, [3, 4]],
            ... )
            InsertManyResult(...)
            >>> my_coll.insert_many(
            ...     [
            ...         {"tag": "a", "$vector": [1, 2]},
            ...         {"tag": "b", "$vector": [3, 4]},
            ...     ]
            ... )
            InsertManyResult(...)

        Note:
            Unordered insertions are executed with some degree of concurrency,
            so it is usually better to prefer this mode unless the order in the
            document sequence is important.

        Note:
            A failure mode for this command is related to certain faulty documents
            found among those to insert: a document may have the an `_id` already
            present on the collection, or its vector dimension may not
            match the collection setting.

            For an ordered insertion, the method will raise an exception at
            the first such faulty document -- nevertheless, all documents processed
            until then will end up being written to the database.

            For unordered insertions, if the error stems from faulty documents
            the insertion proceeds until exhausting the input documents: then,
            an exception is raised -- and all insertable documents will have been
            written to the database, including those "after" the troublesome ones.

            If, on the other hand, there are errors not related to individual
            documents (such as a network connectivity error), the whole
            `insert_many` operation will stop in mid-way, an exception will be raised,
            and only a certain amount of the input documents will
            have made their way to the database.
        """

        if concurrency is None:
            if ordered:
                _concurrency = 1
            else:
                _concurrency = DEFAULT_INSERT_MANY_CONCURRENCY
        else:
            _concurrency = concurrency
        if _concurrency > 1 and ordered:
            raise ValueError("Cannot run ordered insert_many concurrently.")
        if chunk_size is None:
            _chunk_size = MAX_INSERT_NUM_DOCUMENTS
        else:
            _chunk_size = chunk_size
        _documents = _collate_vectors_to_documents(documents, vectors, vectorize)
        logger.info(f"inserting {len(_documents)} documents in '{self.name}'")
        raw_results: List[Dict[str, Any]] = []
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        if ordered:
            options = {"ordered": True}
            inserted_ids: List[Any] = []
            for i in range(0, len(_documents), _chunk_size):
                logger.info(f"inserting a chunk of documents in '{self.name}'")
                chunk_response = self._astra_db_collection.insert_many(
                    documents=_documents[i : i + _chunk_size],
                    options=options,
                    partial_failures_allowed=True,
                    timeout_info=timeout_manager.remaining_timeout_info(),
                )
                logger.info(f"finished inserting a chunk of documents in '{self.name}'")
                # accumulate the results in this call
                chunk_inserted_ids = (chunk_response.get("status") or {}).get(
                    "insertedIds", []
                )
                inserted_ids += chunk_inserted_ids
                raw_results += [chunk_response]
                # if errors, quit early
                if chunk_response.get("errors", []):
                    partial_result = InsertManyResult(
                        raw_results=raw_results,
                        inserted_ids=inserted_ids,
                    )
                    raise InsertManyException.from_response(
                        command=None,
                        raw_response=chunk_response,
                        partial_result=partial_result,
                    )

            # return
            full_result = InsertManyResult(
                raw_results=raw_results,
                inserted_ids=inserted_ids,
            )
            logger.info(
                f"finished inserting {len(_documents)} documents in '{self.name}'"
            )
            return full_result

        else:
            # unordered: concurrent or not, do all of them and parse the results
            options = {"ordered": False}
            if _concurrency > 1:
                with ThreadPoolExecutor(max_workers=_concurrency) as executor:

                    def _chunk_insertor(
                        document_chunk: List[Dict[str, Any]]
                    ) -> Dict[str, Any]:
                        logger.info(f"inserting a chunk of documents in '{self.name}'")
                        im_response = self._astra_db_collection.insert_many(
                            documents=document_chunk,
                            options=options,
                            partial_failures_allowed=True,
                            timeout_info=timeout_manager.remaining_timeout_info(),
                        )
                        logger.info(
                            f"finished inserting a chunk of documents in '{self.name}'"
                        )
                        return im_response

                    raw_results = list(
                        executor.map(
                            _chunk_insertor,
                            (
                                _documents[i : i + _chunk_size]
                                for i in range(0, len(_documents), _chunk_size)
                            ),
                        )
                    )
            else:
                for i in range(0, len(_documents), _chunk_size):
                    logger.info(f"inserting a chunk of documents in '{self.name}'")
                    raw_results.append(
                        self._astra_db_collection.insert_many(
                            _documents[i : i + _chunk_size],
                            options=options,
                            partial_failures_allowed=True,
                            timeout_info=timeout_manager.remaining_timeout_info(),
                        )
                    )
                    logger.info(
                        f"finished inserting a chunk of documents in '{self.name}'"
                    )
            # recast raw_results
            inserted_ids = [
                inserted_id
                for chunk_response in raw_results
                for inserted_id in (chunk_response.get("status") or {}).get(
                    "insertedIds", []
                )
            ]

            # check-raise
            if any(
                [chunk_response.get("errors", []) for chunk_response in raw_results]
            ):
                partial_result = InsertManyResult(
                    raw_results=raw_results,
                    inserted_ids=inserted_ids,
                )
                raise InsertManyException.from_responses(
                    commands=[None for _ in raw_results],
                    raw_responses=raw_results,
                    partial_result=partial_result,
                )

            # return
            full_result = InsertManyResult(
                raw_results=raw_results,
                inserted_ids=inserted_ids,
            )
            logger.info(
                f"finished inserting {len(_documents)} documents in '{self.name}'"
            )
            return full_result

    def find(
        self,
        filter: Optional[FilterType] = None,
        *,
        projection: Optional[ProjectionType] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        include_similarity: Optional[bool] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> Cursor:
        """
        Find documents on the collection, matching a certain provided filter.

        The method returns a Cursor that can then be iterated over. Depending
        on the method call pattern, the iteration over all documents can reflect
        collection mutations occurred since the `find` method was called, or not.
        In cases where the cursor reflects mutations in real-time, it will iterate
        over cursors in an approximate way (i.e. exhibiting occasional skipped
        or duplicate documents). This happens when making use of the `sort`
        option in a non-vector-search manner.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the documents being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            skip: with this integer parameter, what would be the first `skip`
                documents returned by the query are discarded, and the results
                start from the (skip+1)-th document.
                This parameter can be used only in conjunction with an explicit
                `sort` criterion of the ascending/descending type (i.e. it cannot
                be used when not sorting, nor with vector-based ANN search).
            limit: this (integer) parameter sets a limit over how many documents
                are returned. Once `limit` is reached (or the cursor is exhausted
                for lack of matching documents), nothing more is returned.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to perform vector search (i.e. ANN,
                or "approximate nearest-neighbours" search).
                When running similarity search on a collection, no other sorting
                criteria can be specified. Moreover, there is an upper bound
                to the number of documents that can be returned. For details,
                see the Note about upper bounds and the Data API documentation.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            include_similarity: a boolean to request the numeric value of the
                similarity to be returned as an added "$similarity" key in each
                returned document. Can only be used for vector ANN search, i.e.
                when either `vector` is supplied or the `sort` parameter has the
                shape {"$vector": ...}.
            sort: with this dictionary parameter one can control the order
                the documents are returned. See the Note about sorting, as well as
                the one about upper bounds, for details.
            max_time_ms: a timeout, in milliseconds, for each single one
                of the underlying HTTP requests used to fetch documents as the
                cursor is iterated over.

        Returns:
            a Cursor object representing iterations over the matching documents
            (see the Cursor object for how to use it. The simplest thing is to
            run a for loop: `for document in collection.sort(...):`).

        Examples:
            >>> filter = {"seq": {"$exists": True}}
            >>> for doc in my_coll.find(filter, projection={"seq": True}, limit=5):
            ...     print(doc["seq"])
            ...
            37
            35
            10
            36
            27
            >>> cursor1 = my_coll.find(
            ...     {},
            ...     limit=4,
            ...     sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ... )
            >>> [doc["_id"] for doc in cursor1]
            ['97e85f81-...', '1581efe4-...', '...', '...']
            >>> cursor2 = my_coll.find({}, limit=3)
            >>> cursor2.distinct("seq")
            [37, 35, 10]

            >>> my_coll.insert_many([
            ...     {"tag": "A", "$vector": [4, 5]},
            ...     {"tag": "B", "$vector": [3, 4]},
            ...     {"tag": "C", "$vector": [3, 2]},
            ...     {"tag": "D", "$vector": [4, 1]},
            ...     {"tag": "E", "$vector": [2, 5]},
            ... ])
            >>> ann_tags = [
            ...     document["tag"]
            ...     for document in my_coll.find(
            ...         {},
            ...         limit=3,
            ...         vector=[3, 3],
            ...     )
            ... ]
            >>> ann_tags
            ['A', 'B', 'C']
            # (assuming the collection has metric VectorMetric.COSINE)

        Note:
            The following are example values for the `sort` parameter.
            When no particular order is required:
                sort={}  # (default when parameter not provided)
            When sorting by a certain value in ascending/descending order:
                sort={"field": SortDocuments.ASCENDING}
                sort={"field": SortDocuments.DESCENDING}
            When sorting first by "field" and then by "subfield"
            (while modern Python versions preserve the order of dictionaries,
            it is suggested for clarity to employ a `collections.OrderedDict`
            in these cases):
                sort={
                    "field": SortDocuments.ASCENDING,
                    "subfield": SortDocuments.ASCENDING,
                }
            When running a vector similarity (ANN) search:
                sort={"$vector": [0.4, 0.15, -0.5]}

        Note:
            Some combinations of arguments impose an implicit upper bound on the
            number of documents that are returned by the Data API. More specifically:
            (a) Vector ANN searches cannot return more than a number of documents
            that at the time of writing is set to 1000 items.
            (b) When using a sort criterion of the ascending/descending type,
            the Data API will return a smaller number of documents, set to 20
            at the time of writing, and stop there. The returned documents are
            the top results across the whole collection according to the requested
            criterion.
            These provisions should be kept in mind even when subsequently running
            a command such as `.distinct()` on a cursor.

        Note:
            When not specifying sorting criteria at all (by vector or otherwise),
            the cursor can scroll through an arbitrary number of documents as
            the Data API and the client periodically exchange new chunks of documents.
            It should be noted that the behavior of the cursor in the case documents
            have been added/removed after the `find` was started depends on database
            internals and it is not guaranteed, nor excluded, that such "real-time"
            changes in the data would be picked up by the cursor.
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        if include_similarity is not None and not _is_vector_sort(_sort):
            raise ValueError(
                "Cannot use `include_similarity` when not searching through `vector`."
            )
        return (
            Cursor(
                collection=self,
                filter=filter,
                projection=projection,
                max_time_ms=max_time_ms,
                overall_max_time_ms=None,
            )
            .skip(skip)
            .limit(limit)
            .sort(_sort)
            .include_similarity(include_similarity)
        )

    def find_one(
        self,
        filter: Optional[FilterType] = None,
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        include_similarity: Optional[bool] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Run a search, returning the first document in the collection that matches
        provided filters, if any is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the documents being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to perform vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), extracting the most
                similar document in the collection matching the filter.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            include_similarity: a boolean to request the numeric value of the
                similarity to be returned as an added "$similarity" key in the
                returned document. Can only be used for vector ANN search, i.e.
                when either `vector` is supplied or the `sort` parameter has the
                shape {"$vector": ...}.
            sort: with this dictionary parameter one can control the order
                the documents are returned. See the Note about sorting for details.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary expressing the required document, otherwise None.

        Examples:
            >>> my_coll.find_one({})
            {'_id': '68d1e515-...', 'seq': 37}
            >>> my_coll.find_one({"seq": 10})
            {'_id': 'd560e217-...', 'seq': 10}
            >>> my_coll.find_one({"seq": 1011})
            >>> # (returns None for no matches)
            >>> my_coll.find_one({}, projection={"seq": False})
            {'_id': '68d1e515-...'}
            >>> my_coll.find_one(
            ...     {},
            ...     sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ... )
            {'_id': '97e85f81-...', 'seq': 69}
            >>> my_coll.find_one({}, vector=[1, 0])
            {'_id': '...', 'tag': 'D', '$vector': [4.0, 1.0]}

        Note:
            See the `find` method for more details on the accepted parameters
            (whereas `skip` and `limit` are not valid parameters for `find_one`).
        """

        fo_cursor = self.find(
            filter=filter,
            projection=projection,
            skip=None,
            limit=1,
            vector=vector,
            vectorize=vectorize,
            include_similarity=include_similarity,
            sort=sort,
            max_time_ms=max_time_ms,
        )
        try:
            document = fo_cursor.__next__()
            return document  # type: ignore[no-any-return]
        except StopIteration:
            return None

    def distinct(
        self,
        key: str,
        *,
        filter: Optional[FilterType] = None,
        max_time_ms: Optional[int] = None,
    ) -> List[Any]:
        """
        Return a list of the unique values of `key` across the documents
        in the collection that match the provided filter.

        Args:
            key: the name of the field whose value is inspected across documents.
                Keys can use dot-notation to descend to deeper document levels.
                Example of acceptable `key` values:
                    "field"
                    "field.subfield"
                    "field.3"
                    "field.3.subfield"
                If lists are encountered and no numeric index is specified,
                all items in the list are visited.
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            a list of all different values for `key` found across the documents
            that match the filter. The result list has no repeated items.

        Example:
            >>> my_coll.insert_many(
            ...     [
            ...         {"name": "Marco", "food": ["apple", "orange"], "city": "Helsinki"},
            ...         {"name": "Emma", "food": {"likes_fruit": True, "allergies": []}},
            ...     ]
            ... )
            InsertManyResult(raw_results=..., inserted_ids=['c5b99f37-...', 'd6416321-...'])
            >>> my_coll.distinct("name")
            ['Marco', 'Emma']
            >>> my_coll.distinct("city")
            ['Helsinki']
            >>> my_coll.distinct("food")
            ['apple', 'orange', {'likes_fruit': True, 'allergies': []}]
            >>> my_coll.distinct("food.1")
            ['orange']
            >>> my_coll.distinct("food.allergies")
            []
            >>> my_coll.distinct("food.likes_fruit")
            [True]

        Note:
            It must be kept in mind that `distinct` is a client-side operation,
            which effectively browses all required documents using the logic
            of the `find` method and collects the unique values found for `key`.
            As such, there may be performance, latency and ultimately
            billing implications if the amount of matching documents is large.

        Note:
            For details on the behaviour of "distinct" in conjunction with
            real-time changes in the collection contents, see the
            Note of the `find` command.
        """

        f_cursor = Cursor(
            collection=self,
            filter=filter,
            projection={key: True},
            max_time_ms=None,
            overall_max_time_ms=max_time_ms,
        )
        return f_cursor.distinct(key)  # type: ignore[no-any-return]

    @recast_method_sync
    def count_documents(
        self,
        filter: Dict[str, Any],
        *,
        upper_bound: int,
        max_time_ms: Optional[int] = None,
    ) -> int:
        """
        Count the documents in the collection matching the specified filter.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            upper_bound: a required ceiling on the result of the count operation.
                If the actual number of documents exceeds this value,
                an exception will be raised.
                Furthermore, if the actual number of documents exceeds the maximum
                count that the Data API can reach (regardless of upper_bound),
                an exception will be raised.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            the exact count of matching documents.

        Example:
            >>> my_coll.insert_many([{"seq": i} for i in range(20)])
            InsertManyResult(...)
            >>> my_coll.count_documents({}, upper_bound=100)
            20
            >>> my_coll.count_documents({"seq":{"$gt": 15}}, upper_bound=100)
            4
            >>> my_coll.count_documents({}, upper_bound=10)
            Traceback (most recent call last):
                ... ...
            astrapy.exceptions.TooManyDocumentsToCountException

        Note:
            Count operations are expensive: for this reason, the best practice
            is to provide a reasonable `upper_bound` according to the caller
            expectations. Moreover, indiscriminate usage of count operations
            for sizeable amounts of documents (i.e. in the thousands and more)
            is discouraged in favor of alternative application-specific solutions.
            Keep in mind that the Data API has a hard upper limit on the amount
            of documents it will count, and that an exception will be thrown
            by this method if this limit is encountered.
        """

        logger.info("calling count_documents")
        cd_response = self._astra_db_collection.count_documents(
            filter=filter,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info("finished calling count_documents")
        if "count" in cd_response.get("status", {}):
            count: int = cd_response["status"]["count"]
            if cd_response["status"].get("moreData", False):
                raise TooManyDocumentsToCountException(
                    text=f"Document count exceeds {count}, the maximum allowed by the server",
                    server_max_count_exceeded=True,
                )
            else:
                if count > upper_bound:
                    raise TooManyDocumentsToCountException(
                        text="Document count exceeds required upper bound",
                        server_max_count_exceeded=False,
                    )
                else:
                    return count
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from count_documents API command.",
                raw_response=cd_response,
            )

    def estimated_document_count(
        self,
        *,
        max_time_ms: Optional[int] = None,
    ) -> int:
        """
        Query the API server for an estimate of the document count in the collection.

        Contrary to `count_documents`, this method has no filtering parameters.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a server-provided estimate count of the documents in the collection.

        Example:
            >>> my_coll.estimated_document_count()
            35700
        """
        ed_response = self.command(
            {"estimatedDocumentCount": {}},
            max_time_ms=max_time_ms,
        )
        if "count" in ed_response.get("status", {}):
            count: int = ed_response["status"]["count"]
            return count
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from estimated_document_count API command.",
                raw_response=ed_response,
            )

    @recast_method_sync
    def find_one_and_replace(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        return_document: str = ReturnDocument.BEFORE,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document on the collection and replace it entirely with a new one,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            replacement: the new document to write into the collection.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                replaced one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, `replacement` is inserted as a new document
                if no matches are found on the collection. If False,
                the operation silently does nothing in case of no matches.
            return_document: a flag controlling what document is returned:
                if set to `ReturnDocument.BEFORE`, or the string "before",
                the document found on database is returned; if set to
                `ReturnDocument.AFTER`, or the string "after", the new
                document is returned. The default is "before".
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            A document (or a projection thereof, as required), either the one
            before the replace operation or the one after that.
            Alternatively, the method returns None to represent
            that no matching document was found, or that no replacement
            was inserted (depending on the `return_document` parameter).

        Example:
            >>> my_coll.insert_one({"_id": "rule1", "text": "all animals are equal"})
            InsertOneResult(...)
            >>> my_coll.find_one_and_replace(
            ...     {"_id": "rule1"},
            ...     {"text": "some animals are more equal!"},
            ... )
            {'_id': 'rule1', 'text': 'all animals are equal'}
            >>> my_coll.find_one_and_replace(
            ...     {"text": "some animals are more equal!"},
            ...     {"text": "and the pigs are the rulers"},
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ... )
            {'_id': 'rule1', 'text': 'and the pigs are the rulers'}
            >>> my_coll.find_one_and_replace(
            ...     {"_id": "rule2"},
            ...     {"text": "F=ma^2"},
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ... )
            >>> # (returns None for no matches)
            >>> my_coll.find_one_and_replace(
            ...     {"_id": "rule2"},
            ...     {"text": "F=ma"},
            ...     upsert=True,
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ...     projection={"_id": False},
            ... )
            {'text': 'F=ma'}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "returnDocument": return_document,
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_replace on '{self.name}'")
        fo_response = self._astra_db_collection.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            projection=normalize_optional_projection(projection),
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_replace on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            ret_document = fo_response.get("data", {}).get("document")
            if ret_document is None:
                return None
            else:
                return ret_document  # type: ignore[no-any-return]
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_replace API command.",
                raw_response=fo_response,
            )

    @recast_method_sync
    def replace_one(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Replace a single document on the collection with a new one,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            replacement: the new document to write into the collection.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                replaced one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, `replacement` is inserted as a new document
                if no matches are found on the collection. If False,
                the operation silently does nothing in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an UpdateResult object summarizing the outcome of the replace operation.

        Example:
            >>> my_coll.insert_one({"Marco": "Polo"})
            InsertOneResult(...)
            >>> my_coll.replace_one({"Marco": {"$exists": True}}, {"Buda": "Pest"})
            UpdateResult(raw_results=..., update_info={'n': 1, 'updatedExisting': True, 'ok': 1.0, 'nModified': 1})
            >>> my_coll.find_one({"Buda": "Pest"})
            {'_id': '8424905a-...', 'Buda': 'Pest'}
            >>> my_coll.replace_one({"Mirco": {"$exists": True}}, {"Oh": "yeah?"})
            UpdateResult(raw_results=..., update_info={'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0})
            >>> my_coll.replace_one({"Mirco": {"$exists": True}}, {"Oh": "yeah?"}, upsert=True)
            UpdateResult(raw_results=..., update_info={'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '931b47d6-...'})
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_replace on '{self.name}'")
        fo_response = self._astra_db_collection.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_replace on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            fo_status = fo_response.get("status") or {}
            _update_info = _prepare_update_info([fo_status])
            return UpdateResult(
                raw_results=[fo_response],
                update_info=_update_info,
            )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_replace API command.",
                raw_response=fo_response,
            )

    @recast_method_sync
    def find_one_and_update(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        return_document: str = ReturnDocument.BEFORE,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document on the collection and update it as requested,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the document, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                updated one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a new document (resulting from applying the `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            return_document: a flag controlling what document is returned:
                if set to `ReturnDocument.BEFORE`, or the string "before",
                the document found on database is returned; if set to
                `ReturnDocument.AFTER`, or the string "after", the new
                document is returned. The default is "before".
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            A document (or a projection thereof, as required), either the one
            before the replace operation or the one after that.
            Alternatively, the method returns None to represent
            that no matching document was found, or that no update
            was applied (depending on the `return_document` parameter).

        Example:
            >>> my_coll.insert_one({"Marco": "Polo"})
            InsertOneResult(...)
            >>> my_coll.find_one_and_update(
            ...     {"Marco": {"$exists": True}},
            ...     {"$set": {"title": "Mr."}},
            ... )
            {'_id': 'a80106f2-...', 'Marco': 'Polo'}
            >>> my_coll.find_one_and_update(
            ...     {"title": "Mr."},
            ...     {"$inc": {"rank": 3}},
            ...     projection=["title", "rank"],
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ... )
            {'_id': 'a80106f2-...', 'title': 'Mr.', 'rank': 3}
            >>> my_coll.find_one_and_update(
            ...     {"name": "Johnny"},
            ...     {"$set": {"rank": 0}},
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ... )
            >>> # (returns None for no matches)
            >>> my_coll.find_one_and_update(
            ...     {"name": "Johnny"},
            ...     {"$set": {"rank": 0}},
            ...     upsert=True,
            ...     return_document=astrapy.constants.ReturnDocument.AFTER,
            ... )
            {'_id': 'cb4ef2ab-...', 'name': 'Johnny', 'rank': 0}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "returnDocument": return_document,
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_update on '{self.name}'")
        fo_response = self._astra_db_collection.find_one_and_update(
            update=update,
            filter=filter,
            projection=normalize_optional_projection(projection),
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_update on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            ret_document = fo_response.get("data", {}).get("document")
            if ret_document is None:
                return None
            else:
                return ret_document  # type: ignore[no-any-return]
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_update API command.",
                raw_response=fo_response,
            )

    @recast_method_sync
    def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Update a single document on the collection as requested,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the document, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                updated one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a new document (resulting from applying the `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an UpdateResult object summarizing the outcome of the update operation.

        Example:
            >>> my_coll.insert_one({"Marco": "Polo"})
            InsertOneResult(...)
            >>> my_coll.update_one({"Marco": {"$exists": True}}, {"$inc": {"rank": 3}})
            UpdateResult(raw_results=..., update_info={'n': 1, 'updatedExisting': True, 'ok': 1.0, 'nModified': 1})
            >>> my_coll.update_one({"Mirko": {"$exists": True}}, {"$inc": {"rank": 3}})
            UpdateResult(raw_results=..., update_info={'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0})
            >>> my_coll.update_one({"Mirko": {"$exists": True}}, {"$inc": {"rank": 3}}, upsert=True)
            UpdateResult(raw_results=..., update_info={'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '2a45ff60-...'})
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_update on '{self.name}'")
        fo_response = self._astra_db_collection.find_one_and_update(
            update=update,
            sort=_sort,
            filter=filter,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_update on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            fo_status = fo_response.get("status") or {}
            _update_info = _prepare_update_info([fo_status])
            return UpdateResult(
                raw_results=[fo_response],
                update_info=_update_info,
            )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_update API command.",
                raw_response=fo_response,
            )

    @recast_method_sync
    def update_many(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Apply an update operations to all documents matching a condition,
        optionally inserting one documents in absence of matches.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the documents, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a single new document (resulting from applying `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            an UpdateResult object summarizing the outcome of the update operation.

        Example:
            >>> my_coll.insert_many([{"c": "red"}, {"c": "green"}, {"c": "blue"}])
            InsertManyResult(...)
            >>> my_coll.update_many({"c": {"$ne": "green"}}, {"$set": {"nongreen": True}})
            UpdateResult(raw_results=..., update_info={'n': 2, 'updatedExisting': True, 'ok': 1.0, 'nModified': 2})
            >>> my_coll.update_many({"c": "orange"}, {"$set": {"is_also_fruit": True}})
            UpdateResult(raw_results=..., update_info={'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0})
            >>> my_coll.update_many(
            ...     {"c": "orange"},
            ...     {"$set": {"is_also_fruit": True}},
            ...     upsert=True,
            ... )
            UpdateResult(raw_results=..., update_info={'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '46643050-...'})

        Note:
            Similarly to the case of `find` (see its docstring for more details),
            running this command while, at the same time, another process is
            inserting new documents which match the filter of the `update_many`
            can result in an unpredictable fraction of these documents being updated.
            In other words, it cannot be easily predicted whether a given
            newly-inserted document will be picked up by the update_many command or not.
        """

        base_options = {
            "upsert": upsert,
        }
        page_state_options: Dict[str, str] = {}
        um_responses: List[Dict[str, Any]] = []
        um_statuses: List[Dict[str, Any]] = []
        must_proceed = True
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"starting update_many on '{self.name}'")
        while must_proceed:
            options = {**base_options, **page_state_options}
            logger.info(f"calling update_many on '{self.name}'")
            this_um_response = self._astra_db_collection.update_many(
                update=update,
                filter=filter,
                options=options,
                timeout_info=timeout_manager.remaining_timeout_info(),
            )
            logger.info(f"finished calling update_many on '{self.name}'")
            this_um_status = this_um_response.get("status") or {}
            #
            # if errors, quit early
            if this_um_response.get("errors", []):
                partial_update_info = _prepare_update_info(um_statuses)
                partial_result = UpdateResult(
                    raw_results=um_responses,
                    update_info=partial_update_info,
                )
                all_um_responses = um_responses + [this_um_response]
                raise UpdateManyException.from_responses(
                    commands=[None for _ in all_um_responses],
                    raw_responses=all_um_responses,
                    partial_result=partial_result,
                )
            else:
                if "status" not in this_um_response:
                    raise DataAPIFaultyResponseException(
                        text="Faulty response from update_many API command.",
                        raw_response=this_um_response,
                    )
                um_responses.append(this_um_response)
                um_statuses.append(this_um_status)
                next_page_state = this_um_status.get("nextPageState")
                if next_page_state is not None:
                    must_proceed = True
                    page_state_options = {"pageState": next_page_state}
                else:
                    must_proceed = False
                    page_state_options = {}

        update_info = _prepare_update_info(um_statuses)
        logger.info(f"finished update_many on '{self.name}'")
        return UpdateResult(
            raw_results=um_responses,
            update_info=update_info,
        )

    @recast_method_sync
    def find_one_and_delete(
        self,
        filter: Dict[str, Any],
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document in the collection and delete it. The deleted document,
        however, is the return value of the method.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                Note that the `_id` field will be returned with the document
                in any case, regardless of what the provided `projection` requires.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                deleted one. See the `find` method for more on sorting.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            Either the document (or a projection thereof, as requested), or None
            if no matches were found in the first place.

        Example:
            >>> my_coll.insert_many(
            ...     [
            ...         {"species": "swan", "class": "Aves"},
            ...         {"species": "frog", "class": "Amphibia"},
            ...     ],
            ... )
            InsertManyResult(...)
            >>> my_coll.find_one_and_delete(
            ...     {"species": {"$ne": "frog"}},
            ...     projection=["species"],
            ... )
            {'_id': '5997fb48-...', 'species': 'swan'}
            >>> my_coll.find_one_and_delete({"species": {"$ne": "frog"}})
            >>> # (returns None for no matches)
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        _projection = normalize_optional_projection(projection)
        logger.info(f"calling find_one_and_delete on '{self.name}'")
        fo_response = self._astra_db_collection.find_one_and_delete(
            sort=_sort,
            filter=filter,
            projection=_projection,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_delete on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            document = fo_response["data"]["document"]
            return document  # type: ignore[no-any-return]
        else:
            deleted_count = fo_response.get("status", {}).get("deletedCount")
            if deleted_count == 0:
                return None
            else:
                raise DataAPIFaultyResponseException(
                    text="Faulty response from find_one_and_delete API command.",
                    raw_response=fo_response,
                )

    @recast_method_sync
    def delete_one(
        self,
        filter: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> DeleteResult:
        """
        Delete one document matching a provided filter.
        This method never deletes more than a single document, regardless
        of the number of matches to the provided filters.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                deleted one. See the `find` method for more on sorting.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a DeleteResult object summarizing the outcome of the delete operation.

        Example:
            >>> my_coll.insert_many([{"seq": 1}, {"seq": 0}, {"seq": 2}])
            InsertManyResult(...)
            >>> my_coll.delete_one({"seq": 1})
            DeleteResult(raw_results=..., deleted_count=1)
            >>> my_coll.distinct("seq")
            [0, 2]
            >>> my_coll.delete_one(
            ...     {"seq": {"$exists": True}},
            ...     sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ... )
            DeleteResult(raw_results=..., deleted_count=1)
            >>> my_coll.distinct("seq")
            [0]
            >>> my_coll.delete_one({"seq": 2})
            DeleteResult(raw_results=..., deleted_count=0)
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        logger.info(f"calling delete_one_by_predicate on '{self.name}'")
        do_response = self._astra_db_collection.delete_one_by_predicate(
            filter=filter, timeout_info=base_timeout_info(max_time_ms), sort=_sort
        )
        logger.info(f"finished calling delete_one_by_predicate on '{self.name}'")
        if "deletedCount" in do_response.get("status", {}):
            deleted_count = do_response["status"]["deletedCount"]
            if deleted_count == -1:
                return DeleteResult(
                    deleted_count=None,
                    raw_results=[do_response],
                )
            else:
                # expected a non-negative integer:
                return DeleteResult(
                    deleted_count=deleted_count,
                    raw_results=[do_response],
                )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from delete_one API command.",
                raw_response=do_response,
            )

    @recast_method_sync
    def delete_many(
        self,
        filter: Dict[str, Any],
        *,
        max_time_ms: Optional[int] = None,
    ) -> DeleteResult:
        """
        Delete all documents matching a provided filter.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
                The `delete_many` method does not accept an empty filter: see
                `delete_all` to completely erase all contents of a collection
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            a DeleteResult object summarizing the outcome of the delete operation.

        Example:
            >>> my_coll.insert_many([{"seq": 1}, {"seq": 0}, {"seq": 2}])
            InsertManyResult(...)
            >>> my_coll.delete_many({"seq": {"$lte": 1}})
            DeleteResult(raw_results=..., deleted_count=2)
            >>> my_coll.distinct("seq")
            [2]
            >>> my_coll.delete_many({"seq": {"$lte": 1}})
            DeleteResult(raw_results=..., deleted_count=0)

        Note:
            This operation is not atomic. Depending on the amount of matching
            documents, it can keep running (in a blocking way) for a macroscopic
            time. In that case, new documents that are meanwhile inserted
            (e.g. from another process/application) will be deleted during
            the execution of this method call until the collection is devoid
            of matches.
        """

        if not filter:
            raise ValueError(
                "The `filter` parameter to method `delete_many` cannot be "
                "empty. In order to completely clear the contents of a "
                "collection, please use the `delete_all` method."
            )

        dm_responses: List[Dict[str, Any]] = []
        deleted_count = 0
        must_proceed = True
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"starting delete_many on '{self.name}'")
        while must_proceed:
            logger.info(f"calling delete_many on '{self.name}'")
            this_dm_response = self._astra_db_collection.delete_many(
                filter=filter,
                skip_error_check=True,
                timeout_info=timeout_manager.remaining_timeout_info(),
            )
            logger.info(f"finished calling delete_many on '{self.name}'")
            # if errors, quit early
            if this_dm_response.get("errors", []):
                partial_result = DeleteResult(
                    deleted_count=deleted_count,
                    raw_results=dm_responses,
                )
                all_dm_responses = dm_responses + [this_dm_response]
                raise DeleteManyException.from_responses(
                    commands=[None for _ in all_dm_responses],
                    raw_responses=all_dm_responses,
                    partial_result=partial_result,
                )
            else:
                this_dc = this_dm_response.get("status", {}).get("deletedCount")
                if this_dc is None or this_dc < 0:
                    raise DataAPIFaultyResponseException(
                        text="Faulty response from delete_many API command.",
                        raw_response=this_dm_response,
                    )
                dm_responses.append(this_dm_response)
                deleted_count += this_dc
                must_proceed = this_dm_response.get("status", {}).get("moreData", False)

        logger.info(f"finished delete_many on '{self.name}'")
        return DeleteResult(
            deleted_count=deleted_count,
            raw_results=dm_responses,
        )

    @recast_method_sync
    def delete_all(self, *, max_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Delete all documents in a collection.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary of the form {"ok": 1} to signal successful deletion.

        Example:
            >>> my_coll.distinct("seq")
            [2, 1, 0]
            >>> my_coll.count_documents({}, upper_bound=100)
            4
            >>> my_coll.delete_all()
            {'ok': 1}
            >>> my_coll.count_documents({}, upper_bound=100)
            0

        Note:
            Use with caution.
        """

        logger.info(f"calling unfiltered delete_many on '{self.name}'")
        dm_response = self._astra_db_collection.delete_many(
            filter={}, timeout_info=base_timeout_info(max_time_ms)
        )
        logger.info(f"finished calling unfiltered delete_many on '{self.name}'")
        deleted_count = dm_response["status"]["deletedCount"]
        if deleted_count == -1:
            return {"ok": 1}
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from delete_many API command.",
                raw_response=dm_response,
            )

    def bulk_write(
        self,
        requests: Iterable[BaseOperation],
        *,
        ordered: bool = True,
        concurrency: Optional[int] = None,
        max_time_ms: Optional[int] = None,
    ) -> BulkWriteResult:
        """
        Execute an arbitrary amount of operations such as inserts, updates, deletes
        either sequentially or concurrently.

        This method does not execute atomically, i.e. individual operations are
        each performed in the same way as the corresponding collection method,
        and each one is a different and unrelated database mutation.

        Args:
            requests: an iterable over concrete subclasses of `BaseOperation`,
                such as `InsertMany` or `ReplaceOne`. Each such object
                represents an operation ready to be executed on a collection,
                and is instantiated by passing the same parameters as one
                would the corresponding collection method.
            ordered: whether to launch the `requests` one after the other or
                in arbitrary order, possibly in a concurrent fashion. For
                performance reasons, `ordered=False` should be preferred
                when compatible with the needs of the application flow.
            concurrency: maximum number of concurrent operations executing at
                a given time. It cannot be more than one for ordered bulk writes.
            max_time_ms: a timeout, in milliseconds, for the whole bulk write.
                Remember that, if the method call times out, then there's no
                guarantee about what portion of the bulk write has been received
                and successfully executed by the Data API.

        Returns:
            A single BulkWriteResult summarizing the whole list of requested
            operations. The keys in the map attributes of BulkWriteResult
            (when present) are the integer indices of the corresponding operation
            in the `requests` iterable.

        Example:
            >>> from astrapy.operations import InsertMany, ReplaceOne
            >>> op1 = InsertMany([{"a": 1}, {"a": 2}])
            >>> op2 = ReplaceOne({"z": 9}, replacement={"z": 9, "replaced": True}, upsert=True)
            >>> my_coll.bulk_write([op1, op2])
            BulkWriteResult(bulk_api_results={0: ..., 1: ...}, deleted_count=0, inserted_count=3, matched_count=0, modified_count=0, upserted_count=1, upserted_ids={1: '2addd676-...'})
            >>> my_coll.count_documents({}, upper_bound=100)
            3
            >>> my_coll.distinct("replaced")
            [True]
        """

        # lazy importing here against circular-import error
        from astrapy.operations import reduce_bulk_write_results

        if concurrency is None:
            if ordered:
                _concurrency = 1
            else:
                _concurrency = DEFAULT_BULK_WRITE_CONCURRENCY
        else:
            _concurrency = concurrency
        if _concurrency > 1 and ordered:
            raise ValueError("Cannot run ordered bulk_write concurrently.")
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"startng a bulk write on '{self.name}'")
        if ordered:
            bulk_write_results: List[BulkWriteResult] = []
            for operation_i, operation in enumerate(requests):
                try:
                    this_bw_result = operation.execute(
                        self,
                        index_in_bulk_write=operation_i,
                        bulk_write_timeout_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    bulk_write_results.append(this_bw_result)
                except CumulativeOperationException as exc:
                    partial_result = exc.partial_result
                    partial_bw_result = reduce_bulk_write_results(
                        bulk_write_results
                        + [
                            partial_result.to_bulk_write_result(
                                index_in_bulk_write=operation_i
                            )
                        ]
                    )
                    dar_exception = exc.data_api_response_exception()
                    raise BulkWriteException(
                        text=dar_exception.text,
                        error_descriptors=dar_exception.error_descriptors,
                        detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                        partial_result=partial_bw_result,
                        exceptions=[dar_exception],
                    )
                except DataAPIResponseException as exc:
                    # the cumulative exceptions, with their
                    # partially-done-info, are handled above:
                    # here it's just one-shot d.a.r. exceptions
                    partial_bw_result = reduce_bulk_write_results(bulk_write_results)
                    dar_exception = exc.data_api_response_exception()
                    raise BulkWriteException(
                        text=dar_exception.text,
                        error_descriptors=dar_exception.error_descriptors,
                        detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                        partial_result=partial_bw_result,
                        exceptions=[dar_exception],
                    )
            full_bw_result = reduce_bulk_write_results(bulk_write_results)
            logger.info(f"finished a bulk write on '{self.name}'")
            return full_bw_result
        else:

            def _execute_as_either(
                operation: BaseOperation, operation_i: int
            ) -> Tuple[Optional[BulkWriteResult], Optional[DataAPIResponseException]]:
                try:
                    ex_result = operation.execute(
                        self,
                        index_in_bulk_write=operation_i,
                        bulk_write_timeout_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    return (ex_result, None)
                except DataAPIResponseException as exc:
                    return (None, exc)

            with ThreadPoolExecutor(max_workers=_concurrency) as executor:
                bulk_write_either_futures = [
                    executor.submit(
                        _execute_as_either,
                        operation,
                        operation_i,
                    )
                    for operation_i, operation in enumerate(requests)
                ]
                bulk_write_either_results = [
                    bulk_write_either_future.result()
                    for bulk_write_either_future in bulk_write_either_futures
                ]
                # regroup
                bulk_write_successes = [
                    bwr for bwr, _ in bulk_write_either_results if bwr
                ]
                bulk_write_failures = [
                    bwf for _, bwf in bulk_write_either_results if bwf
                ]
                if bulk_write_failures:
                    # extract and cumulate
                    partial_results_from_failures = [
                        failure.partial_result.to_bulk_write_result(
                            index_in_bulk_write=operation_i
                        )
                        for failure in bulk_write_failures
                        if isinstance(failure, CumulativeOperationException)
                    ]
                    partial_bw_result = reduce_bulk_write_results(
                        bulk_write_successes + partial_results_from_failures
                    )
                    # raise and recast the first exception
                    all_dar_exceptions = [
                        bw_failure.data_api_response_exception()
                        for bw_failure in bulk_write_failures
                    ]
                    dar_exception = all_dar_exceptions[0]
                    raise BulkWriteException(
                        text=dar_exception.text,
                        error_descriptors=dar_exception.error_descriptors,
                        detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                        partial_result=partial_bw_result,
                        exceptions=all_dar_exceptions,
                    )
                else:
                    logger.info(f"finished a bulk write on '{self.name}'")
                    return reduce_bulk_write_results(bulk_write_successes)

    def drop(self, *, max_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Drop the collection, i.e. delete it from the database along with
        all the documents it contains.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.
                Remember there is not guarantee that a request that has
                timed out us not in fact honored.

        Returns:
            a dictionary of the form {"ok": 1} to signal successful deletion.

        Example:
            >>> my_coll.find_one({})
            {'_id': '...', 'a': 100}
            >>> my_coll.drop()
            {'ok': 1}
            >>> my_coll.find_one({})
            Traceback (most recent call last):
                ... ...
            astrapy.exceptions.DataAPIResponseException: Collection does not exist, collection name: my_collection

        Note:
            Use with caution.


        Note:
            Once the method succeeds, methods on this object can still be invoked:
            however, this hardly makes sense as the underlying actual collection
            is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased collection any further.

        Note:
            Once the method succeeds, methods on this object can still be invoked:
            however, this hardly makes sense as the underlying actual collection
            is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased collection any further.
        """

        logger.info(f"dropping collection '{self.name}' (self)")
        drop_result = self.database.drop_collection(self, max_time_ms=max_time_ms)
        logger.info(f"finished dropping collection '{self.name}' (self)")
        return drop_result  # type: ignore[no-any-return]

    def command(
        self,
        body: Dict[str, Any],
        *,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a POST request to the Data API for this collection with
        an arbitrary, caller-provided payload.

        Args:
            body: a JSON-serializable dictionary, the payload of the request.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary with the response of the HTTP request.

        Example:
            >>> my_coll.command({"countDocuments": {}})
            {'status': {'count': 123}}
        """

        logger.info(f"calling command on '{self.name}'")
        command_result = self.database.command(
            body=body,
            namespace=self.namespace,
            collection_name=self.name,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished calling command on '{self.name}'")
        return command_result  # type: ignore[no-any-return]


class AsyncCollection:
    """
    A Data API collection, the main object to interact with the Data API,
    especially for DDL operations.
    This class has a synchronous interface.

    A Collection is spawned from a Database object, from which it inherits
    the details on how to reach the API server (endpoint, authentication token).

    Args:
        database: a Database object, instantiated earlier. This represents
            the database the collection belongs to.
        name: the collection name. This parameter should match an existing
            collection on the database.
        namespace: this is the namespace to which the collection belongs.
            If not specified, the database's working namespace is used.
        caller_name: name of the application, or framework, on behalf of which
            the Data API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.

    Examples:
        >>> from astrapy import DataAPIClient, AsyncCollection
        >>> my_client = astrapy.DataAPIClient("AstraCS:...")
        >>> my_async_db = my_client.get_async_database_by_api_endpoint(
        ...    "https://01234567-....apps.astra.datastax.com"
        ... )
        >>> my_async_coll_1 = AsyncCollection(database=my_async_db, name="my_collection")
        >>> my_async coll_2 = asyncio.run(my_async_db.create_collection(
        ...     "my_v_collection",
        ...     dimension=3,
        ...     metric="cosine",
        ... ))
        >>> my_async_coll_3a = asyncio.run(my_async_db.get_collection(
        ...     "my_already_existing_collection",
        ... ))
        >>> my_async_coll_3b = my_async_db.my_already_existing_collection
        >>> my_async_coll_3c = my_async_db["my_already_existing_collection"]

    Note:
        creating an instance of Collection does not trigger actual creation
        of the collection on the database. The latter should have been created
        beforehand, e.g. through the `create_collection` method of a Database.
    """

    def __init__(
        self,
        database: AsyncDatabase,
        name: str,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self._astra_db_collection: AsyncAstraDBCollection = AsyncAstraDBCollection(
            collection_name=name,
            astra_db=database._astra_db,
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )
        # this comes after the above, lets AstraDBCollection resolve namespace
        self._database = database._copy(
            namespace=self._astra_db_collection.astra_db.namespace
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(name="{self.name}", '
            f'namespace="{self.namespace}", database={self.database})'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AsyncCollection):
            return self._astra_db_collection == other._astra_db_collection
        else:
            return False

    def __call__(self, *pargs: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"'{self.__class__.__name__}' object is not callable. If you "
            f"meant to call the '{self.name}' method on a "
            f"'{self.database.__class__.__name__}' object "
            "it is failing because no such method exists."
        )

    def _copy(
        self,
        *,
        database: Optional[AsyncDatabase] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncCollection:
        return AsyncCollection(
            database=database or self.database._copy(),
            name=name or self.name,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self._astra_db_collection.caller_name,
            caller_version=caller_version or self._astra_db_collection.caller_version,
        )

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncCollection:
        """
        Create a clone of this collection with some changed attributes.

        Args:
            name: the name of the collection. This parameter is useful to
                quickly spawn AsyncCollection instances each pointing to a different
                collection existing in the same namespace.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new AsyncCollection instance.

        Example:
            >>> my_other_async_coll = my_async_coll.with_options(
            ...     name="the_other_coll",
            ...     caller_name="caller_identity",
            ... )
        """

        return self._copy(
            name=name,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def to_sync(
        self,
        *,
        database: Optional[Database] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> Collection:
        """
        Create a Collection from this one. Save for the arguments
        explicitly provided as overrides, everything else is kept identical
        to this collection in the copy (the database is converted into
        a sync object).

        Args:
            database: a Database object, instantiated earlier.
                This represents the database the new collection belongs to.
            name: the collection name. This parameter should match an existing
                collection on the database.
            namespace: this is the namespace to which the collection belongs.
                If not specified, the database's working namespace is used.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            the new copy, a Collection instance.

        Example:
            >>> my_async_coll.to_sync().count_documents({}, upper_bound=100)
            77
        """

        return Collection(
            database=database or self.database.to_sync(),
            name=name or self.name,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self._astra_db_collection.caller_name,
            caller_version=caller_version or self._astra_db_collection.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Set a new identity for the application/framework on behalf of which
        the Data API calls are performed (the "caller").

        Args:
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Example:
            >>> my_coll.set_caller(caller_name="the_caller", caller_version="0.1.0")
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._astra_db_collection.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )

    async def options(self, *, max_time_ms: Optional[int] = None) -> CollectionOptions:
        """
        Get the collection options, i.e. its configuration as read from the database.

        The method issues a request to the Data API each time is invoked,
        without caching mechanisms: this ensures up-to-date information
        for usages such as real-time collection validation by the application.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a CollectionOptions instance describing the collection.
            (See also the database `list_collections` method.)

        Example:
            >>> asyncio.run(my_async_coll.options())
            CollectionOptions(vector=CollectionVectorOptions(dimension=3, metric='cosine'))
        """

        logger.info(f"getting collections in search of '{self.name}'")
        self_descriptors = [
            coll_desc
            async for coll_desc in self.database.list_collections(
                max_time_ms=max_time_ms
            )
            if coll_desc.name == self.name
        ]
        logger.info(f"finished getting collections in search of '{self.name}'")
        if self_descriptors:
            return self_descriptors[0].options  # type: ignore[no-any-return]
        else:
            raise CollectionNotFoundException(
                text=f"Collection {self.namespace}.{self.name} not found.",
                namespace=self.namespace,
                collection_name=self.name,
            )

    def info(self) -> CollectionInfo:
        """
        Information on the collection (name, location, database), in the
        form of a CollectionInfo object.

        Not to be confused with the collection `options` method (related
        to the collection internal configuration).

        Example:
            >>> my_async_coll.info().database_info.region
            'us-east1'
            >>> my_async_coll.info().full_name
            'default_keyspace.my_v_collection'

        Note:
            the returned CollectionInfo wraps, among other things,
            the database information: as such, calling this method
            triggers the same-named method of a Database object (which, in turn,
            performs a HTTP request to the DevOps API).
            See the documentation for `Database.info()` for more details.
        """

        return CollectionInfo(
            database_info=self.database.info(),
            namespace=self.namespace,
            name=self.name,
            full_name=self.full_name,
        )

    @property
    def database(self) -> AsyncDatabase:
        """
        a Database object, the database this collection belongs to.

        Example:
            >>> my_async_coll.database.name
            'quicktest'
        """

        return self._database

    @property
    def namespace(self) -> str:
        """
        The namespace this collection is in.

        Example:
            >>> my_async_coll.database.namespace
            'default_keyspace'
        """

        return self.database.namespace

    @property
    def name(self) -> str:
        """
        The name of this collection.

        Example:
            >>> my_async_coll.name
            'my_v_collection'
        """

        # type hint added as for some reason the typechecker gets lost
        return self._astra_db_collection.collection_name  # type: ignore[no-any-return, has-type]

    @property
    def full_name(self) -> str:
        """
        The fully-qualified collection name within the database,
        in the form "namespace.collection_name".

        Example:
            >>> my_async_coll.full_name
            'default_keyspace.my_v_collection'
        """

        return f"{self.namespace}.{self.name}"

    @recast_method_async
    async def insert_one(
        self,
        document: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> InsertOneResult:
        """
        Insert a single document in the collection in an atomic operation.

        Args:
            document: the dictionary expressing the document to insert.
                The `_id` field of the document can be left out, in which
                case it will be created automatically.
            vector: a vector (a list of numbers appropriate for the collection)
                for the document. Passing this parameter is equivalent to
                providing the vector in the "$vector" field of the document itself,
                however the two are mutually exclusive.
            vectorize: a string to be made into a vector, if such a service
                is configured for the collection. Passing this parameter is
                equivalent to providing a `$vectorize` field in the document itself,
                however the two are mutually exclusive.
                Moreover, this parameter cannot coexist with `vector`.
                NOTE: This feature is under current development.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an InsertOneResult object.

        Example:
            >>> async def write_and_count(acol: AsyncCollection) -> None:
            ...     count0 = await acol.count_documents({}, upper_bound=10)
            ...     print("count0", count0)
            ...     await acol.insert_one(
            ...         {
            ...             "age": 30,
            ...             "name": "Smith",
            ...             "food": ["pear", "peach"],
            ...             "likes_fruit": True,
            ...         },
            ...     )
            ...     await acol.insert_one({"_id": "user-123", "age": 50, "name": "Maccio"})
            ...     count1 = await acol.count_documents({}, upper_bound=10)
            ...     print("count1", count1)
            ...
            >>> asyncio.run(write_and_count(my_async_coll))
            count0 0
            count1 2

            >>> asyncio.run(my_async_coll.insert_one({"tag": v"}, vector=[10, 11]))
            InsertOneResult(...)

        Note:
            If an `_id` is explicitly provided, which corresponds to a document
            that exists already in the collection, an error is raised and
            the insertion fails.
        """

        _document = _collate_vector_to_document(document, vector, vectorize)
        logger.info(f"inserting one document in '{self.name}'")
        io_response = await self._astra_db_collection.insert_one(
            _document,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished inserting one document in '{self.name}'")
        if "insertedIds" in io_response.get("status", {}):
            if io_response["status"]["insertedIds"]:
                inserted_id = io_response["status"]["insertedIds"][0]
                return InsertOneResult(
                    raw_results=[io_response],
                    inserted_id=inserted_id,
                )
            else:
                raise ValueError(
                    "Could not complete a insert_one operation. "
                    f"(gotten '${json.dumps(io_response)}')"
                )
        else:
            raise ValueError(
                "Could not complete a insert_one operation. "
                f"(gotten '${json.dumps(io_response)}')"
            )

    @recast_method_async
    async def insert_many(
        self,
        documents: Iterable[DocumentType],
        *,
        vectors: Optional[Iterable[Optional[VectorType]]] = None,
        vectorize: Optional[Iterable[Optional[str]]] = None,
        ordered: bool = True,
        chunk_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        max_time_ms: Optional[int] = None,
    ) -> InsertManyResult:
        """
        Insert a list of documents into the collection.
        This is not an atomic operation.

        Args:
            documents: an iterable of dictionaries, each a document to insert.
                Documents may specify their `_id` field or leave it out, in which
                case it will be added automatically.
            vectors: an optional list of vectors (as many vectors as the provided
                documents) to associate to the documents when inserting.
                Each vector is added to the corresponding document prior to
                insertion on database. The list can be a mixture of None and vectors,
                in which case some documents will not have a vector, unless it is
                specified in their "$vector" field already.
                Passing vectors this way is indeed equivalent to the "$vector" field
                of the documents, however the two are mutually exclusive.
            vectorize: an optional list of strings to be made into as many vectors
                (one per document), if such a service is configured for the collection.
                Passing this parameter is equivalent to providing a `$vectorize`
                field in the documents themselves, however the two are mutually exclusive.
                For any given document, this parameter cannot coexist with the
                corresponding `vector` entry.
                NOTE: This feature is under current development.
            ordered: if True (default), the insertions are processed sequentially.
                If False, they can occur in arbitrary order and possibly concurrently.
            chunk_size: how many documents to include in a single API request.
                Exceeding the server maximum allowed value results in an error.
                Leave it unspecified (recommended) to use the system default.
            concurrency: maximum number of concurrent requests to the API at
                a given time. It cannot be more than one for ordered insertions.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            an InsertManyResult object.

        Examples:
            >>> async def write_and_count(acol: AsyncCollection) -> None:
            ...             count0 = await acol.count_documents({}, upper_bound=10)
            ...             print("count0", count0)
            ...             im_result1 = await acol.insert_many(
            ...                 [
            ...                     {"a": 10},
            ...                     {"a": 5},
            ...                     {"b": [True, False, False]},
            ...                 ],
            ...             )
            ...             print("inserted1", im_result1.inserted_ids)
            ...             count1 = await acol.count_documents({}, upper_bound=100)
            ...             print("count1", count1)
            ...             await acol.insert_many(
            ...                 [{"seq": i} for i in range(50)],
            ...                 ordered=False,
            ...                 concurrency=5,
            ...             )
            ...             count2 = await acol.count_documents({}, upper_bound=100)
            ...             print("count2", count2)
            ...
            >>> asyncio.run(write_and_count(my_async_coll))
            count0 0
            inserted1 ['e3c2a684-...', '1de4949f-...', '167dacc3-...']
            count1 3
            count2 53

            # The following are three equivalent statements:
            >>> asyncio.run(my_async_coll.insert_many(
            ...     [{"tag": "a"}, {"tag": "b"}],
            ...     vectors=[[1, 2], [3, 4]],
            ... ))
            InsertManyResult(...)
            >>> asyncio.run(my_async_coll.insert_many(
            ...     [{"tag": "a", "$vector": [1, 2]}, {"tag": "b"}],
            ...     vectors=[None, [3, 4]],
            ... ))
            InsertManyResult(...)
            >>> asyncio.run(my_async_coll.insert_many(
            ...     [
            ...         {"tag": "a", "$vector": [1, 2]},
            ...         {"tag": "b", "$vector": [3, 4]},
            ...     ]
            ... ))
            InsertManyResult(...)


        Note:
            Unordered insertions are executed with some degree of concurrency,
            so it is usually better to prefer this mode unless the order in the
            document sequence is important.

        Note:
            A failure mode for this command is related to certain faulty documents
            found among those to insert: a document may have the an `_id` already
            present on the collection, or its vector dimension may not
            match the collection setting.

            For an ordered insertion, the method will raise an exception at
            the first such faulty document -- nevertheless, all documents processed
            until then will end up being written to the database.

            For unordered insertions, if the error stems from faulty documents
            the insertion proceeds until exhausting the input documents: then,
            an exception is raised -- and all insertable documents will have been
            written to the database, including those "after" the troublesome ones.

            If, on the other hand, there are errors not related to individual
            documents (such as a network connectivity error), the whole
            `insert_many` operation will stop in mid-way, an exception will be raised,
            and only a certain amount of the input documents will
            have made their way to the database.
        """

        if concurrency is None:
            if ordered:
                _concurrency = 1
            else:
                _concurrency = DEFAULT_INSERT_MANY_CONCURRENCY
        else:
            _concurrency = concurrency
        if _concurrency > 1 and ordered:
            raise ValueError("Cannot run ordered insert_many concurrently.")
        if chunk_size is None:
            _chunk_size = MAX_INSERT_NUM_DOCUMENTS
        else:
            _chunk_size = chunk_size
        _documents = _collate_vectors_to_documents(documents, vectors, vectorize)
        logger.info(f"inserting {len(_documents)} documents in '{self.name}'")
        raw_results: List[Dict[str, Any]] = []
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        if ordered:
            options = {"ordered": True}
            inserted_ids: List[Any] = []
            for i in range(0, len(_documents), _chunk_size):
                logger.info(f"inserting a chunk of documents in '{self.name}'")
                chunk_response = await self._astra_db_collection.insert_many(
                    documents=_documents[i : i + _chunk_size],
                    options=options,
                    partial_failures_allowed=True,
                    timeout_info=timeout_manager.remaining_timeout_info(),
                )
                logger.info(f"finished inserting a chunk of documents in '{self.name}'")
                # accumulate the results in this call
                chunk_inserted_ids = (chunk_response.get("status") or {}).get(
                    "insertedIds", []
                )
                inserted_ids += chunk_inserted_ids
                raw_results += [chunk_response]
                # if errors, quit early
                if chunk_response.get("errors", []):
                    partial_result = InsertManyResult(
                        raw_results=raw_results,
                        inserted_ids=inserted_ids,
                    )
                    raise InsertManyException.from_response(
                        command=None,
                        raw_response=chunk_response,
                        partial_result=partial_result,
                    )

            # return
            full_result = InsertManyResult(
                raw_results=raw_results,
                inserted_ids=inserted_ids,
            )
            logger.info(
                f"finished inserting {len(_documents)} documents in '{self.name}'"
            )
            return full_result

        else:
            # unordered: concurrent or not, do all of them and parse the results
            options = {"ordered": False}

            sem = asyncio.Semaphore(_concurrency)

            async def concurrent_insert_chunk(
                document_chunk: List[DocumentType],
            ) -> Dict[str, Any]:
                async with sem:
                    logger.info(f"inserting a chunk of documents in '{self.name}'")
                    im_response = await self._astra_db_collection.insert_many(
                        document_chunk,
                        options=options,
                        partial_failures_allowed=True,
                        timeout_info=timeout_manager.remaining_timeout_info(),
                    )
                    logger.info(
                        f"finished inserting a chunk of documents in '{self.name}'"
                    )
                    return im_response

            if _concurrency > 1:
                tasks = [
                    asyncio.create_task(
                        concurrent_insert_chunk(_documents[i : i + _chunk_size])
                    )
                    for i in range(0, len(_documents), _chunk_size)
                ]
                raw_results = await asyncio.gather(*tasks)
            else:
                raw_results = [
                    await concurrent_insert_chunk(_documents[i : i + _chunk_size])
                    for i in range(0, len(_documents), _chunk_size)
                ]

            # recast raw_results
            inserted_ids = [
                inserted_id
                for chunk_response in raw_results
                for inserted_id in (chunk_response.get("status") or {}).get(
                    "insertedIds", []
                )
            ]

            # check-raise
            if any(
                [chunk_response.get("errors", []) for chunk_response in raw_results]
            ):
                partial_result = InsertManyResult(
                    raw_results=raw_results,
                    inserted_ids=inserted_ids,
                )
                raise InsertManyException.from_responses(
                    commands=[None for _ in raw_results],
                    raw_responses=raw_results,
                    partial_result=partial_result,
                )

            # return
            full_result = InsertManyResult(
                raw_results=raw_results,
                inserted_ids=inserted_ids,
            )
            logger.info(
                f"finished inserting {len(_documents)} documents in '{self.name}'"
            )
            return full_result

    def find(
        self,
        filter: Optional[FilterType] = None,
        *,
        projection: Optional[ProjectionType] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        include_similarity: Optional[bool] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> AsyncCursor:
        """
        Find documents on the collection, matching a certain provided filter.

        The method returns a Cursor that can then be iterated over. Depending
        on the method call pattern, the iteration over all documents can reflect
        collection mutations occurred since the `find` method was called, or not.
        In cases where the cursor reflects mutations in real-time, it will iterate
        over cursors in an approximate way (i.e. exhibiting occasional skipped
        or duplicate documents). This happens when making use of the `sort`
        option in a non-vector-search manner.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the documents being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            skip: with this integer parameter, what would be the first `skip`
                documents returned by the query are discarded, and the results
                start from the (skip+1)-th document.
                This parameter can be used only in conjunction with an explicit
                `sort` criterion of the ascending/descending type (i.e. it cannot
                be used when not sorting, nor with vector-based ANN search).
            limit: this (integer) parameter sets a limit over how many documents
                are returned. Once `limit` is reached (or the cursor is exhausted
                for lack of matching documents), nothing more is returned.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to perform vector search (i.e. ANN,
                or "approximate nearest-neighbours" search).
                When running similarity search on a collection, no other sorting
                criteria can be specified. Moreover, there is an upper bound
                to the number of documents that can be returned. For details,
                see the Note about upper bounds and the Data API documentation.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            include_similarity: a boolean to request the numeric value of the
                similarity to be returned as an added "$similarity" key in each
                returned document. Can only be used for vector ANN search, i.e.
                when either `vector` is supplied or the `sort` parameter has the
                shape {"$vector": ...}.
            sort: with this dictionary parameter one can control the order
                the documents are returned. See the Note about sorting, as well as
                the one about upper bounds, for details.
            max_time_ms: a timeout, in milliseconds, for each single one
                of the underlying HTTP requests used to fetch documents as the
                cursor is iterated over.

        Returns:
            an AsyncCursor object representing iterations over the matching documents
            (see the AsyncCursor object for how to use it. The simplest thing is to
            run a for loop: `for document in collection.sort(...):`).

        Examples:
            >>> async def run_finds(acol: AsyncCollection) -> None:
            ...             filter = {"seq": {"$exists": True}}
            ...             print("find results 1:")
            ...             async for doc in acol.find(filter, projection={"seq": True}, limit=5):
            ...                 print(doc["seq"])
            ...             async_cursor1 = acol.find(
            ...                 {},
            ...                 limit=4,
            ...                 sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ...             )
            ...             ids = [doc["_id"] async for doc in async_cursor1]
            ...             print("find results 2:", ids)
            ...             async_cursor2 = acol.find({}, limit=3)
            ...             seqs = await async_cursor2.distinct("seq")
            ...             print("distinct results 3:", seqs)
            ...
            >>> asyncio.run(run_finds(my_async_coll))
            find results 1:
            48
            35
            7
            11
            13
            find results 2: ['d656cd9d-...', '479c7ce8-...', '96dc87fd-...', '83f0a21f-...']
            distinct results 3: [48, 35, 7]

            >>> async def run_vector_finds(acol: AsyncCollection) -> None:
            ...     await acol.insert_many([
            ...         {"tag": "A", "$vector": [4, 5]},
            ...         {"tag": "B", "$vector": [3, 4]},
            ...         {"tag": "C", "$vector": [3, 2]},
            ...         {"tag": "D", "$vector": [4, 1]},
            ...         {"tag": "E", "$vector": [2, 5]},
            ...     ])
            ...     ann_tags = [
            ...         document["tag"]
            ...         async for document in acol.find(
            ...             {},
            ...             limit=3,
            ...             vector=[3, 3],
            ...         )
            ...     ]
            ...     return ann_tags
            ...
            >>> asyncio.run(run_vector_finds(my_async_coll))
            ['A', 'B', 'C']
            # (assuming the collection has metric VectorMetric.COSINE)

        Note:
            The following are example values for the `sort` parameter.
            When no particular order is required:
                sort={}
            When sorting by a certain value in ascending/descending order:
                sort={"field": SortDocuments.ASCENDING}
                sort={"field": SortDocuments.DESCENDING}
            When sorting first by "field" and then by "subfield"
            (while modern Python versions preserve the order of dictionaries,
            it is suggested for clarity to employ a `collections.OrderedDict`
            in these cases):
                sort={
                    "field": SortDocuments.ASCENDING,
                    "subfield": SortDocuments.ASCENDING,
                }
            When running a vector similarity (ANN) search:
                sort={"$vector": [0.4, 0.15, -0.5]}

        Note:
            Some combinations of arguments impose an implicit upper bound on the
            number of documents that are returned by the Data API. More specifically:
            (a) Vector ANN searches cannot return more than a number of documents
            that at the time of writing is set to 1000 items.
            (b) When using a sort criterion of the ascending/descending type,
            the Data API will return a smaller number of documents, set to 20
            at the time of writing, and stop there. The returned documents are
            the top results across the whole collection according to the requested
            criterion.
            These provisions should be kept in mind even when subsequently running
            a command such as `.distinct()` on a cursor.

        Note:
            When not specifying sorting criteria at all (by vector or otherwise),
            the cursor can scroll through an arbitrary number of documents as
            the Data API and the client periodically exchange new chunks of documents.
            It should be noted that the behavior of the cursor in the case documents
            have been added/removed after the `find` was started depends on database
            internals and it is not guaranteed, nor excluded, that such "real-time"
            changes in the data would be picked up by the cursor.
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        if include_similarity is not None and not _is_vector_sort(_sort):
            raise ValueError(
                "Cannot use `include_similarity` when not searching through `vector`."
            )
        return (
            AsyncCursor(
                collection=self,
                filter=filter,
                projection=projection,
                max_time_ms=max_time_ms,
                overall_max_time_ms=None,
            )
            .skip(skip)
            .limit(limit)
            .sort(_sort)
            .include_similarity(include_similarity)
        )

    async def find_one(
        self,
        filter: Optional[FilterType] = None,
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        include_similarity: Optional[bool] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Run a search, returning the first document in the collection that matches
        provided filters, if any is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the documents being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to perform vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), extracting the most
                similar document in the collection matching the filter.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            include_similarity: a boolean to request the numeric value of the
                similarity to be returned as an added "$similarity" key in the
                returned document. Can only be used for vector ANN search, i.e.
                when either `vector` is supplied or the `sort` parameter has the
                shape {"$vector": ...}.
            sort: with this dictionary parameter one can control the order
                the documents are returned. See the Note about sorting for details.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary expressing the required document, otherwise None.

        Example:
            >>> async def demo_find_one(acol: AsyncCollection) -> None:
            ....    print("Count:", await acol.count_documents({}, upper_bound=100))
            ...     result0 = await acol.find_one({})
            ...     print("result0", result0)
            ...     result1 = await acol.find_one({"seq": 10})
            ...     print("result1", result1)
            ...     result2 = await acol.find_one({"seq": 1011})
            ...     print("result2", result2)
            ...     result3 = await acol.find_one({}, projection={"seq": False})
            ...     print("result3", result3)
            ...     result4 = await acol.find_one(
            ...         {},
            ...         sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ...     )
            ...     print("result4", result4)
            ...
            >>>
            >>> asyncio.run(demo_find_one(my_async_coll))
            Count: 50
            result0 {'_id': '479c7ce8-...', 'seq': 48}
            result1 {'_id': '93e992c4-...', 'seq': 10}
            result2 None
            result3 {'_id': '479c7ce8-...'}
            result4 {'_id': 'd656cd9d-...', 'seq': 49}

            >>> asyncio.run(my_async_coll.find_one({}, vector=[1, 0]))
            {'_id': '...', 'tag': 'D', '$vector': [4.0, 1.0]}

        Note:
            See the `find` method for more details on the accepted parameters
            (whereas `skip` and `limit` are not valid parameters for `find_one`).
        """

        fo_cursor = self.find(
            filter=filter,
            projection=projection,
            skip=None,
            limit=1,
            vector=vector,
            vectorize=vectorize,
            include_similarity=include_similarity,
            sort=sort,
            max_time_ms=max_time_ms,
        )
        try:
            document = await fo_cursor.__anext__()
            return document  # type: ignore[no-any-return]
        except StopAsyncIteration:
            return None

    async def distinct(
        self,
        key: str,
        *,
        filter: Optional[FilterType] = None,
        max_time_ms: Optional[int] = None,
    ) -> List[Any]:
        """
        Return a list of the unique values of `key` across the documents
        in the collection that match the provided filter.

        Args:
            key: the name of the field whose value is inspected across documents.
                Keys can use dot-notation to descend to deeper document levels.
                Example of acceptable `key` values:
                    "field"
                    "field.subfield"
                    "field.3"
                    "field.3.subfield"
                If lists are encountered and no numeric index is specified,
                all items in the list are visited.
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            a list of all different values for `key` found across the documents
            that match the filter. The result list has no repeated items.

        Example:
            >>> async def run_distinct(acol: AsyncCollection) -> None:
            ...     await acol.insert_many(
            ...         [
            ...             {"name": "Marco", "food": ["apple", "orange"], "city": "Helsinki"},
            ...             {"name": "Emma", "food": {"likes_fruit": True, "allergies": []}},
            ...         ]
            ...     )
            ...     distinct0 = await acol.distinct("name")
            ...     print("distinct('name')", distinct0)
            ...     distinct1 = await acol.distinct("city")
            ...     print("distinct('city')", distinct1)
            ...     distinct2 = await acol.distinct("food")
            ...     print("distinct('food')", distinct2)
            ...     distinct3 = await acol.distinct("food.1")
            ...     print("distinct('food.1')", distinct3)
            ...     distinct4 = await acol.distinct("food.allergies")
            ...     print("distinct('food.allergies')", distinct4)
            ...     distinct5 = await acol.distinct("food.likes_fruit")
            ...     print("distinct('food.likes_fruit')", distinct5)
            ...
            >>> asyncio.run(run_distinct(my_async_coll))
            distinct('name') ['Emma', 'Marco']
            distinct('city') ['Helsinki']
            distinct('food') [{'likes_fruit': True, 'allergies': []}, 'apple', 'orange']
            distinct('food.1') ['orange']
            distinct('food.allergies') []
            distinct('food.likes_fruit') [True]

        Note:
            It must be kept in mind that `distinct` is a client-side operation,
            which effectively browses all required documents using the logic
            of the `find` method and collects the unique values found for `key`.
            As such, there may be performance, latency and ultimately
            billing implications if the amount of matching documents is large.

        Note:
            For details on the behaviour of "distinct" in conjunction with
            real-time changes in the collection contents, see the
            Note of the `find` command.
        """

        f_cursor = AsyncCursor(
            collection=self,
            filter=filter,
            projection={key: True},
            max_time_ms=None,
            overall_max_time_ms=max_time_ms,
        )
        return await f_cursor.distinct(key)  # type: ignore[no-any-return]

    @recast_method_async
    async def count_documents(
        self,
        filter: Dict[str, Any],
        *,
        upper_bound: int,
        max_time_ms: Optional[int] = None,
    ) -> int:
        """
        Count the documents in the collection matching the specified filter.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            upper_bound: a required ceiling on the result of the count operation.
                If the actual number of documents exceeds this value,
                an exception will be raised.
                Furthermore, if the actual number of documents exceeds the maximum
                count that the Data API can reach (regardless of upper_bound),
                an exception will be raised.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            the exact count of matching documents.

        Example:
            >>> async def do_count_docs(acol: AsyncCollection) -> None:
            ...     await acol.insert_many([{"seq": i} for i in range(20)])
            ...     count0 = await acol.count_documents({}, upper_bound=100)
            ...     print("count0", count0)
            ...     count1 = await acol.count_documents({"seq":{"$gt": 15}}, upper_bound=100)
            ...     print("count1", count1)
            ...     count2 = await acol.count_documents({}, upper_bound=10)
            ...     print("count2", count2)
            ...
            >>> asyncio.run(do_count_docs(my_async_coll))
            count0 20
            count1 4
            Traceback (most recent call last):
                ... ...
            astrapy.exceptions.TooManyDocumentsToCountException

        Note:
            Count operations are expensive: for this reason, the best practice
            is to provide a reasonable `upper_bound` according to the caller
            expectations. Moreover, indiscriminate usage of count operations
            for sizeable amounts of documents (i.e. in the thousands and more)
            is discouraged in favor of alternative application-specific solutions.
            Keep in mind that the Data API has a hard upper limit on the amount
            of documents it will count, and that an exception will be thrown
            by this method if this limit is encountered.
        """

        logger.info("calling count_documents")
        cd_response = await self._astra_db_collection.count_documents(
            filter=filter,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info("finished calling count_documents")
        if "count" in cd_response.get("status", {}):
            count: int = cd_response["status"]["count"]
            if cd_response["status"].get("moreData", False):
                raise TooManyDocumentsToCountException(
                    text=f"Document count exceeds {count}, the maximum allowed by the server",
                    server_max_count_exceeded=True,
                )
            else:
                if count > upper_bound:
                    raise TooManyDocumentsToCountException(
                        text="Document count exceeds required upper bound",
                        server_max_count_exceeded=False,
                    )
                else:
                    return count
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from count_documents API command.",
                raw_response=cd_response,
            )

    async def estimated_document_count(
        self,
        *,
        max_time_ms: Optional[int] = None,
    ) -> int:
        """
        Query the API server for an estimate of the document count in the collection.

        Contrary to `count_documents`, this method has no filtering parameters.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a server-provided estimate count of the documents in the collection.

        Example:
            >>> asyncio.run(my_async_coll.estimated_document_count())
            35700
        """
        ed_response = await self.command(
            {"estimatedDocumentCount": {}},
            max_time_ms=max_time_ms,
        )
        if "count" in ed_response.get("status", {}):
            count: int = ed_response["status"]["count"]
            return count
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from estimated_document_count API command.",
                raw_response=ed_response,
            )

    @recast_method_async
    async def find_one_and_replace(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        return_document: str = ReturnDocument.BEFORE,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document on the collection and replace it entirely with a new one,
        optionally inserting a new one if no match is found.

        Args:

            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            replacement: the new document to write into the collection.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                replaced one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, `replacement` is inserted as a new document
                if no matches are found on the collection. If False,
                the operation silently does nothing in case of no matches.
            return_document: a flag controlling what document is returned:
                if set to `ReturnDocument.BEFORE`, or the string "before",
                the document found on database is returned; if set to
                `ReturnDocument.AFTER`, or the string "after", the new
                document is returned. The default is "before".
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            A document, either the one before the replace operation or the
            one after that. Alternatively, the method returns None to represent
            that no matching document was found, or that no replacement
            was inserted (depending on the `return_document` parameter).

        Example:
            >>> async def do_find_one_and_replace(acol: AsyncCollection) -> None:
            ...             await acol.insert_one({"_id": "rule1", "text": "all animals are equal"})
            ...             result0 = await acol.find_one_and_replace(
            ...                 {"_id": "rule1"},
            ...                 {"text": "some animals are more equal!"},
            ...             )
            ...             print("result0", result0)
            ...             result1 = await acol.find_one_and_replace(
            ...                 {"text": "some animals are more equal!"},
            ...                 {"text": "and the pigs are the rulers"},
            ...                 return_document=astrapy.constants.ReturnDocument.AFTER,
            ...             )
            ...             print("result1", result1)
            ...             result2 = await acol.find_one_and_replace(
            ...                 {"_id": "rule2"},
            ...                 {"text": "F=ma^2"},
            ...                 return_document=astrapy.constants.ReturnDocument.AFTER,
            ...             )
            ...             print("result2", result2)
            ...             result3 = await acol.find_one_and_replace(
            ...                 {"_id": "rule2"},
            ...                 {"text": "F=ma"},
            ...                 upsert=True,
            ...                 return_document=astrapy.constants.ReturnDocument.AFTER,
            ...                 projection={"_id": False},
            ...             )
            ...             print("result3", result3)
            ...
            >>> asyncio.run(do_find_one_and_replace(my_async_coll))
            result0 {'_id': 'rule1', 'text': 'all animals are equal'}
            result1 {'_id': 'rule1', 'text': 'and the pigs are the rulers'}
            result2 None
            result3 {'text': 'F=ma'}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "returnDocument": return_document,
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_replace on '{self.name}'")
        fo_response = await self._astra_db_collection.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            projection=normalize_optional_projection(projection),
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_replace on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            ret_document = fo_response.get("data", {}).get("document")
            if ret_document is None:
                return None
            else:
                return ret_document  # type: ignore[no-any-return]
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_replace API command.",
                raw_response=fo_response,
            )

    @recast_method_async
    async def replace_one(
        self,
        filter: Dict[str, Any],
        replacement: DocumentType,
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Replace a single document on the collection with a new one,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            replacement: the new document to write into the collection.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                replaced one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, `replacement` is inserted as a new document
                if no matches are found on the collection. If False,
                the operation silently does nothing in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an UpdateResult object summarizing the outcome of the replace operation.

        Example:
            >>> async def do_replace_one(acol: AsyncCollection) -> None:
            ...     await acol.insert_one({"Marco": "Polo"})
            ...     result0 = await acol.replace_one(
            ...         {"Marco": {"$exists": True}},
            ...         {"Buda": "Pest"},
            ...     )
            ...     print("result0.update_info", result0.update_info)
            ...     doc1 = await acol.find_one({"Buda": "Pest"})
            ...     print("doc1", doc1)
            ...     result1 = await acol.replace_one(
            ...         {"Mirco": {"$exists": True}},
            ...         {"Oh": "yeah?"},
            ...     )
            ...     print("result1.update_info", result1.update_info)
            ...     result2 = await acol.replace_one(
            ...         {"Mirco": {"$exists": True}},
            ...         {"Oh": "yeah?"},
            ...         upsert=True,
            ...     )
            ...     print("result2.update_info", result2.update_info)
            ...
            >>> asyncio.run(do_replace_one(my_async_coll))
            result0.update_info {'n': 1, 'updatedExisting': True, 'ok': 1.0, 'nModified': 1}
            doc1 {'_id': '6e669a5a-...', 'Buda': 'Pest'}
            result1.update_info {'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0}
            result2.update_info {'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '30e34e00-...'}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_replace on '{self.name}'")
        fo_response = await self._astra_db_collection.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_replace on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            fo_status = fo_response.get("status") or {}
            _update_info = _prepare_update_info([fo_status])
            return UpdateResult(
                raw_results=[fo_response],
                update_info=_update_info,
            )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_replace API command.",
                raw_response=fo_response,
            )

    @recast_method_async
    async def find_one_and_update(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        return_document: str = ReturnDocument.BEFORE,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document on the collection and update it as requested,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the document, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                updated one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a new document (resulting from applying the `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            return_document: a flag controlling what document is returned:
                if set to `ReturnDocument.BEFORE`, or the string "before",
                the document found on database is returned; if set to
                `ReturnDocument.AFTER`, or the string "after", the new
                document is returned. The default is "before".
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            A document (or a projection thereof, as required), either the one
            before the replace operation or the one after that.
            Alternatively, the method returns None to represent
            that no matching document was found, or that no update
            was applied (depending on the `return_document` parameter).

        Example:
            >>> async def do_find_one_and_update(acol: AsyncCollection) -> None:
            ...     await acol.insert_one({"Marco": "Polo"})
            ...     result0 = await acol.find_one_and_update(
            ...         {"Marco": {"$exists": True}},
            ...         {"$set": {"title": "Mr."}},
            ...     )
            ...     print("result0", result0)
            ...     result1 = await acol.find_one_and_update(
            ...         {"title": "Mr."},
            ...         {"$inc": {"rank": 3}},
            ...         projection=["title", "rank"],
            ...         return_document=astrapy.constants.ReturnDocument.AFTER,
            ...     )
            ...     print("result1", result1)
            ...     result2 = await acol.find_one_and_update(
            ...         {"name": "Johnny"},
            ...         {"$set": {"rank": 0}},
            ...         return_document=astrapy.constants.ReturnDocument.AFTER,
            ...     )
            ...     print("result2", result2)
            ...     result3 = await acol.find_one_and_update(
            ...         {"name": "Johnny"},
            ...         {"$set": {"rank": 0}},
            ...         upsert=True,
            ...         return_document=astrapy.constants.ReturnDocument.AFTER,
            ...     )
            ...     print("result3", result3)
            ...
            >>> asyncio.run(do_find_one_and_update(my_async_coll))
            result0 {'_id': 'f7c936d3-b0a0-45eb-a676-e2829662a57c', 'Marco': 'Polo'}
            result1 {'_id': 'f7c936d3-b0a0-45eb-a676-e2829662a57c', 'title': 'Mr.', 'rank': 3}
            result2 None
            result3 {'_id': 'db3d678d-14d4-4caa-82d2-d5fb77dab7ec', 'name': 'Johnny', 'rank': 0}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "returnDocument": return_document,
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_update on '{self.name}'")
        fo_response = await self._astra_db_collection.find_one_and_update(
            update=update,
            filter=filter,
            projection=normalize_optional_projection(projection),
            sort=_sort,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_update on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            ret_document = fo_response.get("data", {}).get("document")
            if ret_document is None:
                return None
            else:
                return ret_document  # type: ignore[no-any-return]
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_update API command.",
                raw_response=fo_response,
            )

    @recast_method_async
    async def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Update a single document on the collection as requested,
        optionally inserting a new one if no match is found.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the document, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                updated one. See the `find` method for more on sorting.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a new document (resulting from applying the `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an UpdateResult object summarizing the outcome of the update operation.

        Example:
            >>> async def do_update_one(acol: AsyncCollection) -> None:
            ...     await acol.insert_one({"Marco": "Polo"})
            ...     result0 = await acol.update_one(
            ...         {"Marco": {"$exists": True}},
            ...         {"$inc": {"rank": 3}},
            ...     )
            ...     print("result0.update_info", result0.update_info)
            ...     result1 = await acol.update_one(
            ...         {"Mirko": {"$exists": True}},
            ...         {"$inc": {"rank": 3}},
            ...     )
            ...     print("result1.update_info", result1.update_info)
            ...     result2 = await acol.update_one(
            ...         {"Mirko": {"$exists": True}},
            ...         {"$inc": {"rank": 3}},
            ...         upsert=True,
            ...     )
            ...     print("result2.update_info", result2.update_info)
            ...
            >>> asyncio.run(do_update_one(my_async_coll))
            result0.update_info {'n': 1, 'updatedExisting': True, 'ok': 1.0, 'nModified': 1})
            result1.update_info {'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0})
            result2.update_info {'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '75748092-...'}
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        options = {
            "upsert": upsert,
        }
        logger.info(f"calling find_one_and_update on '{self.name}'")
        fo_response = await self._astra_db_collection.find_one_and_update(
            update=update,
            sort=_sort,
            filter=filter,
            options=options,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_update on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            fo_status = fo_response.get("status") or {}
            _update_info = _prepare_update_info([fo_status])
            return UpdateResult(
                raw_results=[fo_response],
                update_info=_update_info,
            )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from find_one_and_update API command.",
                raw_response=fo_response,
            )

    @recast_method_async
    async def update_many(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        *,
        upsert: bool = False,
        max_time_ms: Optional[int] = None,
    ) -> UpdateResult:
        """
        Apply an update operations to all documents matching a condition,
        optionally inserting one documents in absence of matches.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            update: the update prescription to apply to the documents, expressed
                as a dictionary as per Data API syntax. Examples are:
                    {"$set": {"field": "value}}
                    {"$inc": {"counter": 10}}
                    {"$unset": {"field": ""}}
                See the Data API documentation for the full syntax.
            upsert: this parameter controls the behavior in absence of matches.
                If True, a single new document (resulting from applying `update`
                to an empty document) is inserted if no matches are found on
                the collection. If False, the operation silently does nothing
                in case of no matches.
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            an UpdateResult object summarizing the outcome of the update operation.

        Example:
            >>> async def do_update_many(acol: AsyncCollection) -> None:
            ...     await acol.insert_many([{"c": "red"}, {"c": "green"}, {"c": "blue"}])
            ...     result0 = await acol.update_many(
            ...         {"c": {"$ne": "green"}},
            ...         {"$set": {"nongreen": True}},
            ...     )
            ...     print("result0.update_info", result0.update_info)
            ...     result1 = await acol.update_many(
            ...         {"c": "orange"},
            ...         {"$set": {"is_also_fruit": True}},
            ...     )
            ...     print("result1.update_info", result1.update_info)
            ...     result2 = await acol.update_many(
            ...         {"c": "orange"},
            ...         {"$set": {"is_also_fruit": True}},
            ...         upsert=True,
            ...     )
            ...     print("result2.update_info", result2.update_info)
            ...
            >>> asyncio.run(do_update_many(my_async_coll))
            result0.update_info {'n': 2, 'updatedExisting': True, 'ok': 1.0, 'nModified': 2}
            result1.update_info {'n': 0, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0}
            result2.update_info {'n': 1, 'updatedExisting': False, 'ok': 1.0, 'nModified': 0, 'upserted': '79ffd5a3-ab99-4dff-a2a5-4aaa0e59e854'}

        Note:
            Similarly to the case of `find` (see its docstring for more details),
            running this command while, at the same time, another process is
            inserting new documents which match the filter of the `update_many`
            can result in an unpredictable fraction of these documents being updated.
            In other words, it cannot be easily predicted whether a given
            newly-inserted document will be picked up by the update_many command or not.
        """

        base_options = {
            "upsert": upsert,
        }
        page_state_options: Dict[str, str] = {}
        um_responses: List[Dict[str, Any]] = []
        um_statuses: List[Dict[str, Any]] = []
        must_proceed = True
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"starting update_many on '{self.name}'")
        while must_proceed:
            options = {**base_options, **page_state_options}
            logger.info(f"calling update_many on '{self.name}'")
            this_um_response = await self._astra_db_collection.update_many(
                update=update,
                filter=filter,
                options=options,
                timeout_info=timeout_manager.remaining_timeout_info(),
            )
            logger.info(f"finished calling update_many on '{self.name}'")
            this_um_status = this_um_response.get("status") or {}
            #
            # if errors, quit early
            if this_um_response.get("errors", []):
                partial_update_info = _prepare_update_info(um_statuses)
                partial_result = UpdateResult(
                    raw_results=um_responses,
                    update_info=partial_update_info,
                )
                all_um_responses = um_responses + [this_um_response]
                raise UpdateManyException.from_responses(
                    commands=[None for _ in all_um_responses],
                    raw_responses=all_um_responses,
                    partial_result=partial_result,
                )
            else:
                if "status" not in this_um_response:
                    raise DataAPIFaultyResponseException(
                        text="Faulty response from update_many API command.",
                        raw_response=this_um_response,
                    )
                um_responses.append(this_um_response)
                um_statuses.append(this_um_status)
                next_page_state = this_um_status.get("nextPageState")
                if next_page_state is not None:
                    must_proceed = True
                    page_state_options = {"pageState": next_page_state}
                else:
                    must_proceed = False
                    page_state_options = {}

        update_info = _prepare_update_info(um_statuses)
        logger.info(f"finished update_many on '{self.name}'")
        return UpdateResult(
            raw_results=um_responses,
            update_info=update_info,
        )

    @recast_method_async
    async def find_one_and_delete(
        self,
        filter: Dict[str, Any],
        *,
        projection: Optional[ProjectionType] = None,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> Union[DocumentType, None]:
        """
        Find a document in the collection and delete it. The deleted document,
        however, is the return value of the method.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            projection: used to select a subset of fields in the document being
                returned. The projection can be: an iterable over the field names
                to return; a dictionary {field_name: True} to positively select
                certain fields; or a dictionary {field_name: False} if one wants
                to discard some fields from the response.
                Note that the `_id` field will be returned with the document
                in any case, regardless of what the provided `projection` requires.
                The default is to return the whole documents.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                deleted one. See the `find` method for more on sorting.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            Either the document (or a projection thereof, as requested), or None
            if no matches were found in the first place.

        Example:
            >>> async def do_find_one_and_delete(acol: AsyncCollection) -> None:
            ...     await acol.insert_many(
            ...         [
            ...             {"species": "swan", "class": "Aves"},
            ...             {"species": "frog", "class": "Amphibia"},
            ...         ],
            ...     )
            ...     delete_result0 = await acol.find_one_and_delete(
            ...         {"species": {"$ne": "frog"}},
            ...         projection=["species"],
            ...     )
            ...     print("delete_result0", delete_result0)
            ...     delete_result1 = await acol.find_one_and_delete(
            ...         {"species": {"$ne": "frog"}},
            ...     )
            ...     print("delete_result1", delete_result1)
            ...
            >>> asyncio.run(do_find_one_and_delete(my_async_coll))
            delete_result0 {'_id': 'f335cd0f-...', 'species': 'swan'}
            delete_result1 None
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        _projection = normalize_optional_projection(projection)
        logger.info(f"calling find_one_and_delete on '{self.name}'")
        fo_response = await self._astra_db_collection.find_one_and_delete(
            sort=_sort,
            filter=filter,
            projection=_projection,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished calling find_one_and_delete on '{self.name}'")
        if "document" in fo_response.get("data", {}):
            document = fo_response["data"]["document"]
            return document  # type: ignore[no-any-return]
        else:
            deleted_count = fo_response.get("status", {}).get("deletedCount")
            if deleted_count == 0:
                return None
            else:
                raise DataAPIFaultyResponseException(
                    text="Faulty response from find_one_and_delete API command.",
                    raw_response=fo_response,
                )

    @recast_method_async
    async def delete_one(
        self,
        filter: Dict[str, Any],
        *,
        vector: Optional[VectorType] = None,
        vectorize: Optional[str] = None,
        sort: Optional[SortType] = None,
        max_time_ms: Optional[int] = None,
    ) -> DeleteResult:
        """
        Delete one document matching a provided filter.
        This method never deletes more than a single document, regardless
        of the number of matches to the provided filters.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
            vector: a suitable vector, i.e. a list of float numbers of the appropriate
                dimensionality, to use vector search (i.e. ANN,
                or "approximate nearest-neighbours" search), as the sorting criterion.
                In this way, the matched document (if any) will be the one
                that is most similar to the provided vector.
                This parameter cannot be used together with `sort`.
                See the `find` method for more details on this parameter.
            vectorize: a string to be made into a vector to perform vector search.
                This can be supplied in (exclusive) alternative to `vector`,
                provided such a service is configured for the collection,
                and achieves the same effect.
                NOTE: This feature is under current development.
            sort: with this dictionary parameter one can control the sorting
                order of the documents matching the filter, effectively
                determining what document will come first and hence be the
                deleted one. See the `find` method for more on sorting.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a DeleteResult object summarizing the outcome of the delete operation.

        Example:
            >>> my_coll.insert_many([{"seq": 1}, {"seq": 0}, {"seq": 2}])
            InsertManyResult(...)
            >>> my_coll.delete_one({"seq": 1})
            DeleteResult(raw_results=..., deleted_count=1)
            >>> my_coll.distinct("seq")
            [0, 2]
            >>> my_coll.delete_one(
            ...     {"seq": {"$exists": True}},
            ...     sort={"seq": astrapy.constants.SortDocuments.DESCENDING},
            ... )
            DeleteResult(raw_results=..., deleted_count=1)
            >>> my_coll.distinct("seq")
            [0]
            >>> my_coll.delete_one({"seq": 2})
            DeleteResult(raw_results=..., deleted_count=0)
        """

        _sort = _collate_vector_to_sort(sort, vector, vectorize)
        logger.info(f"calling delete_one_by_predicate on '{self.name}'")
        do_response = await self._astra_db_collection.delete_one_by_predicate(
            filter=filter,
            timeout_info=base_timeout_info(max_time_ms),
            sort=_sort,
        )
        logger.info(f"finished calling delete_one_by_predicate on '{self.name}'")
        if "deletedCount" in do_response.get("status", {}):
            deleted_count = do_response["status"]["deletedCount"]
            if deleted_count == -1:
                return DeleteResult(
                    deleted_count=None,
                    raw_results=[do_response],
                )
            else:
                # expected a non-negative integer:
                return DeleteResult(
                    deleted_count=deleted_count,
                    raw_results=[do_response],
                )
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from delete_one API command.",
                raw_response=do_response,
            )

    @recast_method_async
    async def delete_many(
        self,
        filter: Dict[str, Any],
        *,
        max_time_ms: Optional[int] = None,
    ) -> DeleteResult:
        """
        Delete all documents matching a provided filter.

        Args:
            filter: a predicate expressed as a dictionary according to the
                Data API filter syntax. Examples are:
                    {}
                    {"name": "John"}
                    {"price": {"$le": 100}}
                    {"$and": [{"name": "John"}, {"price": {"$le": 100}}]}
                See the Data API documentation for the full set of operators.
                The `delete_many` method does not accept an empty filter: see
                `delete_all` to completely erase all contents of a collection
            max_time_ms: a timeout, in milliseconds, for the operation.

        Returns:
            a DeleteResult object summarizing the outcome of the delete operation.

        Example:
            >>> async def do_delete_many(acol: AsyncCollection) -> None:
            ...     await acol.insert_many([{"seq": 1}, {"seq": 0}, {"seq": 2}])
            ...     delete_result0 = await acol.delete_many({"seq": {"$lte": 1}})
            ...     print("delete_result0.deleted_count", delete_result0.deleted_count)
            ...     distinct1 = await acol.distinct("seq")
            ...     print("distinct1", distinct1)
            ...     delete_result2 = await acol.delete_many({"seq": {"$lte": 1}})
            ...     print("delete_result2.deleted_count", delete_result2.deleted_count)
            ...
            >>> asyncio.run(do_delete_many(my_async_coll))
            delete_result0.deleted_count 2
            distinct1 [2]
            delete_result2.deleted_count 0

        Note:
            This operation is not atomic. Depending on the amount of matching
            documents, it can keep running (in a blocking way) for a macroscopic
            time. In that case, new documents that are meanwhile inserted
            (e.g. from another process/application) will be deleted during
            the execution of this method call until the collection is devoid
            of matches.
        """

        if not filter:
            raise ValueError(
                "The `filter` parameter to method `delete_many` cannot be "
                "empty. In order to completely clear the contents of a "
                "collection, please use the `delete_all` method."
            )

        dm_responses: List[Dict[str, Any]] = []
        deleted_count = 0
        must_proceed = True
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"starting delete_many on '{self.name}'")
        while must_proceed:
            logger.info(f"calling delete_many on '{self.name}'")
            this_dm_response = await self._astra_db_collection.delete_many(
                filter=filter,
                skip_error_check=True,
                timeout_info=timeout_manager.remaining_timeout_info(),
            )
            logger.info(f"finished calling delete_many on '{self.name}'")
            # if errors, quit early
            if this_dm_response.get("errors", []):
                partial_result = DeleteResult(
                    deleted_count=deleted_count,
                    raw_results=dm_responses,
                )
                all_dm_responses = dm_responses + [this_dm_response]
                raise DeleteManyException.from_responses(
                    commands=[None for _ in all_dm_responses],
                    raw_responses=all_dm_responses,
                    partial_result=partial_result,
                )
            else:
                this_dc = this_dm_response.get("status", {}).get("deletedCount")
                if this_dc is None or this_dc < 0:
                    raise DataAPIFaultyResponseException(
                        text="Faulty response from delete_many API command.",
                        raw_response=this_dm_response,
                    )
                dm_responses.append(this_dm_response)
                deleted_count += this_dc
                must_proceed = this_dm_response.get("status", {}).get("moreData", False)

        logger.info(f"finished delete_many on '{self.name}'")
        return DeleteResult(
            deleted_count=deleted_count,
            raw_results=dm_responses,
        )

    @recast_method_async
    async def delete_all(self, *, max_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Delete all documents in a collection.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary of the form {"ok": 1} to signal successful deletion.

        Example:
            >>> async def do_delete_all(acol: AsyncCollection) -> None:
            ...     distinct0 = await acol.distinct("seq")
            ...     print("distinct0", distinct0)
            ...     count1 = await acol.count_documents({}, upper_bound=100)
            ...     print("count1", count1)
            ...     delete_result2 = await acol.delete_all()
            ...     print("delete_result2", delete_result2)
            ...     count3 = await acol.count_documents({}, upper_bound=100)
            ...     print("count3", count3)
            ...
            >>> asyncio.run(do_delete_all(my_async_coll))
            distinct0 [4, 2, 3, 0, 1]
            count1 5
            delete_result2 {'ok': 1}
            count3 0

        Note:
            Use with caution.

        Note:
            Once the method succeeds, methods on this object can still be invoked:
            however, this hardly makes sense as the underlying actual collection
            is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased collection any further.
        """

        logger.info(f"calling unfiltered delete_many on '{self.name}'")
        dm_response = await self._astra_db_collection.delete_many(
            filter={}, timeout_info=base_timeout_info(max_time_ms)
        )
        logger.info(f"finished calling unfiltered delete_many on '{self.name}'")
        deleted_count = dm_response["status"]["deletedCount"]
        if deleted_count == -1:
            return {"ok": 1}
        else:
            raise DataAPIFaultyResponseException(
                text="Faulty response from delete_many API command.",
                raw_response=dm_response,
            )

    async def bulk_write(
        self,
        requests: Iterable[AsyncBaseOperation],
        *,
        ordered: bool = True,
        concurrency: Optional[int] = None,
        max_time_ms: Optional[int] = None,
    ) -> BulkWriteResult:
        """
        Execute an arbitrary amount of operations such as inserts, updates, deletes
        either sequentially or concurrently.

        This method does not execute atomically, i.e. individual operations are
        each performed in the same way as the corresponding collection method,
        and each one is a different and unrelated database mutation.

        Args:
            requests: an iterable over concrete subclasses of `BaseOperation`,
                such as `AsyncInsertMany` or `AsyncReplaceOne`. Each such object
                represents an operation ready to be executed on a collection,
                and is instantiated by passing the same parameters as one
                would the corresponding collection method.
            ordered: whether to launch the `requests` one after the other or
                in arbitrary order, possibly in a concurrent fashion. For
                performance reasons, `ordered=False` should be preferred
                when compatible with the needs of the application flow.
            concurrency: maximum number of concurrent operations executing at
                a given time. It cannot be more than one for ordered bulk writes.
            max_time_ms: a timeout, in milliseconds, for the whole bulk write.
                Remember that, if the method call times out, then there's no
                guarantee about what portion of the bulk write has been received
                and successfully executed by the Data API.

        Returns:
            A single BulkWriteResult summarizing the whole list of requested
            operations. The keys in the map attributes of BulkWriteResult
            (when present) are the integer indices of the corresponding operation
            in the `requests` iterable.

        Example:
            >>> from astrapy.operations import AsyncInsertMany, AsyncReplaceOne, AsyncOperation
            >>> from astrapy.results import BulkWriteResult
            >>>
            >>> async def do_bulk_write(
            ...     acol: AsyncCollection,
            ...     async_operations: List[AsyncOperation],
            ... ) -> BulkWriteResult:
            ...     bw_result = await acol.bulk_write(async_operations)
            ...     count0 = await acol.count_documents({}, upper_bound=100)
            ...     print("count0", count0)
            ...     distinct0 = await acol.distinct("replaced")
            ...     print("distinct0", distinct0)
            ...     return bw_result
            ...
            >>> op1 = AsyncInsertMany([{"a": 1}, {"a": 2}])
            >>> op2 = AsyncReplaceOne(
            ...     {"z": 9},
            ...     replacement={"z": 9, "replaced": True},
            ...     upsert=True,
            ... )
            >>> result = asyncio.run(do_bulk_write(my_async_coll, [op1, op2]))
            count0 3
            distinct0 [True]
            >>> print("result", result)
            result BulkWriteResult(bulk_api_results={0: ..., 1: ...}, deleted_count=0, inserted_count=3, matched_count=0, modified_count=0, upserted_count=1, upserted_ids={1: 'ccd0a800-...'})
        """

        # lazy importing here against circular-import error
        from astrapy.operations import reduce_bulk_write_results

        if concurrency is None:
            if ordered:
                _concurrency = 1
            else:
                _concurrency = DEFAULT_BULK_WRITE_CONCURRENCY
        else:
            _concurrency = concurrency
        if _concurrency > 1 and ordered:
            raise ValueError("Cannot run ordered bulk_write concurrently.")
        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)
        logger.info(f"startng a bulk write on '{self.name}'")
        if ordered:
            bulk_write_results: List[BulkWriteResult] = []
            for operation_i, operation in enumerate(requests):
                try:
                    this_bw_result = await operation.execute(
                        self,
                        index_in_bulk_write=operation_i,
                        bulk_write_timeout_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    bulk_write_results.append(this_bw_result)
                except CumulativeOperationException as exc:
                    partial_result = exc.partial_result
                    partial_bw_result = reduce_bulk_write_results(
                        bulk_write_results
                        + [
                            partial_result.to_bulk_write_result(
                                index_in_bulk_write=operation_i
                            )
                        ]
                    )
                    dar_exception = exc.data_api_response_exception()
                    raise BulkWriteException(
                        text=dar_exception.text,
                        error_descriptors=dar_exception.error_descriptors,
                        detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                        partial_result=partial_bw_result,
                        exceptions=[dar_exception],
                    )
                except DataAPIResponseException as exc:
                    # the cumulative exceptions, with their
                    # partially-done-info, are handled above:
                    # here it's just one-shot d.a.r. exceptions
                    partial_bw_result = reduce_bulk_write_results(bulk_write_results)
                    dar_exception = exc.data_api_response_exception()
                    raise BulkWriteException(
                        text=dar_exception.text,
                        error_descriptors=dar_exception.error_descriptors,
                        detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                        partial_result=partial_bw_result,
                        exceptions=[dar_exception],
                    )
            full_bw_result = reduce_bulk_write_results(bulk_write_results)
            logger.info(f"finished a bulk write on '{self.name}'")
            return full_bw_result
        else:

            sem = asyncio.Semaphore(_concurrency)

            async def _concurrent_execute_as_either(
                operation: AsyncBaseOperation, operation_i: int
            ) -> Tuple[Optional[BulkWriteResult], Optional[DataAPIResponseException]]:
                async with sem:
                    try:
                        ex_result = await operation.execute(
                            self,
                            index_in_bulk_write=operation_i,
                            bulk_write_timeout_ms=timeout_manager.remaining_timeout_ms(),
                        )
                        return (ex_result, None)
                    except DataAPIResponseException as exc:
                        return (None, exc)

            tasks = [
                asyncio.create_task(
                    _concurrent_execute_as_either(operation, operation_i)
                )
                for operation_i, operation in enumerate(requests)
            ]
            bulk_write_either_results = await asyncio.gather(*tasks)
            # regroup
            bulk_write_successes = [bwr for bwr, _ in bulk_write_either_results if bwr]
            bulk_write_failures = [bwf for _, bwf in bulk_write_either_results if bwf]
            if bulk_write_failures:
                # extract and cumulate
                partial_results_from_failures = [
                    failure.partial_result.to_bulk_write_result(
                        index_in_bulk_write=operation_i
                    )
                    for failure in bulk_write_failures
                    if isinstance(failure, CumulativeOperationException)
                ]
                partial_bw_result = reduce_bulk_write_results(
                    bulk_write_successes + partial_results_from_failures
                )
                # raise and recast the first exception
                all_dar_exceptions = [
                    bw_failure.data_api_response_exception()
                    for bw_failure in bulk_write_failures
                ]
                dar_exception = all_dar_exceptions[0]
                raise BulkWriteException(
                    text=dar_exception.text,
                    error_descriptors=dar_exception.error_descriptors,
                    detailed_error_descriptors=dar_exception.detailed_error_descriptors,
                    partial_result=partial_bw_result,
                    exceptions=all_dar_exceptions,
                )
            else:
                logger.info(f"finished a bulk write on '{self.name}'")
                return reduce_bulk_write_results(bulk_write_successes)

    async def drop(self, *, max_time_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Drop the collection, i.e. delete it from the database along with
        all the documents it contains.

        Args:
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.
                Remember there is not guarantee that a request that has
                timed out us not in fact honored.

        Returns:
            a dictionary of the form {"ok": 1} to signal successful deletion.

        Example:
            >>> async def drop_and_check(acol: AsyncCollection) -> None:
            ...     doc0 = await acol.find_one({})
            ...     print("doc0", doc0)
            ...     drop_result = await acol.drop()
            ...     print("drop_result", drop_result)
            ...     doc1 = await acol.find_one({})
            ...
            >>> asyncio.run(drop_and_check(my_async_coll))
            doc0 {'_id': '...', 'z': -10}
            drop_result {'ok': 1}
            Traceback (most recent call last):
                ... ...
            astrapy.exceptions.DataAPIResponseException: Collection does not exist, collection name: my_collection

        Note:
            Use with caution.

        Note:
            Once the method succeeds, methods on this object can still be invoked:
            however, this hardly makes sense as the underlying actual collection
            is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased collection any further.
        """

        logger.info(f"dropping collection '{self.name}' (self)")
        drop_result = await self.database.drop_collection(self, max_time_ms=max_time_ms)
        logger.info(f"finished dropping collection '{self.name}' (self)")
        return drop_result  # type: ignore[no-any-return]

    async def command(
        self,
        body: Dict[str, Any],
        *,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a POST request to the Data API for this collection with
        an arbitrary, caller-provided payload.

        Args:
            body: a JSON-serializable dictionary, the payload of the request.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary with the response of the HTTP request.

        Example:
            >>> asyncio.await(my_async_coll.command({"countDocuments": {}}))
            {'status': {'count': 123}}
        """

        logger.info(f"calling command on '{self.name}'")
        command_result = await self.database.command(
            body=body,
            namespace=self.namespace,
            collection_name=self.name,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished calling command on '{self.name}'")
        return command_result  # type: ignore[no-any-return]
