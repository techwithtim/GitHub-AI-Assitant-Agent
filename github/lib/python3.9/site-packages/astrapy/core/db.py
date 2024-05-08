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
import deprecation
import httpx
import logging
import json
import threading

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
)
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue
from types import TracebackType
from typing import (
    Any,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
)

from astrapy import __version__
from astrapy.core.api import APIRequestError, api_request, async_api_request
from astrapy.core.defaults import (
    DEFAULT_AUTH_HEADER,
    DEFAULT_JSON_API_PATH,
    DEFAULT_JSON_API_VERSION,
    DEFAULT_KEYSPACE_NAME,
    MAX_INSERT_NUM_DOCUMENTS,
)
from astrapy.core.utils import (
    convert_vector_to_floats,
    make_payload,
    normalize_for_api,
    restore_from_api,
    http_methods,
    to_httpx_timeout,
    TimeoutInfoWideType,
)
from astrapy.core.core_types import (
    API_DOC,
    API_RESPONSE,
    PaginableRequestMethod,
    AsyncPaginableRequestMethod,
)


logger = logging.getLogger(__name__)


class AstraDBCollection:
    # Initialize the shared httpx client as a class attribute
    client = httpx.Client()

    def __init__(
        self,
        collection_name: str,
        astra_db: Optional[AstraDB] = None,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Initialize an AstraDBCollection instance.

        Args:
            collection_name (str): The name of the collection.
            astra_db (AstraDB, optional): An instance of Astra DB.
            token (str, optional): Authentication token for Astra DB.
            api_endpoint (str, optional): API endpoint URL.
            namespace (str, optional): Namespace for the database.
            caller_name (str, optional): identity of the caller ("my_framework")
                If passing a client, its caller is used as fallback
            caller_version (str, optional): version of the caller code ("1.0.3")
                If passing a client, its caller is used as fallback
        """
        # Check for presence of the Astra DB object
        if astra_db is None:
            if token is None or api_endpoint is None:
                raise AssertionError("Must provide token and api_endpoint")

            astra_db = AstraDB(
                token=token,
                api_endpoint=api_endpoint,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            )
        else:
            # if astra_db passed, copy and apply possible overrides
            astra_db = astra_db.copy(
                token=token,
                api_endpoint=api_endpoint,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            )

        # Set the remaining instance attributes
        self.astra_db = astra_db
        self.caller_name: Optional[str] = self.astra_db.caller_name
        self.caller_version: Optional[str] = self.astra_db.caller_version
        self.collection_name = collection_name
        self.base_path: str = f"{self.astra_db.base_path}/{self.collection_name}"

    def __repr__(self) -> str:
        return f'AstraDBCollection[astra_db="{self.astra_db}", collection_name="{self.collection_name}"]'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AstraDBCollection):
            return all(
                [
                    self.collection_name == other.collection_name,
                    self.astra_db == other.astra_db,
                    self.caller_name == other.caller_name,
                    self.caller_version == other.caller_version,
                ]
            )
        else:
            return False

    def copy(
        self,
        *,
        collection_name: Optional[str] = None,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AstraDBCollection:
        return AstraDBCollection(
            collection_name=collection_name or self.collection_name,
            astra_db=self.astra_db.copy(
                token=token,
                api_endpoint=api_endpoint,
                api_path=api_path,
                api_version=api_version,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            ),
            caller_name=caller_name or self.caller_name,
            caller_version=caller_version or self.caller_version,
        )

    def to_async(self) -> AsyncAstraDBCollection:
        return AsyncAstraDBCollection(
            collection_name=self.collection_name,
            astra_db=self.astra_db.to_async(),
            caller_name=self.caller_name,
            caller_version=self.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self.astra_db.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )
        self.caller_name = caller_name
        self.caller_version = caller_version

    def _request(
        self,
        method: str = http_methods.POST,
        path: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        url_params: Optional[Dict[str, Any]] = None,
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        direct_response = api_request(
            client=self.client,
            base_url=self.astra_db.base_url,
            auth_header=DEFAULT_AUTH_HEADER,
            token=self.astra_db.token,
            method=method,
            json_data=normalize_for_api(json_data),
            url_params=url_params,
            path=path,
            skip_error_check=skip_error_check,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
            timeout=to_httpx_timeout(timeout_info),
        )
        response = restore_from_api(direct_response)
        return response

    def post_raw_request(
        self, body: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        return self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=body,
            timeout_info=timeout_info,
        )

    def _get(
        self,
        path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Optional[API_RESPONSE]:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = self._request(
            method=http_methods.GET,
            path=full_path,
            url_params=options,
            timeout_info=timeout_info,
        )
        if isinstance(response, dict):
            return response
        return None

    def _put(
        self,
        path: Optional[str] = None,
        document: Optional[API_RESPONSE] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = self._request(
            method=http_methods.PUT,
            path=full_path,
            json_data=document,
            timeout_info=timeout_info,
        )
        return response

    def _post(
        self,
        path: Optional[str] = None,
        document: Optional[API_DOC] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = self._request(
            method=http_methods.POST,
            path=full_path,
            json_data=document,
            timeout_info=timeout_info,
        )
        return response

    def _recast_as_sort_projection(
        self, vector: List[float], fields: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Given a vector and optionally a list of fields,
        reformulate them as a sort, projection pair for regular
        'find'-like API calls (with basic validation as well).
        """
        # Must pass a vector
        if not vector:
            raise ValueError("Must pass a vector")

        # Edge case for field selection
        if fields and "$similarity" in fields:
            raise ValueError("Please use the `include_similarity` parameter")

        # Build the new vector parameter
        sort: Dict[str, Any] = {"$vector": vector}

        # Build the new fields parameter
        # Note: do not leave projection={}, make it None
        # (or it will devour $similarity away in the API response)
        if fields is not None and len(fields) > 0:
            projection = {f: 1 for f in fields}
        else:
            projection = None

        return sort, projection

    def get(
        self, path: Optional[str] = None, timeout_info: TimeoutInfoWideType = None
    ) -> Optional[API_RESPONSE]:
        """
        Retrieve a document from the collection by its path.

        Args:
            path (str, optional): The path of the document to retrieve.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The retrieved document.
        """
        return self._get(path=path, timeout_info=timeout_info)

    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find documents in the collection that match the given filter.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return matching documents.
            options (dict, optional): Additional options for the query.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The query response containing matched documents.
        """
        json_query = make_payload(
            top_level="find",
            filter=filter,
            projection=projection,
            options=options,
            sort=sort,
        )

        response = self._post(document=json_query, timeout_info=timeout_info)

        return response

    def vector_find(
        self,
        vector: List[float],
        *,
        limit: int,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        include_similarity: bool = True,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[API_DOC]:
        """
        Perform a vector-based search in the collection.

        Args:
            vector (list): The vector to search with.
            limit (int): The maximum number of documents to return.
            filter (dict, optional): Criteria to filter documents.
            fields (list, optional): Specifies the fields to return.
            include_similarity (bool, optional): Whether to include similarity score in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            list: A list of documents matching the vector search criteria.
        """
        # Must pass a limit
        if not limit:
            raise ValueError("Must pass a limit")

        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            convert_vector_to_floats(vector),
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = self.find(
            filter=filter,
            projection=projection,
            sort=sort,
            options={
                "limit": limit,
                "includeSimilarity": include_similarity,
            },
            timeout_info=timeout_info,
        )

        return cast(List[API_DOC], raw_find_result["data"]["documents"])

    @staticmethod
    def paginate(
        *,
        request_method: PaginableRequestMethod,
        options: Optional[Dict[str, Any]],
        prefetched: Optional[int] = None,
    ) -> Generator[API_DOC, None, None]:
        """
        Generate paginated results for a given database query method.

        Args:
            request_method (function): The database query method to paginate.
            options (dict, optional): Options for the database query.
            prefetched (int, optional): Number of pre-fetched documents.

        Yields:
            dict: The next document in the paginated result set.
        """
        _options = options or {}
        response0 = request_method(options=_options)
        next_page_state = response0["data"]["nextPageState"]
        options0 = _options
        if next_page_state is not None and prefetched:

            def queued_paginate(
                queue: Queue[Optional[API_DOC]],
                request_method: PaginableRequestMethod,
                options: Optional[Dict[str, Any]],
            ) -> None:
                try:
                    for row in AstraDBCollection.paginate(
                        request_method=request_method, options=options
                    ):
                        queue.put(row)
                finally:
                    queue.put(None)

            queue: Queue[Optional[API_DOC]] = Queue(prefetched)
            options1 = {**options0, **{"pageState": next_page_state}}
            t = threading.Thread(
                target=queued_paginate, args=(queue, request_method, options1)
            )
            t.start()
            for document in response0["data"]["documents"]:
                yield document
            doc = queue.get()
            while doc is not None:
                yield doc
                doc = queue.get()
            t.join()
        else:
            for document in response0["data"]["documents"]:
                yield document
            while next_page_state is not None and not prefetched:
                options1 = {**options0, **{"pageState": next_page_state}}
                response1 = request_method(options=options1)
                for document in response1["data"]["documents"]:
                    yield document
                next_page_state = response1["data"]["nextPageState"]

    def paginated_find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        prefetched: Optional[int] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Iterator[API_DOC]:
        """
        Perform a paginated search in the collection.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return matching documents.
            options (dict, optional): Additional options for the query.
            prefetched (int, optional): Number of pre-fetched documents.
            timeout_info: a float, or a TimeoutInfo dict, for each
                single HTTP request.
                This is a paginated method, that issues several requests as it
                needs more data. This parameter controls a single request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            generator: A generator yielding documents in the paginated result set.
        """
        partialed_find = partial(
            self.find,
            filter=filter,
            projection=projection,
            sort=sort,
            timeout_info=timeout_info,
        )
        return self.paginate(
            request_method=partialed_find,
            options=options,
            prefetched=prefetched,
        )

    def pop(
        self,
        filter: Dict[str, Any],
        pop: Dict[str, Any],
        options: Dict[str, Any],
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Pop the last data in the tags array

        Args:
            filter (dict): Criteria to identify the document to update.
            pop (dict): The pop to apply to the tags.
            options (dict): Additional options for the update operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The original document before the update.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update={"$pop": pop},
            options=options,
        )

        response = self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def push(
        self,
        filter: Dict[str, Any],
        push: Dict[str, Any],
        options: Dict[str, Any],
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Push new data to the tags array

        Args:
            filter (dict): Criteria to identify the document to update.
            push (dict): The push to apply to the tags.
            options (dict): Additional options for the update operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the update operation.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update={"$push": push},
            options=options,
        )

        response = self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def find_one_and_replace(
        self,
        replacement: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and replace it.

        Args:
            replacement (dict): The new document to replace the existing one.
            filter (dict, optional): Criteria to filter documents.
            sort (dict, optional): Specifies the order in which to find the document.
            options (dict, optional): Additional options for the operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and replace operation.
        """
        json_query = make_payload(
            top_level="findOneAndReplace",
            filter=filter,
            projection=projection,
            replacement=replacement,
            options=options,
            sort=sort,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def vector_find_one_and_replace(
        self,
        vector: List[float],
        replacement: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search and replace the first matched document.

        Args:
            vector (dict): The vector to search with.
            replacement (dict): The new document to replace the existing one.
            filter (dict, optional): Criteria to filter documents.
            fields (list, optional): Specifies the fields to return in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: either the matched document or None if nothing found
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            convert_vector_to_floats(vector),
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = self.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            projection=projection,
            sort=sort,
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    def find_one_and_update(
        self,
        update: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = {},
        filter: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and update it.

        Args:
            update (dict): The update to apply to the document.
            sort (dict, optional): Specifies the order in which to find the document.
            filter (dict, optional): Criteria to filter documents.
            options (dict, optional): Additional options for the operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and update operation.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update=update,
            options=options,
            sort=sort,
            projection=projection,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def vector_find_one_and_update(
        self,
        vector: List[float],
        update: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search and update the first matched document.

        Args:
            vector (list): The vector to search with.
            update (dict): The update to apply to the matched document.
            filter (dict, optional): Criteria to filter documents before applying the vector search.
            fields (list, optional): Specifies the fields to return in the updated document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: The result of the vector-based find and
                update operation, or None if nothing found
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            convert_vector_to_floats(vector),
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = self.find_one_and_update(
            update=update,
            filter=filter,
            sort=sort,
            projection=projection,
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    def find_one_and_delete(
        self,
        sort: Optional[Dict[str, Any]] = {},
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and delete it.

        Args:
            sort (dict, optional): Specifies the order in which to find the document.
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and delete operation.
        """
        json_query = make_payload(
            top_level="findOneAndDelete",
            filter=filter,
            sort=sort,
            projection=projection,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def count_documents(
        self, filter: Dict[str, Any] = {}, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Count documents matching a given predicate (expressed as filter).

        Args:
            filter (dict, defaults to {}): Criteria to filter documents.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: the response, either
                {"status": {"count": <NUMBER> }}
            or
                {"errors": [...]}
        """
        json_query = make_payload(
            top_level="countDocuments",
            filter=filter,
        )

        response = self._post(document=json_query, timeout_info=timeout_info)

        return response

    def find_one(
        self,
        filter: Optional[Dict[str, Any]] = {},
        projection: Optional[Dict[str, Any]] = {},
        sort: Optional[Dict[str, Any]] = {},
        options: Optional[Dict[str, Any]] = {},
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document in the collection.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return the document.
            options (dict, optional): Additional options for the query.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: the response, either
                {"data": {"document": <DOCUMENT> }}
            or
                {"data": {"document": None}}
            depending on whether a matching document is found or not.
        """
        json_query = make_payload(
            top_level="findOne",
            filter=filter,
            projection=projection,
            options=options,
            sort=sort,
        )

        response = self._post(document=json_query, timeout_info=timeout_info)

        return response

    def vector_find_one(
        self,
        vector: List[float],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        include_similarity: bool = True,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search to find a single document in the collection.

        Args:
            vector (list): The vector to search with.
            filter (dict, optional): Additional criteria to filter documents.
            fields (list, optional): Specifies the fields to return in the result.
            include_similarity (bool, optional): Whether to include similarity score in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: The found document or None if no matching document is found.
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            convert_vector_to_floats(vector),
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = self.find_one(
            filter=filter,
            projection=projection,
            sort=sort,
            options={"includeSimilarity": include_similarity},
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    def insert_one(
        self,
        document: API_DOC,
        failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Insert a single document into the collection.

        Args:
            document (dict): The document to insert.
            failures_allowed (bool): Whether to allow failures in the insert operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the insert operation.
        """
        json_query = make_payload(top_level="insertOne", document=document)

        response = self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            skip_error_check=failures_allowed,
            timeout_info=timeout_info,
        )

        return response

    def insert_many(
        self,
        documents: List[API_DOC],
        options: Optional[Dict[str, Any]] = None,
        partial_failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Insert multiple documents into the collection.

        Args:
            documents (list): A list of documents to insert.
            options (dict, optional): Additional options for the insert operation.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures through the insertion (i.e. on some documents).
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the insert operation.
        """
        json_query = make_payload(
            top_level="insertMany", documents=documents, options=options
        )

        # Send the data
        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            skip_error_check=partial_failures_allowed,
            timeout_info=timeout_info,
        )

        return response

    def chunked_insert_many(
        self,
        documents: List[API_DOC],
        options: Optional[Dict[str, Any]] = None,
        partial_failures_allowed: bool = False,
        chunk_size: int = MAX_INSERT_NUM_DOCUMENTS,
        concurrency: int = 1,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[Union[API_RESPONSE, Exception]]:
        """
        Insert multiple documents into the collection, handling chunking and
        optionally with concurrent insertions.

        Args:
            documents (list): A list of documents to insert.
            options (dict, optional): Additional options for the insert operation.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures in the chunk. Should be used combined with
                options={"ordered": False} in most cases.
            chunk_size (int, optional): Override the default insertion chunk size.
            concurrency (int, optional): The number of concurrent chunk insertions.
                Default is no concurrency.
            timeout_info: a float, or a TimeoutInfo dict, for each single HTTP request.
                This method runs a number of HTTP requests as it works on chunked
                data. The timeout refers to each individual such request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            list: The responses from the database after the chunked insert operation.
                This is a list of individual responses from the API: the caller
                will need to inspect them all, e.g. to collate the inserted IDs.
        """
        results: List[Union[API_RESPONSE, Exception]] = []

        # Raise a warning if ordered and concurrency
        if options and options.get("ordered") is True and concurrency > 1:
            logger.warning(
                "Using ordered insert with concurrency may lead to unexpected results."
            )

        # If we have concurrency as 1, don't use a thread pool
        if concurrency == 1:
            # Split the documents into chunks
            for i in range(0, len(documents), chunk_size):
                try:
                    results.append(
                        self.insert_many(
                            documents[i : i + chunk_size],
                            options,
                            partial_failures_allowed,
                            timeout_info=timeout_info,
                        )
                    )
                except APIRequestError as e:
                    if partial_failures_allowed:
                        results.append(e)
                    else:
                        raise e
            return results

        # Perform the bulk insert with concurrency otherwise
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit the jobs
            futures = [
                executor.submit(
                    self.insert_many,
                    documents[i : i + chunk_size],
                    options,
                    partial_failures_allowed,
                    timeout_info=timeout_info,
                )
                for i in range(0, len(documents), chunk_size)
            ]

            # Collect the results
            for future in futures:
                try:
                    results.append(future.result())
                except APIRequestError as e:
                    if partial_failures_allowed:
                        results.append(e)
                    else:
                        raise e

        return results

    def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Update a single document in the collection.

        Args:
            filter (dict): Criteria to identify the document to update.
            update (dict): The update to apply to the document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = make_payload(
            top_level="updateOne",
            filter=filter,
            update=update,
            sort=sort,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def update_many(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Updates multiple documents in the collection.

        Args:
            filter (dict): Criteria to identify the document to update.
            update (dict): The update to apply to the document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = make_payload(
            top_level="updateMany",
            filter=filter,
            update=update,
            options=options,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def replace(
        self, path: str, document: API_DOC, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Replace a document in the collection.

        Args:
            path (str): The path to the document to replace.
            document (dict): The new document to replace the existing one.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the replace operation.
        """
        return self._put(path=path, document=document, timeout_info=timeout_info)

    @deprecation.deprecated(  # type: ignore
        deprecated_in="0.7.0",
        removed_in="1.0.0",
        current_version=__version__,
        details="Use the 'delete_one' method instead",
    )
    def delete(self, id: str, timeout_info: TimeoutInfoWideType = None) -> API_RESPONSE:
        return self.delete_one(id, timeout_info=timeout_info)

    def delete_one(
        self,
        id: str,
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete a single document from the collection based on its ID.

        Args:
            id (str): The ID of the document to delete.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = make_payload(
            top_level="deleteOne",
            filter={"_id": id},
            sort=sort,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def delete_one_by_predicate(
        self,
        filter: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete a single document from the collection based on a filter clause

        Args:
            filter: any filter dictionary
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = make_payload(
            top_level="deleteOne",
            filter=filter,
            sort=sort,
        )

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def delete_many(
        self,
        filter: Dict[str, Any],
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete many documents from the collection based on a filter condition

        Args:
            filter (dict): Criteria to identify the documents to delete.
            skip_error_check (bool): whether to ignore the check for API error
                and return the response untouched. Default is False.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = {
            "deleteMany": {
                "filter": filter,
            }
        }

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            skip_error_check=skip_error_check,
            timeout_info=timeout_info,
        )

        return response

    def chunked_delete_many(
        self, filter: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> List[API_RESPONSE]:
        """
        Delete many documents from the collection based on a filter condition,
        chaining several API calls until exhaustion of the documents to delete.

        Args:
            filter (dict): Criteria to identify the documents to delete.
            timeout_info: a float, or a TimeoutInfo dict, for each single HTTP request.
                This method runs a number of HTTP requests as it works on a
                pagination basis. The timeout refers to each individual such request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            List[dict]: The responses from the database from all the calls
        """
        responses = []
        must_proceed = True
        while must_proceed:
            dm_response = self.delete_many(filter=filter, timeout_info=timeout_info)
            responses.append(dm_response)
            must_proceed = dm_response.get("status", {}).get("moreData", False)
        return responses

    def clear(self, timeout_info: TimeoutInfoWideType = None) -> API_RESPONSE:
        """
        Clear the collection, deleting all documents

        Args:
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database.
        """
        clear_response = self.delete_many(filter={}, timeout_info=timeout_info)

        if clear_response.get("status", {}).get("deletedCount") != -1:
            raise ValueError(
                f"Could not issue a clear-collection API command (response: {json.dumps(clear_response)})."
            )

        return clear_response

    def delete_subdocument(
        self, id: str, subdoc: str, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Delete a subdocument or field from a document in the collection.

        Args:
            id (str): The ID of the document containing the subdocument.
            subdoc (str): The key of the subdocument or field to remove.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = {
            "findOneAndUpdate": {
                "filter": {"_id": id},
                "update": {"$unset": {subdoc: ""}},
            }
        }

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    @deprecation.deprecated(  # type: ignore
        deprecated_in="0.7.0",
        removed_in="1.0.0",
        current_version=__version__,
        details="Use the 'upsert_one' method instead",
    )
    def upsert(
        self, document: API_DOC, timeout_info: TimeoutInfoWideType = None
    ) -> str:
        return self.upsert_one(document, timeout_info=timeout_info)

    def upsert_one(
        self, document: API_DOC, timeout_info: TimeoutInfoWideType = None
    ) -> str:
        """
        Emulate an upsert operation for a single document in the collection.

        This method attempts to insert the document.
        If a document with the same _id exists, it updates the existing document.

        Args:
            document (dict): The document to insert or update.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP requests.
                This method may issue one or two requests, depending on what
                is detected on DB. This timeout controls each HTTP request individually.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            str: The _id of the inserted or updated document.
        """
        # Build the payload for the insert attempt
        result = self.insert_one(
            document, failures_allowed=True, timeout_info=timeout_info
        )

        # If the call failed because of preexisting doc, then we replace it
        if "errors" in result:
            if (
                "errorCode" in result["errors"][0]
                and result["errors"][0]["errorCode"] == "DOCUMENT_ALREADY_EXISTS"
            ):
                # Now we attempt the update
                result = self.find_one_and_replace(
                    replacement=document,
                    filter={"_id": document["_id"]},
                    timeout_info=timeout_info,
                )
                upserted_id = cast(str, result["data"]["document"]["_id"])
            else:
                raise ValueError(result)
        else:
            if result.get("status", {}).get("insertedIds", []):
                upserted_id = cast(str, result["status"]["insertedIds"][0])
            else:
                raise ValueError("Unexplained empty insertedIds from API")

        return upserted_id

    def upsert_many(
        self,
        documents: list[API_DOC],
        concurrency: int = 1,
        partial_failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[Union[str, Exception]]:
        """
        Emulate an upsert operation for multiple documents in the collection.

        This method attempts to insert the documents.
        If a document with the same _id exists, it updates the existing document.

        Args:
            documents (List[dict]): The documents to insert or update.
            concurrency (int, optional): The number of concurrent upserts.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures in the batch.
            timeout_info: a float, or a TimeoutInfo dict, for each HTTP request.
                This method issues a separate HTTP request for each document to
                insert: the timeout controls each such request individually.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            List[Union[str, Exception]]: A list of "_id"s of the inserted or updated documents.
        """
        results: List[Union[str, Exception]] = []

        # If concurrency is 1, no need for thread pool
        if concurrency == 1:
            for document in documents:
                try:
                    results.append(self.upsert_one(document, timeout_info=timeout_info))
                except Exception as e:
                    results.append(e)
            return results

        # Perform the bulk upsert with concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit the jobs
            futures = [
                executor.submit(self.upsert, document, timeout_info=timeout_info)
                for document in documents
            ]

            # Collect the results
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    if partial_failures_allowed:
                        results.append(e)
                    else:
                        raise e

        return results


class AsyncAstraDBCollection:
    def __init__(
        self,
        collection_name: str,
        astra_db: Optional[AsyncAstraDB] = None,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Initialize an AstraDBCollection instance.

        Args:
            collection_name (str): The name of the collection.
            astra_db (AstraDB, optional): An instance of Astra DB.
            token (str, optional): Authentication token for Astra DB.
            api_endpoint (str, optional): API endpoint URL.
            namespace (str, optional): Namespace for the database.
            caller_name (str, optional): identity of the caller ("my_framework")
                If passing a client, its caller is used as fallback
            caller_version (str, optional): version of the caller code ("1.0.3")
                If passing a client, its caller is used as fallback
        """
        # Check for presence of the Astra DB object
        if astra_db is None:
            if token is None or api_endpoint is None:
                raise AssertionError("Must provide token and api_endpoint")

            astra_db = AsyncAstraDB(
                token=token,
                api_endpoint=api_endpoint,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            )
        else:
            # if astra_db passed, copy and apply possible overrides
            astra_db = astra_db.copy(
                token=token,
                api_endpoint=api_endpoint,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            )

        # Set the remaining instance attributes
        self.astra_db: AsyncAstraDB = astra_db
        self.caller_name: Optional[str] = self.astra_db.caller_name
        self.caller_version: Optional[str] = self.astra_db.caller_version
        self.client = astra_db.client
        self.collection_name = collection_name
        self.base_path: str = f"{self.astra_db.base_path}/{self.collection_name}"

    def __repr__(self) -> str:
        return f'AsyncAstraDBCollection[astra_db="{self.astra_db}", collection_name="{self.collection_name}"]'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AsyncAstraDBCollection):
            return all(
                [
                    self.collection_name == other.collection_name,
                    self.astra_db == other.astra_db,
                    self.caller_name == other.caller_name,
                    self.caller_version == other.caller_version,
                ]
            )
        else:
            return False

    def copy(
        self,
        *,
        collection_name: Optional[str] = None,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncAstraDBCollection:
        return AsyncAstraDBCollection(
            collection_name=collection_name or self.collection_name,
            astra_db=self.astra_db.copy(
                token=token,
                api_endpoint=api_endpoint,
                api_path=api_path,
                api_version=api_version,
                namespace=namespace,
                caller_name=caller_name,
                caller_version=caller_version,
            ),
            caller_name=caller_name or self.caller_name,
            caller_version=caller_version or self.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self.astra_db.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )
        self.caller_name = caller_name
        self.caller_version = caller_version

    def to_sync(self) -> AstraDBCollection:
        return AstraDBCollection(
            collection_name=self.collection_name,
            astra_db=self.astra_db.to_sync(),
            caller_name=self.caller_name,
            caller_version=self.caller_version,
        )

    async def _request(
        self,
        method: str = http_methods.POST,
        path: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        url_params: Optional[Dict[str, Any]] = None,
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
        **kwargs: Any,
    ) -> API_RESPONSE:
        adirect_response = await async_api_request(
            client=self.client,
            base_url=self.astra_db.base_url,
            auth_header=DEFAULT_AUTH_HEADER,
            token=self.astra_db.token,
            method=method,
            json_data=normalize_for_api(json_data),
            url_params=url_params,
            path=path,
            skip_error_check=skip_error_check,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
            timeout=to_httpx_timeout(timeout_info),
        )
        response = restore_from_api(adirect_response)
        return response

    async def post_raw_request(
        self, body: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        return await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=body,
            timeout_info=timeout_info,
        )

    async def _get(
        self,
        path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Optional[API_RESPONSE]:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = await self._request(
            method=http_methods.GET,
            path=full_path,
            url_params=options,
            timeout_info=timeout_info,
        )
        if isinstance(response, dict):
            return response
        return None

    async def _put(
        self,
        path: Optional[str] = None,
        document: Optional[API_RESPONSE] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = await self._request(
            method=http_methods.PUT,
            path=full_path,
            json_data=document,
            timeout_info=timeout_info,
        )
        return response

    async def _post(
        self,
        path: Optional[str] = None,
        document: Optional[API_DOC] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        response = await self._request(
            method=http_methods.POST,
            path=full_path,
            json_data=document,
            timeout_info=timeout_info,
        )
        return response

    def _recast_as_sort_projection(
        self, vector: List[float], fields: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Given a vector and optionally a list of fields,
        reformulate them as a sort, projection pair for regular
        'find'-like API calls (with basic validation as well).
        """
        # Must pass a vector
        if not vector:
            raise ValueError("Must pass a vector")

        # Edge case for field selection
        if fields and "$similarity" in fields:
            raise ValueError("Please use the `include_similarity` parameter")

        # Build the new vector parameter
        sort: Dict[str, Any] = {"$vector": vector}

        # Build the new fields parameter
        # Note: do not leave projection={}, make it None
        # (or it will devour $similarity away in the API response)
        if fields is not None and len(fields) > 0:
            projection = {f: 1 for f in fields}
        else:
            projection = None

        return sort, projection

    async def get(
        self, path: Optional[str] = None, timeout_info: TimeoutInfoWideType = None
    ) -> Optional[API_RESPONSE]:
        """
        Retrieve a document from the collection by its path.

        Args:
            path (str, optional): The path of the document to retrieve.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The retrieved document.
        """
        return await self._get(path=path, timeout_info=timeout_info)

    async def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find documents in the collection that match the given filter.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return matching documents.
            options (dict, optional): Additional options for the query.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The query response containing matched documents.
        """
        json_query = make_payload(
            top_level="find",
            filter=filter,
            projection=projection,
            options=options,
            sort=sort,
        )

        response = await self._post(document=json_query, timeout_info=timeout_info)

        return response

    async def vector_find(
        self,
        vector: List[float],
        *,
        limit: int,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        include_similarity: bool = True,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[API_DOC]:
        """
        Perform a vector-based search in the collection.

        Args:
            vector (list): The vector to search with.
            limit (int): The maximum number of documents to return.
            filter (dict, optional): Criteria to filter documents.
            fields (list, optional): Specifies the fields to return.
            include_similarity (bool, optional): Whether to include similarity score in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            list: A list of documents matching the vector search criteria.
        """
        # Must pass a limit
        if not limit:
            raise ValueError("Must pass a limit")

        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            vector,
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = await self.find(
            filter=filter,
            projection=projection,
            sort=sort,
            options={
                "limit": limit,
                "includeSimilarity": include_similarity,
            },
            timeout_info=timeout_info,
        )

        return cast(List[API_DOC], raw_find_result["data"]["documents"])

    @staticmethod
    async def paginate(
        *,
        request_method: AsyncPaginableRequestMethod,
        options: Optional[Dict[str, Any]],
        prefetched: Optional[int] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> AsyncGenerator[API_DOC, None]:
        """
        Generate paginated results for a given database query method.

        Args:
            request_method (function): The database query method to paginate.
            options (dict, optional): Options for the database query.
            prefetched (int, optional): Number of pre-fetched documents.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Yields:
            dict: The next document in the paginated result set.
        """
        _options = options or {}
        response0 = await request_method(options=_options)
        next_page_state = response0["data"]["nextPageState"]
        options0 = _options
        if next_page_state is not None and prefetched:

            async def queued_paginate(
                queue: asyncio.Queue[Optional[API_DOC]],
                request_method: AsyncPaginableRequestMethod,
                options: Optional[Dict[str, Any]],
            ) -> None:
                try:
                    async for doc in AsyncAstraDBCollection.paginate(
                        request_method=request_method, options=options
                    ):
                        await queue.put(doc)
                finally:
                    await queue.put(None)

            queue: asyncio.Queue[Optional[API_DOC]] = asyncio.Queue(prefetched)
            options1 = {**options0, **{"pageState": next_page_state}}
            asyncio.create_task(queued_paginate(queue, request_method, options1))
            for document in response0["data"]["documents"]:
                yield document
            doc = await queue.get()
            while doc is not None:
                yield doc
                doc = await queue.get()
        else:
            for document in response0["data"]["documents"]:
                yield document
            while next_page_state is not None:
                options1 = {**options0, **{"pageState": next_page_state}}
                response1 = await request_method(options=options1)
                for document in response1["data"]["documents"]:
                    yield document
                next_page_state = response1["data"]["nextPageState"]

    def paginated_find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        prefetched: Optional[int] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> AsyncIterator[API_DOC]:
        """
        Perform a paginated search in the collection.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return matching documents.
            options (dict, optional): Additional options for the query.
            prefetched (int, optional): Number of pre-fetched documents
            timeout_info: a float, or a TimeoutInfo dict, for each
                single HTTP request.
                This is a paginated method, that issues several requests as it
                needs more data. This parameter controls a single request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            generator: A generator yielding documents in the paginated result set.
        """
        partialed_find = partial(
            self.find,
            filter=filter,
            projection=projection,
            sort=sort,
            timeout_info=timeout_info,
        )
        return self.paginate(
            request_method=partialed_find,
            options=options,
            prefetched=prefetched,
        )

    async def pop(
        self,
        filter: Dict[str, Any],
        pop: Dict[str, Any],
        options: Dict[str, Any],
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Pop the last data in the tags array

        Args:
            filter (dict): Criteria to identify the document to update.
            pop (dict): The pop to apply to the tags.
            options (dict): Additional options for the update operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The original document before the update.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update={"$pop": pop},
            options=options,
        )

        response = await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def push(
        self,
        filter: Dict[str, Any],
        push: Dict[str, Any],
        options: Dict[str, Any],
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Push new data to the tags array

        Args:
            filter (dict): Criteria to identify the document to update.
            push (dict): The push to apply to the tags.
            options (dict): Additional options for the update operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the update operation.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update={"$push": push},
            options=options,
        )

        response = await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def find_one_and_replace(
        self,
        replacement: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and replace it.

        Args:
            replacement (dict): The new document to replace the existing one.
            filter (dict, optional): Criteria to filter documents.
            sort (dict, optional): Specifies the order in which to find the document.
            options (dict, optional): Additional options for the operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and replace operation.
        """
        json_query = make_payload(
            top_level="findOneAndReplace",
            filter=filter,
            projection=projection,
            replacement=replacement,
            options=options,
            sort=sort,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def vector_find_one_and_replace(
        self,
        vector: List[float],
        replacement: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search and replace the first matched document.

        Args:
            vector (dict): The vector to search with.
            replacement (dict): The new document to replace the existing one.
            filter (dict, optional): Criteria to filter documents.
            fields (list, optional): Specifies the fields to return in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: either the matched document or None if nothing found
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            vector,
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = await self.find_one_and_replace(
            replacement=replacement,
            filter=filter,
            projection=projection,
            sort=sort,
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    async def find_one_and_update(
        self,
        update: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = {},
        filter: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and update it.

        Args:
            sort (dict, optional): Specifies the order in which to find the document.
            update (dict): The update to apply to the document.
            filter (dict, optional): Criteria to filter documents.
            options (dict, optional): Additional options for the operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and update operation.
        """
        json_query = make_payload(
            top_level="findOneAndUpdate",
            filter=filter,
            update=update,
            options=options,
            sort=sort,
            projection=projection,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def vector_find_one_and_update(
        self,
        vector: List[float],
        update: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search and update the first matched document.

        Args:
            vector (list): The vector to search with.
            update (dict): The update to apply to the matched document.
            filter (dict, optional): Criteria to filter documents before applying the vector search.
            fields (list, optional): Specifies the fields to return in the updated document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: The result of the vector-based find and
                update operation, or None if nothing found
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            vector,
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = await self.find_one_and_update(
            update=update,
            filter=filter,
            sort=sort,
            projection=projection,
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    async def find_one_and_delete(
        self,
        sort: Optional[Dict[str, Any]] = {},
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document and delete it.

        Args:
            sort (dict, optional): Specifies the order in which to find the document.
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The result of the find and delete operation.
        """
        json_query = make_payload(
            top_level="findOneAndDelete",
            filter=filter,
            sort=sort,
            projection=projection,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def count_documents(
        self, filter: Dict[str, Any] = {}, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Count documents matching a given predicate (expressed as filter).

        Args:
            filter (dict, defaults to {}): Criteria to filter documents.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: the response, either
                {"status": {"count": <NUMBER> }}
            or
                {"errors": [...]}
        """
        json_query = make_payload(
            top_level="countDocuments",
            filter=filter,
        )

        response = await self._post(document=json_query, timeout_info=timeout_info)

        return response

    async def find_one(
        self,
        filter: Optional[Dict[str, Any]] = {},
        projection: Optional[Dict[str, Any]] = {},
        sort: Optional[Dict[str, Any]] = {},
        options: Optional[Dict[str, Any]] = {},
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Find a single document in the collection.

        Args:
            filter (dict, optional): Criteria to filter documents.
            projection (dict, optional): Specifies the fields to return.
            sort (dict, optional): Specifies the order in which to return the document.
            options (dict, optional): Additional options for the query.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: the response, either
                {"data": {"document": <DOCUMENT> }}
            or
                {"data": {"document": None}}
            depending on whether a matching document is found or not.
        """
        json_query = make_payload(
            top_level="findOne",
            filter=filter,
            projection=projection,
            options=options,
            sort=sort,
        )

        response = await self._post(document=json_query, timeout_info=timeout_info)

        return response

    async def vector_find_one(
        self,
        vector: List[float],
        *,
        filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        include_similarity: bool = True,
        timeout_info: TimeoutInfoWideType = None,
    ) -> Union[API_DOC, None]:
        """
        Perform a vector-based search to find a single document in the collection.

        Args:
            vector (list): The vector to search with.
            filter (dict, optional): Additional criteria to filter documents.
            fields (list, optional): Specifies the fields to return in the result.
            include_similarity (bool, optional): Whether to include similarity score in the result.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict or None: The found document or None if no matching document is found.
        """
        # Pre-process the included arguments
        sort, projection = self._recast_as_sort_projection(
            vector,
            fields=fields,
        )

        # Call the underlying find() method to search
        raw_find_result = await self.find_one(
            filter=filter,
            projection=projection,
            sort=sort,
            options={"includeSimilarity": include_similarity},
            timeout_info=timeout_info,
        )

        return cast(Union[API_DOC, None], raw_find_result["data"]["document"])

    async def insert_one(
        self,
        document: API_DOC,
        failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Insert a single document into the collection.

        Args:
            document (dict): The document to insert.
            failures_allowed (bool): Whether to allow failures in the insert operation.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the insert operation.
        """
        json_query = make_payload(top_level="insertOne", document=document)

        response = await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            skip_error_check=failures_allowed,
            timeout_info=timeout_info,
        )

        return response

    async def insert_many(
        self,
        documents: List[API_DOC],
        options: Optional[Dict[str, Any]] = None,
        partial_failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Insert multiple documents into the collection.

        Args:
            documents (list): A list of documents to insert.
            options (dict, optional): Additional options for the insert operation.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures through the insertion (i.e. on some documents).
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the insert operation.
        """
        json_query = make_payload(
            top_level="insertMany", documents=documents, options=options
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            skip_error_check=partial_failures_allowed,
            timeout_info=timeout_info,
        )

        return response

    async def chunked_insert_many(
        self,
        documents: List[API_DOC],
        options: Optional[Dict[str, Any]] = None,
        partial_failures_allowed: bool = False,
        chunk_size: int = MAX_INSERT_NUM_DOCUMENTS,
        concurrency: int = 1,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[Union[API_RESPONSE, Exception]]:
        """
        Insert multiple documents into the collection, handling chunking and
        optionally with concurrent insertions.

        Args:
            documents (list): A list of documents to insert.
            options (dict, optional): Additional options for the insert operation.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures in the chunk. Should be used combined with
                options={"ordered": False} in most cases.
            chunk_size (int, optional): Override the default insertion chunk size.
            concurrency (int, optional): The number of concurrent chunk insertions.
                Default is no concurrency.
            timeout_info: a float, or a TimeoutInfo dict, for each single HTTP request.
                This method runs a number of HTTP requests as it works on chunked
                data. The timeout refers to each individual such request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            list: The responses from the database after the chunked insert operation.
                This is a list of individual responses from the API: the caller
                will need to inspect them all, e.g. to collate the inserted IDs.
        """
        sem = asyncio.Semaphore(concurrency)

        async def concurrent_insert_many(
            docs: List[API_DOC],
            index: int,
            partial_failures_allowed: bool,
        ) -> Union[API_RESPONSE, Exception]:
            async with sem:
                logger.debug(f"Processing chunk #{index + 1} of size {len(docs)}")
                try:
                    return await self.insert_many(
                        documents=docs,
                        options=options,
                        partial_failures_allowed=partial_failures_allowed,
                        timeout_info=timeout_info,
                    )
                except APIRequestError as e:
                    if partial_failures_allowed:
                        return e
                    else:
                        raise e

        if concurrency > 1:
            tasks = [
                asyncio.create_task(
                    concurrent_insert_many(
                        documents[i : i + chunk_size], i, partial_failures_allowed
                    )
                )
                for i in range(0, len(documents), chunk_size)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # this ensures the expectation of
            # "sequential strictly obeys fail-fast if ordered and concurrency==1"
            results = [
                await concurrent_insert_many(
                    documents[i : i + chunk_size], i, partial_failures_allowed
                )
                for i in range(0, len(documents), chunk_size)
            ]
        return results

    async def update_one(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Update a single document in the collection.

        Args:
            filter (dict): Criteria to identify the document to update.
            update (dict): The update to apply to the document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = make_payload(
            top_level="updateOne",
            filter=filter,
            update=update,
            sort=sort,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def update_many(
        self,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Updates multiple documents in the collection.

        Args:
            filter (dict): Criteria to identify the document to update.
            update (dict): The update to apply to the document.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = make_payload(
            top_level="updateMany",
            filter=filter,
            update=update,
            options=options,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def replace(
        self, path: str, document: API_DOC, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Replace a document in the collection.

        Args:
            path (str): The path to the document to replace.
            document (dict): The new document to replace the existing one.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the replace operation.
        """
        return await self._put(path=path, document=document, timeout_info=timeout_info)

    async def delete_one(
        self,
        id: str,
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete a single document from the collection based on its ID.

        Args:
            id (str): The ID of the document to delete.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = make_payload(
            top_level="deleteOne",
            filter={"_id": id},
            sort=sort,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def delete_one_by_predicate(
        self,
        filter: Dict[str, Any],
        sort: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete a single document from the collection based on a filter clause

        Args:
            filter: any filter dictionary
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = make_payload(
            top_level="deleteOne",
            filter=filter,
            sort=sort,
        )

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def delete_many(
        self,
        filter: Dict[str, Any],
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Delete many documents from the collection based on a filter condition

        Args:
            filter (dict): Criteria to identify the documents to delete.
            skip_error_check (bool): whether to ignore the check for API error
                and return the response untouched. Default is False.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the delete operation.
        """
        json_query = {
            "deleteMany": {
                "filter": filter,
            }
        }

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            skip_error_check=skip_error_check,
            timeout_info=timeout_info,
        )

        return response

    async def chunked_delete_many(
        self, filter: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> List[API_RESPONSE]:
        """
        Delete many documents from the collection based on a filter condition,
        chaining several API calls until exhaustion of the documents to delete.

        Args:
            filter (dict): Criteria to identify the documents to delete.
            timeout_info: a float, or a TimeoutInfo dict, for each single HTTP request.
                This method runs a number of HTTP requests as it works on a
                pagination basis. The timeout refers to each individual such request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            List[dict]: The responses from the database from all the calls
        """
        responses = []
        must_proceed = True
        while must_proceed:
            dm_response = await self.delete_many(
                filter=filter, timeout_info=timeout_info
            )
            responses.append(dm_response)
            must_proceed = dm_response.get("status", {}).get("moreData", False)
        return responses

    async def clear(self, timeout_info: TimeoutInfoWideType = None) -> API_RESPONSE:
        """
        Clear the collection, deleting all documents

        Args:
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database.
        """
        clear_response = await self.delete_many(filter={}, timeout_info=timeout_info)

        if clear_response.get("status", {}).get("deletedCount") != -1:
            raise ValueError(
                f"Could not issue a clear-collection API command (response: {json.dumps(clear_response)})."
            )

        return clear_response

    async def delete_subdocument(
        self, id: str, subdoc: str, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Delete a subdocument or field from a document in the collection.

        Args:
            id (str): The ID of the document containing the subdocument.
            subdoc (str): The key of the subdocument or field to remove.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database after the update operation.
        """
        json_query = {
            "findOneAndUpdate": {
                "filter": {"_id": id},
                "update": {"$unset": {subdoc: ""}},
            }
        }

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    @deprecation.deprecated(  # type: ignore
        deprecated_in="0.7.0",
        removed_in="1.0.0",
        current_version=__version__,
        details="Use the 'upsert_one' method instead",
    )
    async def upsert(
        self, document: API_DOC, timeout_info: TimeoutInfoWideType = None
    ) -> str:
        return await self.upsert_one(document, timeout_info=timeout_info)

    async def upsert_one(
        self,
        document: API_DOC,
        timeout_info: TimeoutInfoWideType = None,
    ) -> str:
        """
        Emulate an upsert operation for a single document in the collection.

        This method attempts to insert the document.
        If a document with the same _id exists, it updates the existing document.

        Args:
            document (dict): The document to insert or update.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP requests.
                This method may issue one or two requests, depending on what
                is detected on DB. This timeout controls each HTTP request individually.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            str: The _id of the inserted or updated document.
        """
        # Build the payload for the insert attempt
        result = await self.insert_one(
            document, failures_allowed=True, timeout_info=timeout_info
        )

        # If the call failed because of preexisting doc, then we replace it
        if "errors" in result:
            if (
                "errorCode" in result["errors"][0]
                and result["errors"][0]["errorCode"] == "DOCUMENT_ALREADY_EXISTS"
            ):
                # Now we attempt the update
                result = await self.find_one_and_replace(
                    replacement=document,
                    filter={"_id": document["_id"]},
                    timeout_info=timeout_info,
                )
                upserted_id = cast(str, result["data"]["document"]["_id"])
            else:
                raise ValueError(result)
        else:
            if result.get("status", {}).get("insertedIds", []):
                upserted_id = cast(str, result["status"]["insertedIds"][0])
            else:
                raise ValueError("Unexplained empty insertedIds from API")

        return upserted_id

    async def upsert_many(
        self,
        documents: list[API_DOC],
        concurrency: int = 1,
        partial_failures_allowed: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> List[Union[str, Exception]]:
        """
        Emulate an upsert operation for multiple documents in the collection.
        This method attempts to insert the documents.
        If a document with the same _id exists, it updates the existing document.

        Args:
            documents (List[dict]): The documents to insert or update.
            concurrency (int, optional): The number of concurrent upserts.
            partial_failures_allowed (bool, optional): Whether to allow partial
                failures in the batch.
            timeout_info: a float, or a TimeoutInfo dict, for each HTTP request.
                This method issues a separate HTTP request for each document to
                insert: the timeout controls each such request individually.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            List[Union[str, Exception]]: A list of "_id"s of the inserted or updated documents.
        """
        sem = asyncio.Semaphore(concurrency)

        async def concurrent_upsert(doc: API_DOC) -> str:
            async with sem:
                return await self.upsert_one(document=doc, timeout_info=timeout_info)

        tasks = [asyncio.create_task(concurrent_upsert(doc)) for doc in documents]
        results = await asyncio.gather(
            *tasks, return_exceptions=partial_failures_allowed
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        return results  # type: ignore


class AstraDB:
    # Initialize the shared httpx client as a class attribute
    client = httpx.Client()

    def __init__(
        self,
        token: str,
        api_endpoint: str,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Initialize an Astra DB instance.

        Args:
            token (str): Authentication token for Astra DB.
            api_endpoint (str): API endpoint URL.
            api_path (str, optional): used to override default URI construction
            api_version (str, optional): to override default URI construction
            namespace (str, optional): Namespace for the database.
            caller_name (str, optional): identity of the caller ("my_framework")
            caller_version (str, optional): version of the caller code ("1.0.3")
        """
        self.caller_name = caller_name
        self.caller_version = caller_version

        if token is None or api_endpoint is None:
            raise AssertionError("Must provide token and api_endpoint")

        if namespace is None:
            logger.info(
                f"ASTRA_DB_KEYSPACE is not set. Defaulting to '{DEFAULT_KEYSPACE_NAME}'"
            )
            namespace = DEFAULT_KEYSPACE_NAME

        # Store the API token
        self.token = token

        self.api_endpoint = api_endpoint

        # Set the Base URL for the API calls
        self.base_url = self.api_endpoint.strip("/")

        # Set the API version and path from the call
        self.api_path = (api_path or DEFAULT_JSON_API_PATH).strip("/")
        self.api_version = (api_version or DEFAULT_JSON_API_VERSION).strip("/")

        # Set the namespace
        self.namespace = namespace

        # Finally, construct the full base path
        self.base_path: str = f"/{self.api_path}/{self.api_version}/{self.namespace}"

    def __repr__(self) -> str:
        return f'AstraDB[endpoint="{self.base_url}", keyspace="{self.namespace}"]'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AstraDB):
            # work on the "normalized" quantities (stripped, etc)
            return all(
                [
                    self.token == other.token,
                    self.base_url == other.base_url,
                    self.base_path == other.base_path,
                    self.caller_name == other.caller_name,
                    self.caller_version == other.caller_version,
                ]
            )
        else:
            return False

    def copy(
        self,
        *,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AstraDB:
        return AstraDB(
            token=token or self.token,
            api_endpoint=api_endpoint or self.base_url,
            api_path=api_path or self.api_path,
            api_version=api_version or self.api_version,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self.caller_name,
            caller_version=caller_version or self.caller_version,
        )

    def to_async(self) -> AsyncAstraDB:
        return AsyncAstraDB(
            token=self.token,
            api_endpoint=self.base_url,
            api_path=self.api_path,
            api_version=self.api_version,
            namespace=self.namespace,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self.caller_name = caller_name
        self.caller_version = caller_version

    def _request(
        self,
        method: str = http_methods.POST,
        path: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        url_params: Optional[Dict[str, Any]] = None,
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        direct_response = api_request(
            client=self.client,
            base_url=self.base_url,
            auth_header=DEFAULT_AUTH_HEADER,
            token=self.token,
            method=method,
            json_data=normalize_for_api(json_data),
            url_params=url_params,
            path=path,
            skip_error_check=skip_error_check,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
            timeout=to_httpx_timeout(timeout_info),
        )
        response = restore_from_api(direct_response)
        return response

    def post_raw_request(
        self, body: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        return self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=body,
            timeout_info=timeout_info,
        )

    def collection(self, collection_name: str) -> AstraDBCollection:
        """
        Retrieve a collection from the database.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            AstraDBCollection: The collection object.
        """
        return AstraDBCollection(collection_name=collection_name, astra_db=self)

    def get_collections(
        self,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Retrieve a list of collections from the database.

        Args:
            options (dict, optional): Options to get the collection list
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: An object containing the list of collections in the database:
                {"status": {"collections": [...]}}
        """
        # Parse the options parameter
        if options is None:
            options = {}

        json_query = make_payload(
            top_level="findCollections",
            options=options,
        )

        response = self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    def create_collection(
        self,
        collection_name: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        service_dict: Optional[Dict[str, str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> AstraDBCollection:
        """
        Create a new collection in the database.

        Args:
            collection_name (str): The name of the collection to create.
            options (dict, optional): Options for the collection.
            dimension (int, optional): Dimension for vector search.
            metric (str, optional): Metric choice for vector search.
            service_dict (dict, optional): a definition for the $vectorize service
                NOTE: This feature is under current development.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            AstraDBCollection: The created collection object.
        """
        # options from named params
        vector_options = {
            k: v
            for k, v in {
                "dimension": dimension,
                "metric": metric,
                "service": service_dict,
            }.items()
            if v is not None
        }

        # overlap/merge with stuff in options.vector
        dup_params = set((options or {}).get("vector", {}).keys()) & set(
            vector_options.keys()
        )

        # If any params are duplicated, we raise an error
        if dup_params:
            dups = ", ".join(sorted(dup_params))
            raise ValueError(
                f"Parameter(s) {dups} passed both to the method and in the options"
            )

        # Build our options dictionary if we have vector options
        if vector_options:
            options = options or {}
            options["vector"] = {
                **options.get("vector", {}),
                **vector_options,
            }

        # Build the final json payload
        jsondata = {
            k: v
            for k, v in {"name": collection_name, "options": options}.items()
            if v is not None
        }

        # Make the request to the endpoint
        self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data={"createCollection": jsondata},
            timeout_info=timeout_info,
        )

        # Get the instance object as the return of the call
        return AstraDBCollection(astra_db=self, collection_name=collection_name)

    def delete_collection(
        self, collection_name: str, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Delete a collection from the database.

        Args:
            collection_name (str): The name of the collection to delete.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database.
        """
        # Make sure we provide a collection name
        if not collection_name:
            raise ValueError("Must provide a collection name")

        response = self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data={"deleteCollection": {"name": collection_name}},
            timeout_info=timeout_info,
        )

        return response

    @deprecation.deprecated(  # type: ignore
        deprecated_in="0.7.0",
        removed_in="1.0.0",
        current_version=__version__,
        details="Use the 'AstraDBCollection.clear()' method instead",
    )
    def truncate_collection(
        self, collection_name: str, timeout_info: TimeoutInfoWideType = None
    ) -> AstraDBCollection:
        """
        Clear a collection in the database, deleting all stored documents.

        Args:
            collection_name (str): The name of the collection to clear.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            collection: an AstraDBCollection instance
        """
        collection = AstraDBCollection(
            collection_name=collection_name,
            astra_db=self,
        )
        clear_response = collection.clear(timeout_info=timeout_info)

        if clear_response.get("status", {}).get("deletedCount") != -1:
            raise ValueError(
                f"Could not issue a truncation API command (response: {json.dumps(clear_response)})."
            )

        # return the collection itself
        return collection


class AsyncAstraDB:
    def __init__(
        self,
        token: str,
        api_endpoint: str,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Initialize an Astra DB instance.

        Args:
            token (str): Authentication token for Astra DB.
            api_endpoint (str): API endpoint URL.
            api_path (str, optional): used to override default URI construction
            api_version (str, optional): to override default URI construction
            namespace (str, optional): Namespace for the database.
            caller_name (str, optional): identity of the caller ("my_framework")
            caller_version (str, optional): version of the caller code ("1.0.3")
        """
        self.caller_name = caller_name
        self.caller_version = caller_version

        self.client = httpx.AsyncClient()
        if token is None or api_endpoint is None:
            raise AssertionError("Must provide token and api_endpoint")

        if namespace is None:
            logger.info(
                f"ASTRA_DB_KEYSPACE is not set. Defaulting to '{DEFAULT_KEYSPACE_NAME}'"
            )
            namespace = DEFAULT_KEYSPACE_NAME

        # Store the API token
        self.token = token

        self.api_endpoint = api_endpoint

        # Set the Base URL for the API calls
        self.base_url = self.api_endpoint.strip("/")

        # Set the API version and path from the call
        self.api_path = (api_path or DEFAULT_JSON_API_PATH).strip("/")
        self.api_version = (api_version or DEFAULT_JSON_API_VERSION).strip("/")

        # Set the namespace
        self.namespace = namespace

        # Finally, construct the full base path
        self.base_path: str = f"/{self.api_path}/{self.api_version}/{self.namespace}"

    def __repr__(self) -> str:
        return f'AsyncAstraDB[endpoint="{self.base_url}", keyspace="{self.namespace}"]'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AsyncAstraDB):
            # work on the "normalized" quantities (stripped, etc)
            return all(
                [
                    self.token == other.token,
                    self.base_url == other.base_url,
                    self.base_path == other.base_path,
                    self.caller_name == other.caller_name,
                    self.caller_version == other.caller_version,
                ]
            )
        else:
            return False

    async def __aenter__(self) -> AsyncAstraDB:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        await self.client.aclose()

    def copy(
        self,
        *,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncAstraDB:
        return AsyncAstraDB(
            token=token or self.token,
            api_endpoint=api_endpoint or self.base_url,
            api_path=api_path or self.api_path,
            api_version=api_version or self.api_version,
            namespace=namespace or self.namespace,
            caller_name=caller_name or self.caller_name,
            caller_version=caller_version or self.caller_version,
        )

    def to_sync(self) -> AstraDB:
        return AstraDB(
            token=self.token,
            api_endpoint=self.base_url,
            api_path=self.api_path,
            api_version=self.api_version,
            namespace=self.namespace,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self.caller_name = caller_name
        self.caller_version = caller_version

    async def _request(
        self,
        method: str = http_methods.POST,
        path: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        url_params: Optional[Dict[str, Any]] = None,
        skip_error_check: bool = False,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        adirect_response = await async_api_request(
            client=self.client,
            base_url=self.base_url,
            auth_header=DEFAULT_AUTH_HEADER,
            token=self.token,
            method=method,
            json_data=normalize_for_api(json_data),
            url_params=url_params,
            path=path,
            skip_error_check=skip_error_check,
            caller_name=self.caller_name,
            caller_version=self.caller_version,
            timeout=to_httpx_timeout(timeout_info),
        )
        response = restore_from_api(adirect_response)
        return response

    async def post_raw_request(
        self, body: Dict[str, Any], timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        return await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=body,
            timeout_info=timeout_info,
        )

    async def collection(self, collection_name: str) -> AsyncAstraDBCollection:
        """
        Retrieve a collection from the database.

        Args:
            collection_name (str): The name of the collection to retrieve.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            AstraDBCollection: The collection object.
        """
        return AsyncAstraDBCollection(collection_name=collection_name, astra_db=self)

    async def get_collections(
        self,
        options: Optional[Dict[str, Any]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> API_RESPONSE:
        """
        Retrieve a list of collections from the database.

        Args:
            options (dict, optional): Options to get the collection list
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: An object containing the list of collections in the database:
                {"status": {"collections": [...]}}
        """
        # Parse the options parameter
        if options is None:
            options = {}

        json_query = make_payload(
            top_level="findCollections",
            options=options,
        )

        response = await self._request(
            method=http_methods.POST,
            path=self.base_path,
            json_data=json_query,
            timeout_info=timeout_info,
        )

        return response

    async def create_collection(
        self,
        collection_name: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        service_dict: Optional[Dict[str, str]] = None,
        timeout_info: TimeoutInfoWideType = None,
    ) -> AsyncAstraDBCollection:
        """
        Create a new collection in the database.

        Args:
            collection_name (str): The name of the collection to create.
            options (dict, optional): Options for the collection.
            dimension (int, optional): Dimension for vector search.
            metric (str, optional): Metric choice for vector search.
            service_dict (dict, optional): a definition for the $vectorize service
                NOTE: This feature is under current development.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            AsyncAstraDBCollection: The created collection object.
        """
        # options from named params
        vector_options = {
            k: v
            for k, v in {
                "dimension": dimension,
                "metric": metric,
                "service": service_dict,
            }.items()
            if v is not None
        }

        # overlap/merge with stuff in options.vector
        dup_params = set((options or {}).get("vector", {}).keys()) & set(
            vector_options.keys()
        )

        # If any params are duplicated, we raise an error
        if dup_params:
            dups = ", ".join(sorted(dup_params))
            raise ValueError(
                f"Parameter(s) {dups} passed both to the method and in the options"
            )

        # Build our options dictionary if we have vector options
        if vector_options:
            options = options or {}
            options["vector"] = {
                **options.get("vector", {}),
                **vector_options,
            }

        # Build the final json payload
        jsondata = {
            k: v
            for k, v in {"name": collection_name, "options": options}.items()
            if v is not None
        }

        # Make the request to the endpoint
        await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data={"createCollection": jsondata},
            timeout_info=timeout_info,
        )

        # Get the instance object as the return of the call
        return AsyncAstraDBCollection(astra_db=self, collection_name=collection_name)

    async def delete_collection(
        self, collection_name: str, timeout_info: TimeoutInfoWideType = None
    ) -> API_RESPONSE:
        """
        Delete a collection from the database.

        Args:
            collection_name (str): The name of the collection to delete.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            dict: The response from the database.
        """
        # Make sure we provide a collection name
        if not collection_name:
            raise ValueError("Must provide a collection name")

        response = await self._request(
            method=http_methods.POST,
            path=f"{self.base_path}",
            json_data={"deleteCollection": {"name": collection_name}},
            timeout_info=timeout_info,
        )

        return response

    @deprecation.deprecated(  # type: ignore
        deprecated_in="0.7.0",
        removed_in="1.0.0",
        current_version=__version__,
        details="Use the 'AsyncAstraDBCollection.clear()' method instead",
    )
    async def truncate_collection(
        self, collection_name: str, timeout_info: TimeoutInfoWideType = None
    ) -> AsyncAstraDBCollection:
        """
        Clear a collection in the database, deleting all stored documents.

        Args:
            collection_name (str): The name of the collection to clear.
            timeout_info: a float, or a TimeoutInfo dict, for the HTTP request.
                Note that a 'read' timeout event will not block the action taken
                by the API server if it has received the request already.

        Returns:
            collection: an AsyncAstraDBCollection instance
        """

        collection = AsyncAstraDBCollection(
            collection_name=collection_name,
            astra_db=self,
        )
        clear_response = await collection.clear(timeout_info=timeout_info)

        if clear_response.get("status", {}).get("deletedCount") != -1:
            raise ValueError(
                f"Could not issue a truncation API command (response: {json.dumps(clear_response)})."
            )

        # return the collection itself
        return collection
