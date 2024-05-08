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

import logging

from types import TracebackType
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING

from astrapy.core.db import AstraDB, AsyncAstraDB
from astrapy.exceptions import (
    CollectionAlreadyExistsException,
    DataAPIFaultyResponseException,
    DevOpsAPIException,
    MultiCallTimeoutManager,
    recast_method_sync,
    recast_method_async,
    base_timeout_info,
)
from astrapy.cursors import AsyncCommandCursor, CommandCursor
from astrapy.info import (
    DatabaseInfo,
    CollectionDescriptor,
    CollectionVectorServiceOptions,
)
from astrapy.admin import parse_api_endpoint, fetch_database_info

if TYPE_CHECKING:
    from astrapy.collection import AsyncCollection, Collection
    from astrapy.admin import AstraDBDatabaseAdmin


logger = logging.getLogger(__name__)


def _validate_create_collection_options(
    dimension: Optional[int],
    metric: Optional[str],
    service: Optional[Union[CollectionVectorServiceOptions, Dict[str, Any]]],
    indexing: Optional[Dict[str, Any]],
    default_id_type: Optional[str],
    additional_options: Optional[Dict[str, Any]],
) -> None:
    if additional_options:
        if "vector" in additional_options:
            raise ValueError(
                "`additional_options` dict parameter to create_collection "
                "cannot have a `vector` key. Please use the specific "
                "method parameter."
            )
        if "indexing" in additional_options:
            raise ValueError(
                "`additional_options` dict parameter to create_collection "
                "cannot have a `indexing` key. Please use the specific "
                "method parameter."
            )
        if "defaultId" in additional_options and default_id_type is not None:
            # this leaves the workaround to pass more info in the defaultId
            # should that become part of the specs:
            raise ValueError(
                "`additional_options` dict parameter to create_collection "
                "cannot have a `defaultId` key when passing the "
                "`default_id_type` parameter as well."
            )
    is_vector: bool
    if service is not None or dimension is not None:
        is_vector = True
    else:
        is_vector = False
    if not is_vector and metric is not None:
        raise ValueError(
            "Cannot specify `metric` for non-vector collections in the "
            "create_collection method."
        )


class Database:
    """
    A Data API database. This is the entry-point object for doing database-level
    DML, such as creating/deleting collections, and for obtaining Collection
    objects themselves. This class has a synchronous interface.

    A Database comes with an "API Endpoint", which implies a Database object
    instance reaches a specific region (relevant point in case of multi-region
    databases).

    Args:
        api_endpoint: the full "API Endpoint" string used to reach the Data API.
            Example: "https://<database_id>-<region>.apps.astra.datastax.com"
        token: an Access Token to the database. Example: "AstraCS:xyz..."
        namespace: this is the namespace all method calls will target, unless
            one is explicitly specified in the call. If no namespace is supplied
            when creating a Database, the name "default_namespace" is set.
        caller_name: name of the application, or framework, on behalf of which
            the Data API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.
        api_path: path to append to the API Endpoint. In typical usage, this
            should be left to its default of "/api/json".
        api_version: version specifier to append to the API path. In typical
            usage, this should be left to its default of "v1".

    Example:
        >>> from astrapy import DataAPIClient
        >>> my_client = astrapy.DataAPIClient("AstraCS:...")
        >>> my_db = my_client.get_database_by_api_endpoint(
        ...    "https://01234567-....apps.astra.datastax.com"
        ... )

    Note:
        creating an instance of Database does not trigger actual creation
        of the database itself, which should exist beforehand. To create databases,
        see the AstraDBAdmin class.
    """

    def __init__(
        self,
        api_endpoint: str,
        token: str,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self._astra_db = AstraDB(
            token=token,
            api_endpoint=api_endpoint,
            api_path=api_path,
            api_version=api_version,
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )
        self._name: Optional[str] = None

    def __getattr__(self, collection_name: str) -> Collection:
        return self.get_collection(name=collection_name)

    def __getitem__(self, collection_name: str) -> Collection:
        return self.get_collection(name=collection_name)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(api_endpoint="{self._astra_db.api_endpoint}", '
            f'token="{self._astra_db.token[:12]}...", namespace="{self._astra_db.namespace}")'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Database):
            return self._astra_db == other._astra_db
        else:
            return False

    def _copy(
        self,
        *,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> Database:
        return Database(
            api_endpoint=api_endpoint or self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            namespace=namespace or self._astra_db.namespace,
            caller_name=caller_name or self._astra_db.caller_name,
            caller_version=caller_version or self._astra_db.caller_version,
            api_path=api_path or self._astra_db.api_path,
            api_version=api_version or self._astra_db.api_version,
        )

    def with_options(
        self,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> Database:
        """
        Create a clone of this database with some changed attributes.

        Args:
            namespace: this is the namespace all method calls will target, unless
                one is explicitly specified in the call. If no namespace is supplied
                when creating a Database, the name "default_namespace" is set.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new `Database` instance.

        Example:
            >>> my_db_2 = my_db.with_options(
            ...     namespace="the_other_namespace",
            ...     caller_name="the_caller",
            ...     caller_version="0.1.0",
            ... )
        """

        return self._copy(
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def to_async(
        self,
        *,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> AsyncDatabase:
        """
        Create an AsyncDatabase from this one. Save for the arguments
        explicitly provided as overrides, everything else is kept identical
        to this database in the copy.

        Args:
            api_endpoint: the full "API Endpoint" string used to reach the Data API.
                Example: "https://<database_id>-<region>.apps.astra.datastax.com"
            token: an Access Token to the database. Example: "AstraCS:xyz..."
            namespace: this is the namespace all method calls will target, unless
                one is explicitly specified in the call. If no namespace is supplied
                when creating a Database, the name "default_namespace" is set.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".

        Returns:
            the new copy, an `AsyncDatabase` instance.

        Example:
            >>> my_async_db = my_db.to_async()
            >>> asyncio.run(my_async_db.list_collection_names())
        """

        return AsyncDatabase(
            api_endpoint=api_endpoint or self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            namespace=namespace or self._astra_db.namespace,
            caller_name=caller_name or self._astra_db.caller_name,
            caller_version=caller_version or self._astra_db.caller_version,
            api_path=api_path or self._astra_db.api_path,
            api_version=api_version or self._astra_db.api_version,
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
            >>> my_db.set_caller(caller_name="the_caller", caller_version="0.1.0")
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._astra_db.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def info(self) -> DatabaseInfo:
        """
        Additional information on the database as a DatabaseInfo instance.

        Some of the returned properties are dynamic throughout the lifetime
        of the database (such as raw_info["keyspaces"]). For this reason,
        each invocation of this method triggers a new request to the DevOps API.

        Example:
            >>> my_db.info().region
            'eu-west-1'

            >>> my_db.info().raw_info['datacenters'][0]['dateCreated']
            '2023-01-30T12:34:56Z'

        Note:
            see the DatabaseInfo documentation for a caveat about the difference
            between the `region` and the `raw_info["region"]` attributes.
        """

        logger.info("getting database info")
        database_info = fetch_database_info(
            self._astra_db.api_endpoint,
            token=self._astra_db.token,
            namespace=self.namespace,
        )
        if database_info is not None:
            logger.info("finished getting database info")
            return database_info
        else:
            raise DevOpsAPIException(
                "Database is not in a supported environment for this operation."
            )

    @property
    def id(self) -> str:
        """
        The ID of this database.

        Example:
            >>> my_db.id
            '01234567-89ab-cdef-0123-456789abcdef'
        """

        parsed_api_endpoint = parse_api_endpoint(self._astra_db.api_endpoint)
        if parsed_api_endpoint is not None:
            return parsed_api_endpoint.database_id
        else:
            raise DevOpsAPIException(
                "Database is not in a supported environment for this operation."
            )

    def name(self) -> str:
        """
        The name of this database. Note that this bears no unicity guarantees.

        Calling this method the first time involves a request
        to the DevOps API (the resulting database name is then cached).
        See the `info()` method for more details.

        Example:
            >>> my_db.name()
            'the_application_database'
        """

        if self._name is None:
            self._name = self.info().name
        return self._name

    @property
    def namespace(self) -> str:
        """
        The namespace this database uses as target for all commands when
        no method-call-specific namespace is specified.

        Example:
            >>> my_db.namespace
            'the_keyspace'
        """

        return self._astra_db.namespace

    def get_collection(
        self, name: str, *, namespace: Optional[str] = None
    ) -> Collection:
        """
        Spawn a `Collection` object instance representing a collection
        on this database.

        Creating a `Collection` instance does not have any effect on the
        actual state of the database: in other words, for the created
        `Collection` instance to be used meaningfully, the collection
        must exist already (for instance, it should have been created
        previously by calling the `create_collection` method).

        Args:
            name: the name of the collection.
            namespace: the namespace containing the collection. If no namespace
                is specified, the general setting for this database is used.

        Returns:
            a `Collection` instance, representing the desired collection
                (but without any form of validation).

        Example:
            >>> my_col = my_db.get_collection("my_collection")
            >>> my_col.count_documents({}, upper_bound=100)
            41

        Note:
            The attribute and indexing syntax forms achieve the same effect
            as this method. In other words, the following are equivalent:
                my_db.get_collection("coll_name")
                my_db.coll_name
                my_db["coll_name"]
        """

        # lazy importing here against circular-import error
        from astrapy.collection import Collection

        _namespace = namespace or self._astra_db.namespace
        return Collection(self, name, namespace=_namespace)

    @recast_method_sync
    def create_collection(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        service: Optional[Union[CollectionVectorServiceOptions, Dict[str, Any]]] = None,
        indexing: Optional[Dict[str, Any]] = None,
        default_id_type: Optional[str] = None,
        additional_options: Optional[Dict[str, Any]] = None,
        check_exists: Optional[bool] = None,
        max_time_ms: Optional[int] = None,
    ) -> Collection:
        """
        Creates a collection on the database and return the Collection
        instance that represents it.

        This is a blocking operation: the method returns when the collection
        is ready to be used. As opposed to the `get_collection` instance,
        this method triggers causes the collection to be actually created on DB.

        Args:
            name: the name of the collection.
            namespace: the namespace where the collection is to be created.
                If not specified, the general setting for this database is used.
            dimension: for vector collections, the dimension of the vectors
                (i.e. the number of their components).
            metric: the similarity metric used for vector searches.
                Allowed values are `VectorMetric.DOT_PRODUCT`, `VectorMetric.EUCLIDEAN`
                or `VectorMetric.COSINE` (default).
            service: a dictionary describing a service for
                embedding computation, e.g. `{"provider": "ab", "modelName": "xy"}`.
                Alternatively, a CollectionVectorServiceOptions object to the same effect.
                NOTE: This feature is under current development.
            indexing: optional specification of the indexing options for
                the collection, in the form of a dictionary such as
                    {"deny": [...]}
                or
                    {"allow": [...]}
            default_id_type: this sets what type of IDs the API server will
                generate when inserting documents that do not specify their
                `_id` field explicitly. Can be set to any of the values
                `DefaultIdType.UUID`, `DefaultIdType.OBJECTID`,
                `DefaultIdType.UUIDV6`, `DefaultIdType.UUIDV7`,
                `DefaultIdType.DEFAULT`.
            additional_options: any further set of key-value pairs that will
                be added to the "options" part of the payload when sending
                the Data API command to create a collection.
            check_exists: whether to run an existence check for the collection
                name before attempting to create the collection:
                If check_exists is True, an error is raised when creating
                an existing collection.
                If it is False, the creation is attempted. In this case, for
                preexisting collections, the command will succeed or fail
                depending on whether the options match or not.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a (synchronous) `Collection` instance, representing the
            newly-created collection.

        Example:
            >>> new_col = my_db.create_collection("my_v_col", dimension=3)
            >>> new_col.insert_one({"name": "the_row"}, vector=[0.4, 0.5, 0.7])
            InsertOneResult(raw_results=..., inserted_id='e22dd65e-...-...-...')

        Note:
            A collection is considered a vector collection if at least one of
            `dimension` or `service` are provided and not null. In that case,
            and only in that case, is `metric` an accepted parameter.
            Note, moreover, that if passing both these parameters, then
            the dimension must be compatible with the chosen service.
        """

        _validate_create_collection_options(
            dimension=dimension,
            metric=metric,
            service=service,
            indexing=indexing,
            default_id_type=default_id_type,
            additional_options=additional_options,
        )
        _options = {
            **(additional_options or {}),
            **({"indexing": indexing} if indexing else {}),
            **({"defaultId": {"type": default_id_type}} if default_id_type else {}),
        }

        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)

        if check_exists is None:
            _check_exists = True
        else:
            _check_exists = check_exists
        existing_names: List[str]
        if _check_exists:
            logger.info(f"checking collection existence for '{name}'")
            existing_names = self.list_collection_names(
                namespace=namespace,
                max_time_ms=timeout_manager.remaining_timeout_ms(),
            )
        else:
            existing_names = []

        driver_db = self._astra_db.copy(namespace=namespace)
        if name in existing_names:
            raise CollectionAlreadyExistsException(
                text=f"CollectionInvalid: collection {name} already exists",
                namespace=driver_db.namespace,
                collection_name=name,
            )

        service_dict: Optional[Dict[str, Any]]
        if service is not None:
            service_dict = service if isinstance(service, dict) else service.as_dict()
        else:
            service_dict = None

        logger.info(f"creating collection '{name}'")
        driver_db.create_collection(
            name,
            options=_options,
            dimension=dimension,
            metric=metric,
            service_dict=service_dict,
            timeout_info=timeout_manager.remaining_timeout_info(),
        )
        logger.info(f"finished creating collection '{name}'")
        return self.get_collection(name, namespace=namespace)

    @recast_method_sync
    def drop_collection(
        self,
        name_or_collection: Union[str, Collection],
        *,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop a collection from the database, along with all documents therein.

        Args:
            name_or_collection: either the name of a collection or
                a `Collection` instance.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary in the form {"ok": 1} if the command succeeds.

        Example:
            >>> my_db.list_collection_names()
            ['a_collection', 'my_v_col', 'another_col']
            >>> my_db.drop_collection("my_v_col")
            {'ok': 1}
            >>> my_db.list_collection_names()
            ['a_collection', 'another_col']

        Note:
            when providing a collection name, it is assumed that the collection
            is to be found in the namespace set at database instance level.
        """

        # lazy importing here against circular-import error
        from astrapy.collection import Collection

        if isinstance(name_or_collection, Collection):
            _namespace = name_or_collection.namespace
            _name: str = name_or_collection.name
            logger.info(f"dropping collection '{_name}'")
            dc_response = self._astra_db.copy(namespace=_namespace).delete_collection(
                _name,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(f"finished dropping collection '{_name}'")
            return dc_response.get("status", {})  # type: ignore[no-any-return]
        else:
            logger.info(f"dropping collection '{name_or_collection}'")
            dc_response = self._astra_db.delete_collection(
                name_or_collection,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(f"finished dropping collection '{name_or_collection}'")
            return dc_response.get("status", {})  # type: ignore[no-any-return]

    @recast_method_sync
    def list_collections(
        self,
        *,
        namespace: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> CommandCursor[CollectionDescriptor]:
        """
        List all collections in a given namespace for this database.

        Args:
            namespace: the namespace to be inspected. If not specified,
                the general setting for this database is assumed.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a `CommandCursor` to iterate over CollectionDescriptor instances,
            each corresponding to a collection.

        Example:
            >>> ccur = my_db.list_collections()
            >>> ccur
            <astrapy.cursors.CommandCursor object at ...>
            >>> list(ccur)
            [CollectionDescriptor(name='my_v_col', options=CollectionOptions())]
            >>> for coll_dict in my_db.list_collections():
            ...     print(coll_dict)
            ...
            CollectionDescriptor(name='my_v_col', options=CollectionOptions())
        """

        if namespace:
            _client = self._astra_db.copy(namespace=namespace)
        else:
            _client = self._astra_db
        logger.info("getting collections")
        gc_response = _client.get_collections(
            options={"explain": True}, timeout_info=base_timeout_info(max_time_ms)
        )
        if "collections" not in gc_response.get("status", {}):
            raise DataAPIFaultyResponseException(
                text="Faulty response from get_collections API command.",
                raw_response=gc_response,
            )
        else:
            # we know this is a list of dicts, to marshal into "descriptors"
            logger.info("finished getting collections")
            return CommandCursor(
                address=self._astra_db.base_url,
                items=[
                    CollectionDescriptor.from_dict(col_dict)
                    for col_dict in gc_response["status"]["collections"]
                ],
            )

    @recast_method_sync
    def list_collection_names(
        self,
        *,
        namespace: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> List[str]:
        """
        List the names of all collections in a given namespace of this database.

        Args:
            namespace: the namespace to be inspected. If not specified,
                the general setting for this database is assumed.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a list of the collection names as strings, in no particular order.

        Example:
            >>> my_db.list_collection_names()
            ['a_collection', 'another_col']
        """

        logger.info("getting collection names")
        gc_response = self._astra_db.copy(namespace=namespace).get_collections(
            timeout_info=base_timeout_info(max_time_ms)
        )
        if "collections" not in gc_response.get("status", {}):
            raise DataAPIFaultyResponseException(
                text="Faulty response from get_collections API command.",
                raw_response=gc_response,
            )
        else:
            # we know this is a list of strings
            logger.info("finished getting collection names")
            return gc_response["status"]["collections"]  # type: ignore[no-any-return]

    @recast_method_sync
    def command(
        self,
        body: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        collection_name: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a POST request to the Data API for this database with
        an arbitrary, caller-provided payload.

        Args:
            body: a JSON-serializable dictionary, the payload of the request.
            namespace: the namespace to use. Requests always target a namespace:
                if not specified, the general setting for this database is assumed.
            collection_name: if provided, the collection name is appended at the end
                of the endpoint. In this way, this method allows collection-level
                arbitrary POST requests as well.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary with the response of the HTTP request.

        Example:
            >>> my_db.command({"findCollections": {}})
            {'status': {'collections': ['my_coll']}}
            >>> my_db.command({"countDocuments": {}}, collection_name="my_coll")
            {'status': {'count': 123}}
        """

        if namespace:
            _client = self._astra_db.copy(namespace=namespace)
        else:
            _client = self._astra_db
        if collection_name:
            _collection = _client.collection(collection_name)
            logger.info(f"issuing custom command to API (on '{collection_name}')")
            req_response = _collection.post_raw_request(
                body=body,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(
                f"finished issuing custom command to API (on '{collection_name}')"
            )
            return req_response
        else:
            logger.info("issuing custom command to API")
            req_response = _client.post_raw_request(
                body=body,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info("finished issuing custom command to API")
            return req_response

    def get_database_admin(
        self,
        *,
        token: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Return an AstraDBDatabaseAdmin object corresponding to this database, for
        use in admin tasks such as managing namespaces.

        Args:
            token: an access token with enough permission on the database to
                perform the desired tasks. If omitted (as it can generally be done),
                the token of this Database is used.
            dev_ops_url: in case of custom deployments, this can be used to specify
                the URL to the DevOps API, such as "https://api.astra.datastax.com".
                Generally it can be omitted. The environment (prod/dev/...) is
                determined from the API Endpoint.
            dev_ops_api_version: this can specify a custom version of the DevOps API
                (such as "v2"). Generally not needed.

        Returns:
            An AstraDBDatabaseAdmin instance targeting this database.

        Example:
            >>> my_db_admin = my_db.get_database_admin()
            >>> if "new_namespace" not in my_db_admin.list_namespaces():
            ...     my_db_admin.create_namespace("new_namespace")
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'new_namespace']
        """

        # lazy importing here to avoid circular dependency
        from astrapy.admin import AstraDBDatabaseAdmin

        return AstraDBDatabaseAdmin.from_api_endpoint(
            api_endpoint=self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            caller_name=self._astra_db.caller_name,
            caller_version=self._astra_db.caller_version,
            dev_ops_url=dev_ops_url,
            dev_ops_api_version=dev_ops_api_version,
        )


class AsyncDatabase:
    """
    A Data API database. This is the entry-point object for doing database-level
    DML, such as creating/deleting collections, and for obtaining Collection
    objects themselves. This class has an asynchronous interface.

    A Database comes with an "API Endpoint", which implies a Database object
    instance reaches a specific region (relevant point in case of multi-region
    databases).

    Args:
        api_endpoint: the full "API Endpoint" string used to reach the Data API.
            Example: "https://<database_id>-<region>.apps.astra.datastax.com"
        token: an Access Token to the database. Example: "AstraCS:xyz..."
        namespace: this is the namespace all method calls will target, unless
            one is explicitly specified in the call. If no namespace is supplied
            when creating a Database, the name "default_namespace" is set.
        caller_name: name of the application, or framework, on behalf of which
            the Data API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.
        api_path: path to append to the API Endpoint. In typical usage, this
            should be left to its default of "/api/json".
        api_version: version specifier to append to the API path. In typical
            usage, this should be left to its default of "v1".

    Example:
        >>> from astrapy import DataAPIClient
        >>> my_client = astrapy.DataAPIClient("AstraCS:...")
        >>> my_db = my_client.get_async_database_by_api_endpoint(
        ...    "https://01234567-....apps.astra.datastax.com"
        ... )

    Note:
        creating an instance of AsyncDatabase does not trigger actual creation
        of the database itself, which should exist beforehand. To create databases,
        see the AstraDBAdmin class.
    """

    def __init__(
        self,
        api_endpoint: str,
        token: str,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        self._astra_db = AsyncAstraDB(
            token=token,
            api_endpoint=api_endpoint,
            api_path=api_path,
            api_version=api_version,
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )
        self._name: Optional[str] = None

    def __getattr__(self, collection_name: str) -> AsyncCollection:
        return self.to_sync().get_collection(name=collection_name).to_async()

    def __getitem__(self, collection_name: str) -> AsyncCollection:
        return self.to_sync().get_collection(name=collection_name).to_async()

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(api_endpoint="{self._astra_db.api_endpoint}", '
            f'token="{self._astra_db.token[:12]}...", namespace="{self._astra_db.namespace}")'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AsyncDatabase):
            return self._astra_db == other._astra_db
        else:
            return False

    async def __aenter__(self) -> AsyncDatabase:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        await self._astra_db.__aexit__(
            exc_type=exc_type,
            exc_value=exc_value,
            traceback=traceback,
        )

    def _copy(
        self,
        *,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> AsyncDatabase:
        return AsyncDatabase(
            api_endpoint=api_endpoint or self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            namespace=namespace or self._astra_db.namespace,
            caller_name=caller_name or self._astra_db.caller_name,
            caller_version=caller_version or self._astra_db.caller_version,
            api_path=api_path or self._astra_db.api_path,
            api_version=api_version or self._astra_db.api_version,
        )

    def with_options(
        self,
        *,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AsyncDatabase:
        """
        Create a clone of this database with some changed attributes.

        Args:
            namespace: this is the namespace all method calls will target, unless
                one is explicitly specified in the call. If no namespace is supplied
                when creating a Database, the name "default_namespace" is set.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new `AsyncDatabase` instance.

        Example:
            >>> my_async_db_2 = my_async_db.with_options(
            ...     namespace="the_other_namespace",
            ...     caller_name="the_caller",
            ...     caller_version="0.1.0",
            ... )
        """

        return self._copy(
            namespace=namespace,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def to_sync(
        self,
        *,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> Database:
        """
        Create a (synchronous) Database from this one. Save for the arguments
        explicitly provided as overrides, everything else is kept identical
        to this database in the copy.

        Args:
            api_endpoint: the full "API Endpoint" string used to reach the Data API.
                Example: "https://<database_id>-<region>.apps.astra.datastax.com"
            token: an Access Token to the database. Example: "AstraCS:xyz..."
            namespace: this is the namespace all method calls will target, unless
                one is explicitly specified in the call. If no namespace is supplied
                when creating a Database, the name "default_namespace" is set.
            caller_name: name of the application, or framework, on behalf of which
                the Data API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".

        Returns:
            the new copy, a `Database` instance.

        Example:
            >>> my_sync_db = my_async_db.to_sync()
            >>> my_sync_db.list_collection_names()
            ['a_collection', 'another_collection']
        """

        return Database(
            api_endpoint=api_endpoint or self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            namespace=namespace or self._astra_db.namespace,
            caller_name=caller_name or self._astra_db.caller_name,
            caller_version=caller_version or self._astra_db.caller_version,
            api_path=api_path or self._astra_db.api_path,
            api_version=api_version or self._astra_db.api_version,
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
            >>> my_db.set_caller(caller_name="the_caller", caller_version="0.1.0")
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._astra_db.set_caller(
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def info(self) -> DatabaseInfo:
        """
        Additional information on the database as a DatabaseInfo instance.

        Some of the returned properties are dynamic throughout the lifetime
        of the database (such as raw_info["keyspaces"]). For this reason,
        each invocation of this method triggers a new request to the DevOps API.

        Example:
            >>> my_async_db.info().region
            'eu-west-1'

            >>> my_async_db.info().raw_info['datacenters'][0]['dateCreated']
            '2023-01-30T12:34:56Z'

        Note:
            see the DatabaseInfo documentation for a caveat about the difference
            between the `region` and the `raw_info["region"]` attributes.
        """

        logger.info("getting database info")
        database_info = fetch_database_info(
            self._astra_db.api_endpoint,
            token=self._astra_db.token,
            namespace=self.namespace,
        )
        if database_info is not None:
            logger.info("finished getting database info")
            return database_info
        else:
            raise DevOpsAPIException(
                "Database is not in a supported environment for this operation."
            )

    @property
    def id(self) -> str:
        """
        The ID of this database.

        Example:
            >>> my_async_db.id
            '01234567-89ab-cdef-0123-456789abcdef'
        """

        parsed_api_endpoint = parse_api_endpoint(self._astra_db.api_endpoint)
        if parsed_api_endpoint is not None:
            return parsed_api_endpoint.database_id
        else:
            raise DevOpsAPIException(
                "Database is not in a supported environment for this operation."
            )

    def name(self) -> str:
        """
        The name of this database. Note that this bears no unicity guarantees.

        Calling this method the first time involves a request
        to the DevOps API (the resulting database name is then cached).
        See the `info()` method for more details.

        Example:
            >>> my_async_db.name()
            'the_application_database'
        """

        if self._name is None:
            self._name = self.info().name
        return self._name

    @property
    def namespace(self) -> str:
        """
        The namespace this database uses as target for all commands when
        no method-call-specific namespace is specified.

        Example:
            >>> my_async_db.namespace
            'the_keyspace'
        """

        return self._astra_db.namespace

    async def get_collection(
        self, name: str, *, namespace: Optional[str] = None
    ) -> AsyncCollection:
        """
        Spawn an `AsyncCollection` object instance representing a collection
        on this database.

        Creating an `AsyncCollection` instance does not have any effect on the
        actual state of the database: in other words, for the created
        `AsyncCollection` instance to be used meaningfully, the collection
        must exist already (for instance, it should have been created
        previously by calling the `create_collection` method).

        Args:
            name: the name of the collection.
            namespace: the namespace containing the collection. If no namespace
                is specified, the setting for this database is used.

        Returns:
            an `AsyncCollection` instance, representing the desired collection
                (but without any form of validation).

        Example:
            >>> async def count_docs(adb: AsyncDatabase, c_name: str) -> int:
            ...    async_col = await adb.get_collection(c_name)
            ...    return await async_col.count_documents({}, upper_bound=100)
            ...
            >>> asyncio.run(count_docs(my_async_db, "my_collection"))
            45

        Note: the attribute and indexing syntax forms achieve the same effect
            as this method, returning an AsyncCollection, albeit
            in a synchronous way. In other words, the following are equivalent:
                await my_async_db.get_collection("coll_name")
                my_async_db.coll_name
                my_async_db["coll_name"]
        """

        # lazy importing here against circular-import error
        from astrapy.collection import AsyncCollection

        _namespace = namespace or self._astra_db.namespace
        return AsyncCollection(self, name, namespace=_namespace)

    @recast_method_async
    async def create_collection(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        service: Optional[Union[CollectionVectorServiceOptions, Dict[str, Any]]] = None,
        indexing: Optional[Dict[str, Any]] = None,
        default_id_type: Optional[str] = None,
        additional_options: Optional[Dict[str, Any]] = None,
        check_exists: Optional[bool] = None,
        max_time_ms: Optional[int] = None,
    ) -> AsyncCollection:
        """
        Creates a collection on the database and return the AsyncCollection
        instance that represents it.

        This is a blocking operation: the method returns when the collection
        is ready to be used. As opposed to the `get_collection` instance,
        this method triggers causes the collection to be actually created on DB.

        Args:
            name: the name of the collection.
            namespace: the namespace where the collection is to be created.
                If not specified, the general setting for this database is used.
            dimension: for vector collections, the dimension of the vectors
                (i.e. the number of their components).
            metric: the similarity metric used for vector searches.
                Allowed values are `VectorMetric.DOT_PRODUCT`, `VectorMetric.EUCLIDEAN`
                or `VectorMetric.COSINE` (default).
            service: a dictionary describing a service for
                embedding computation, e.g. `{"provider": "ab", "modelName": "xy"}`.
                Alternatively, a CollectionVectorServiceOptions object to the same effect.
                NOTE: This feature is under current development.
            indexing: optional specification of the indexing options for
                the collection, in the form of a dictionary such as
                    {"deny": [...]}
                or
                    {"allow": [...]}
            default_id_type: this sets what type of IDs the API server will
                generate when inserting documents that do not specify their
                `_id` field explicitly. Can be set to any of the values
                `DefaultIdType.UUID`, `DefaultIdType.OBJECTID`,
                `DefaultIdType.UUIDV6`, `DefaultIdType.UUIDV7`,
                `DefaultIdType.DEFAULT`.
            additional_options: any further set of key-value pairs that will
                be added to the "options" part of the payload when sending
                the Data API command to create a collection.
            check_exists: whether to run an existence check for the collection
                name before attempting to create the collection:
                If check_exists is True, an error is raised when creating
                an existing collection.
                If it is False, the creation is attempted. In this case, for
                preexisting collections, the command will succeed or fail
                depending on whether the options match or not.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an `AsyncCollection` instance, representing the newly-created collection.

        Example:
            >>> async def create_and_insert(adb: AsyncDatabase) -> Dict[str, Any]:
            ...     new_a_col = await adb.create_collection("my_v_col", dimension=3)
            ...     return await new_a_col.insert_one(
            ...         {"name": "the_row"},
            ...         vector=[0.4, 0.5, 0.7],
            ...     )
            ...
            >>> asyncio.run(create_and_insert(my_async_db))
            InsertOneResult(raw_results=..., inserted_id='08f05ecf-...-...-...')

        Note:
            A collection is considered a vector collection if at least one of
            `dimension` or `service` are provided and not null. In that case,
            and only in that case, is `metric` an accepted parameter.
            Note, moreover, that if passing both these parameters, then
            the dimension must be compatible with the chosen service.
        """

        _validate_create_collection_options(
            dimension=dimension,
            metric=metric,
            service=service,
            indexing=indexing,
            default_id_type=default_id_type,
            additional_options=additional_options,
        )
        _options = {
            **(additional_options or {}),
            **({"indexing": indexing} if indexing else {}),
            **({"defaultId": {"type": default_id_type}} if default_id_type else {}),
        }

        timeout_manager = MultiCallTimeoutManager(overall_max_time_ms=max_time_ms)

        if check_exists is None:
            _check_exists = True
        else:
            _check_exists = check_exists
        existing_names: List[str]
        if _check_exists:
            logger.info(f"checking collection existence for '{name}'")
            existing_names = await self.list_collection_names(
                namespace=namespace,
                max_time_ms=timeout_manager.remaining_timeout_ms(),
            )
        else:
            existing_names = []
        driver_db = self._astra_db.copy(namespace=namespace)
        if name in existing_names:
            raise CollectionAlreadyExistsException(
                text=f"CollectionInvalid: collection {name} already exists",
                namespace=driver_db.namespace,
                collection_name=name,
            )

        service_dict: Optional[Dict[str, Any]]
        if service is not None:
            service_dict = service if isinstance(service, dict) else service.as_dict()
        else:
            service_dict = None

        logger.info(f"creating collection '{name}'")
        await driver_db.create_collection(
            name,
            options=_options,
            dimension=dimension,
            metric=metric,
            service_dict=service_dict,
            timeout_info=timeout_manager.remaining_timeout_info(),
        )
        logger.info(f"finished creating collection '{name}'")
        return await self.get_collection(name, namespace=namespace)

    @recast_method_async
    async def drop_collection(
        self,
        name_or_collection: Union[str, AsyncCollection],
        *,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop a collection from the database, along with all documents therein.

        Args:
            name_or_collection: either the name of a collection or
                an `AsyncCollection` instance.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary in the form {"ok": 1} if the command succeeds.

        Example:
            >>> asyncio.run(my_async_db.list_collection_names())
            ['a_collection', 'my_v_col', 'another_col']
            >>> asyncio.run(my_async_db.drop_collection("my_v_col"))
            {'ok': 1}
            >>> asyncio.run(my_async_db.list_collection_names())
            ['a_collection', 'another_col']

        Note:
            when providing a collection name, it is assumed that the collection
            is to be found in the namespace set at database instance level.
        """

        # lazy importing here against circular-import error
        from astrapy.collection import AsyncCollection

        if isinstance(name_or_collection, AsyncCollection):
            _namespace = name_or_collection.namespace
            _name = name_or_collection.name
            logger.info(f"dropping collection '{_name}'")
            dc_response = await self._astra_db.copy(
                namespace=_namespace
            ).delete_collection(
                _name,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(f"finished dropping collection '{_name}'")
            return dc_response.get("status", {})  # type: ignore[no-any-return]
        else:
            logger.info(f"dropping collection '{name_or_collection}'")
            dc_response = await self._astra_db.delete_collection(
                name_or_collection,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(f"finished dropping collection '{name_or_collection}'")
            return dc_response.get("status", {})  # type: ignore[no-any-return]

    @recast_method_sync
    def list_collections(
        self,
        *,
        namespace: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> AsyncCommandCursor[CollectionDescriptor]:
        """
        List all collections in a given namespace for this database.

        Args:
            namespace: the namespace to be inspected. If not specified,
                the general setting for this database is assumed.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            an `AsyncCommandCursor` to iterate over CollectionDescriptor instances,
            each corresponding to a collection.

        Example:
            >>> async def a_list_colls(adb: AsyncDatabase) -> None:
            ...     a_ccur = adb.list_collections()
            ...     print("* a_ccur:", a_ccur)
            ...     print("* list:", [coll async for coll in a_ccur])
            ...     async for coll in adb.list_collections():
            ...         print("* coll:", coll)
            ...
            >>> asyncio.run(a_list_colls(my_async_db))
            * a_ccur: <astrapy.cursors.AsyncCommandCursor object at ...>
            * list: [CollectionDescriptor(name='my_v_col', options=CollectionOptions())]
            * coll: CollectionDescriptor(name='my_v_col', options=CollectionOptions())
        """

        _client: AsyncAstraDB
        if namespace:
            _client = self._astra_db.copy(namespace=namespace)
        else:
            _client = self._astra_db
        logger.info("getting collections")
        gc_response = _client.to_sync().get_collections(
            options={"explain": True},
            timeout_info=base_timeout_info(max_time_ms),
        )
        if "collections" not in gc_response.get("status", {}):
            raise DataAPIFaultyResponseException(
                text="Faulty response from get_collections API command.",
                raw_response=gc_response,
            )
        else:
            # we know this is a list of dicts, to marshal into "descriptors"
            logger.info("finished getting collections")
            return AsyncCommandCursor(
                address=self._astra_db.base_url,
                items=[
                    CollectionDescriptor.from_dict(col_dict)
                    for col_dict in gc_response["status"]["collections"]
                ],
            )

    @recast_method_async
    async def list_collection_names(
        self,
        *,
        namespace: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> List[str]:
        """
        List the names of all collections in a given namespace of this database.

        Args:
            namespace: the namespace to be inspected. If not specified,
                the general setting for this database is assumed.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a list of the collection names as strings, in no particular order.

        Example:
            >>> asyncio.run(my_async_db.list_collection_names())
            ['a_collection', 'another_col']
        """

        logger.info("getting collection names")
        gc_response = await self._astra_db.copy(namespace=namespace).get_collections(
            timeout_info=base_timeout_info(max_time_ms)
        )
        if "collections" not in gc_response.get("status", {}):
            raise DataAPIFaultyResponseException(
                text="Faulty response from get_collections API command.",
                raw_response=gc_response,
            )
        else:
            # we know this is a list of strings
            logger.info("finished getting collection names")
            return gc_response["status"]["collections"]  # type: ignore[no-any-return]

    @recast_method_async
    async def command(
        self,
        body: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        collection_name: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a POST request to the Data API for this database with
        an arbitrary, caller-provided payload.

        Args:
            body: a JSON-serializable dictionary, the payload of the request.
            namespace: the namespace to use. Requests always target a namespace:
                if not specified, the general setting for this database is assumed.
            collection_name: if provided, the collection name is appended at the end
                of the endpoint. In this way, this method allows collection-level
                arbitrary POST requests as well.
            max_time_ms: a timeout, in milliseconds, for the underlying HTTP request.

        Returns:
            a dictionary with the response of the HTTP request.

        Example:
            >>> asyncio.run(my_async_db.command({"findCollections": {}}))
            {'status': {'collections': ['my_coll']}}
            >>> asyncio.run(my_async_db.command(
            ...     {"countDocuments": {}},
            ...     collection_name="my_coll",
            ... )
            {'status': {'count': 123}}
        """

        if namespace:
            _client = self._astra_db.copy(namespace=namespace)
        else:
            _client = self._astra_db
        if collection_name:
            _collection = await _client.collection(collection_name)
            logger.info(f"issuing custom command to API (on '{collection_name}')")
            req_response = await _collection.post_raw_request(
                body=body,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info(
                f"finished issuing custom command to API (on '{collection_name}')"
            )
            return req_response
        else:
            logger.info("issuing custom command to API")
            req_response = await _client.post_raw_request(
                body=body,
                timeout_info=base_timeout_info(max_time_ms),
            )
            logger.info("finished issuing custom command to API")
            return req_response

    def get_database_admin(
        self,
        *,
        token: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Return an AstraDBDatabaseAdmin object corresponding to this database, for
        use in admin tasks such as managing namespaces.

        Args:
            token: an access token with enough permission on the database to
                perform the desired tasks. If omitted (as it can generally be done),
                the token of this Database is used.
            dev_ops_url: in case of custom deployments, this can be used to specify
                the URL to the DevOps API, such as "https://api.astra.datastax.com".
                Generally it can be omitted. The environment (prod/dev/...) is
                determined from the API Endpoint.
            dev_ops_api_version: this can specify a custom version of the DevOps API
                (such as "v2"). Generally not needed.

        Returns:
            An AstraDBDatabaseAdmin instance targeting this database.

        Example:
            >>> my_db_admin = my_async_db.get_database_admin()
            >>> if "new_namespace" not in my_db_admin.list_namespaces():
            ...     my_db_admin.create_namespace("new_namespace")
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'new_namespace']
        """

        # lazy importing here to avoid circular dependency
        from astrapy.admin import AstraDBDatabaseAdmin

        return AstraDBDatabaseAdmin.from_api_endpoint(
            api_endpoint=self._astra_db.api_endpoint,
            token=token or self._astra_db.token,
            caller_name=self._astra_db.caller_name,
            caller_version=self._astra_db.caller_version,
            dev_ops_url=dev_ops_url,
            dev_ops_api_version=dev_ops_api_version,
        )
