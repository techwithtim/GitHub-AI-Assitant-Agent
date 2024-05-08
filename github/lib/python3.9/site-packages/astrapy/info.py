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

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DatabaseInfo:
    """
    Represents the identifying information for a database,
    including the region the connection is established to.

    Attributes:
        id: the database ID.
        region: the ID of the region through which the connection to DB is done.
        namespace: the namespace this DB is set to work with.
        name: the database name. Not necessarily unique: there can be multiple
            databases with the same name.
        environment: a label, whose value is one of Environment.PROD, Environment.DEV
            or Environment.TEST.
        raw_info: the full response from the DevOPS API call to get this info.

    Note:
        The `raw_info` dictionary usually has a `region` key describing
        the default region as configured in the database, which does not
        necessarily (for multi-region databases) match the region through
        which the connection is established: the latter is the one specified
        by the "api endpoint" used for connecting. In other words, for multi-region
        databases it is possible that
            database_info.region != database_info.raw_info["region"]
        Conversely, in case of a DatabaseInfo not obtained through a
        connected database, such as when calling `Admin.list_databases()`,
        all fields except `environment` (e.g. namespace, region, etc)
        are set as found on the DevOps API response directly.
    """

    id: str
    region: str
    namespace: str
    name: str
    environment: str
    raw_info: Optional[Dict[str, Any]]


@dataclass
class AdminDatabaseInfo:
    """
    Represents the full response from the DevOps API about a database info.

    Most attributes just contain the corresponding part of the raw response:
    for this reason, please consult the DevOps API documentation for details.

    Attributes:
        info: a DatabaseInfo instance for the underlying database.
            The DatabaseInfo is a subset of the information described by
            AdminDatabaseInfo - in terms of the DevOps API response,
            it corresponds to just its "info" subdictionary.
        available_actions: the "availableActions" value in the full API response.
        cost: the "cost" value in the full API response.
        cqlsh_url: the "cqlshUrl" value in the full API response.
        creation_time: the "creationTime" value in the full API response.
        data_endpoint_url: the "dataEndpointUrl" value in the full API response.
        grafana_url: the "grafanaUrl" value in the full API response.
        graphql_url: the "graphqlUrl" value in the full API response.
        id: the "id" value in the full API response.
        last_usage_time: the "lastUsageTime" value in the full API response.
        metrics: the "metrics" value in the full API response.
        observed_status: the "observedStatus" value in the full API response.
        org_id: the "orgId" value in the full API response.
        owner_id: the "ownerId" value in the full API response.
        status: the "status" value in the full API response.
        storage: the "storage" value in the full API response.
        termination_time: the "terminationTime" value in the full API response.
        raw_info: the full raw response from the DevOps API.
    """

    info: DatabaseInfo
    available_actions: Optional[List[str]]
    cost: Dict[str, Any]
    cqlsh_url: str
    creation_time: str
    data_endpoint_url: str
    grafana_url: str
    graphql_url: str
    id: str
    last_usage_time: str
    metrics: Dict[str, Any]
    observed_status: str
    org_id: str
    owner_id: str
    status: str
    storage: Dict[str, Any]
    termination_time: str
    raw_info: Optional[Dict[str, Any]]


@dataclass
class CollectionInfo:
    """
    Represents the identifying information for a collection,
    including the information about the database the collection belongs to.

    Attributes:
        database_info: a DatabaseInfo instance for the underlying database.
        namespace: the namespace where the collection is located.
        name: collection name. Unique within a namespace.
        full_name: identifier for the collection within the database,
            in the form "namespace.collection_name".
    """

    database_info: DatabaseInfo
    namespace: str
    name: str
    full_name: str


@dataclass
class CollectionDefaultIDOptions:
    """
    The "defaultId" component of the collection options.
    See the Data API specifications for allowed values.

    Attributes:
        default_id_type: string such as `objectId`, `uuid6` and so on.
    """

    default_id_type: str

    def as_dict(self) -> Dict[str, Any]:
        """Recast this object into a dictionary."""

        return {"type": self.default_id_type}

    @staticmethod
    def from_dict(
        raw_dict: Optional[Dict[str, Any]]
    ) -> Optional[CollectionDefaultIDOptions]:
        """
        Create an instance of CollectionDefaultIDOptions from a dictionary
        such as one from the Data API.
        """

        if raw_dict is not None:
            return CollectionDefaultIDOptions(default_id_type=raw_dict["type"])
        else:
            return None


@dataclass
class CollectionVectorServiceOptions:
    """
    The "vector.service" component of the collection options.
    See the Data API specifications for allowed values.

    NOTE: This feature is under current development.

    Attributes:
        provider: the name of a service provider for embedding calculation.
        model_name: the name of a specific model for use by the service.
    """

    provider: Optional[str]
    model_name: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        """Recast this object into a dictionary."""

        return {
            k: v
            for k, v in {
                "provider": self.provider,
                "modelName": self.model_name,
            }.items()
            if v is not None
        }

    @staticmethod
    def from_dict(
        raw_dict: Optional[Dict[str, Any]]
    ) -> Optional[CollectionVectorServiceOptions]:
        """
        Create an instance of CollectionVectorServiceOptions from a dictionary
        such as one from the Data API.
        """

        if raw_dict is not None:
            return CollectionVectorServiceOptions(
                provider=raw_dict.get("provider"),
                model_name=raw_dict.get("modelName"),
            )
        else:
            return None


@dataclass
class CollectionVectorOptions:
    """
    The "vector" component of the collection options.
    See the Data API specifications for allowed values.

    Attributes:
        dimension: an optional positive integer, the dimensionality of the vector space.
        metric: an optional metric among `VectorMetric.DOT_PRODUCT`,
            `VectorMetric.EUCLIDEAN` and `VectorMetric.COSINE`.
        service: an optional X object in case a service is configured for the collection.
            NOTE: This feature is under current development.
    """

    dimension: Optional[int]
    metric: Optional[str]
    service: Optional[CollectionVectorServiceOptions]

    def as_dict(self) -> Dict[str, Any]:
        """Recast this object into a dictionary."""

        return {
            k: v
            for k, v in {
                "dimension": self.dimension,
                "metric": self.metric,
                "service": None if self.service is None else self.service.as_dict(),
            }.items()
            if v is not None
        }

    @staticmethod
    def from_dict(
        raw_dict: Optional[Dict[str, Any]]
    ) -> Optional[CollectionVectorOptions]:
        """
        Create an instance of CollectionVectorOptions from a dictionary
        such as one from the Data API.
        """

        if raw_dict is not None:
            return CollectionVectorOptions(
                dimension=raw_dict.get("dimension"),
                metric=raw_dict.get("metric"),
                service=CollectionVectorServiceOptions.from_dict(
                    raw_dict.get("service")
                ),
            )
        else:
            return None


@dataclass
class CollectionOptions:
    """
    A structure expressing the options of a collection.
    See the Data API specifications for detailed specification and allowed values.

    Attributes:
        vector: an optional CollectionVectorOptions object.
        indexing: an optional dictionary with the "indexing" collection properties.
        default_id: an optional CollectionDefaultIDOptions object.
        raw_options: the raw response from the Data API for the collection configuration.
    """

    vector: Optional[CollectionVectorOptions]
    indexing: Optional[Dict[str, Any]]
    default_id: Optional[CollectionDefaultIDOptions]
    raw_options: Optional[Dict[str, Any]]

    def __repr__(self) -> str:
        not_null_pieces = [
            pc
            for pc in [
                None if self.vector is None else f"vector={self.vector.__repr__()}",
                (
                    None
                    if self.indexing is None
                    else f"indexing={self.indexing.__repr__()}"
                ),
                (
                    None
                    if self.default_id is None
                    else f"default_id={self.default_id.__repr__()}"
                ),
            ]
            if pc is not None
        ]
        return f"{self.__class__.__name__}({', '.join(not_null_pieces)})"

    def as_dict(self) -> Dict[str, Any]:
        """Recast this object into a dictionary."""

        return {
            k: v
            for k, v in {
                "vector": None if self.vector is None else self.vector.as_dict(),
                "indexing": self.indexing,
                "defaultId": (
                    None if self.default_id is None else self.default_id.as_dict()
                ),
            }.items()
            if v is not None
        }

    def flatten(self) -> Dict[str, Any]:
        """
        Recast this object as a flat key-value pair suitable for
        use as kwargs in a create_collection method call (including recasts).
        """

        _dimension: Optional[int]
        _metric: Optional[str]
        _indexing: Optional[Dict[str, Any]]
        _service: Optional[Dict[str, Any]]
        _default_id_type: Optional[str]
        if self.vector is not None:
            _dimension = self.vector.dimension
            _metric = self.vector.metric
            if self.vector.service is None:
                _service = None
            else:
                _service = self.vector.service.as_dict()
        else:
            _dimension = None
            _metric = None
            _service = None
        _indexing = self.indexing
        if self.default_id is not None:
            _default_id_type = self.default_id.default_id_type
        else:
            _default_id_type = None

        return {
            k: v
            for k, v in {
                "dimension": _dimension,
                "metric": _metric,
                "service": _service,
                "indexing": _indexing,
                "default_id_type": _default_id_type,
            }.items()
            if v is not None
        }

    @staticmethod
    def from_dict(raw_dict: Dict[str, Any]) -> CollectionOptions:
        """
        Create an instance of CollectionOptions from a dictionary
        such as one from the Data API.
        """

        return CollectionOptions(
            vector=CollectionVectorOptions.from_dict(raw_dict.get("vector")),
            indexing=raw_dict.get("indexing"),
            default_id=CollectionDefaultIDOptions.from_dict(raw_dict.get("defaultId")),
            raw_options=raw_dict,
        )


@dataclass
class CollectionDescriptor:
    """
    A structure expressing full description of a collection as the Data API
    returns it, i.e. its name and its `options` sub-structure.

    Attributes:
        name: the name of the collection.
        options: a CollectionOptions instance.
        raw_descriptor: the raw response from the Data API.
    """

    name: str
    options: CollectionOptions
    raw_descriptor: Optional[Dict[str, Any]]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name.__repr__()}, "
            f"options={self.options.__repr__()})"
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Recast this object into a dictionary.
        Empty `options` will not be returned at all.
        """

        return {
            k: v
            for k, v in {
                "name": self.name,
                "options": self.options.as_dict(),
            }.items()
            if v
        }

    def flatten(self) -> Dict[str, Any]:
        """
        Recast this object as a flat key-value pair suitable for
        use as kwargs in a create_collection method call (including recasts).
        """

        return {
            **(self.options.flatten()),
            **{"name": self.name},
        }

    @staticmethod
    def from_dict(raw_dict: Dict[str, Any]) -> CollectionDescriptor:
        """
        Create an instance of CollectionDescriptor from a dictionary
        such as one from the Data API.
        """

        return CollectionDescriptor(
            name=raw_dict["name"],
            options=CollectionOptions.from_dict(raw_dict.get("options") or {}),
            raw_descriptor=raw_dict,
        )
