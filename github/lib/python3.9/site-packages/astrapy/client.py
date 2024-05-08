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
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from astrapy.admin import (
    Environment,
    api_endpoint_parser,
    build_api_endpoint,
    database_id_matcher,
    fetch_raw_database_info_from_id_token,
    parse_api_endpoint,
)


if TYPE_CHECKING:
    from astrapy import AsyncDatabase, Database
    from astrapy.admin import AstraDBAdmin


logger = logging.getLogger(__name__)


class DataAPIClient:
    """
    A client for using the Data API. This is the main entry point and sits
    at the top of the conceptual "client -> database -> collection" hierarchy.

    The client is created by passing a suitable Access Token. Starting from the
    client:
        - databases (Database and AsyncDatabase) are created for working with data
        - AstraDBAdmin objects can be created for admin-level work

    Args:
        token: an Access Token to the database. Example: `"AstraCS:xyz..."`.
        environment: a string representing the target Astra environment.
            It can be left unspecified for the default value of `Environment.PROD`;
            other values are `Environment.DEV` and `Environment.TEST`.
        caller_name: name of the application, or framework, on behalf of which
            the Data API and DevOps API calls are performed. This ends up in
            the request user-agent.
        caller_version: version of the caller.

    Example:
        >>> from astrapy import DataAPIClient
        >>> my_client = DataAPIClient("AstraCS:...")
        >>> my_db0 = my_client.get_database_by_api_endpoint(
        ...     "https://01234567-....apps.astra.datastax.com"
        ... )
        >>> my_coll = my_db0.create_collection("movies", dimension=512)
        >>> my_coll.insert_one({"title": "The Title"}, vector=...)
        >>> my_db1 = my_client.get_database("01234567-...")
        >>> my_db2 = my_client.get_database("01234567-...", region="us-east1")
        >>> my_adm0 = my_client.get_admin()
        >>> my_adm1 = my_client.get_admin(token=more_powerful_token_override)
        >>> database_list = my_adm0.list_databases()
    """

    def __init__(
        self,
        token: str,
        *,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        self.token = token
        self.environment = environment or Environment.PROD
        self._caller_name = caller_name
        self._caller_version = caller_version

    def __repr__(self) -> str:
        env_desc: str
        if self.environment == Environment.PROD:
            env_desc = ""
        else:
            env_desc = f', environment="{self.environment}"'
        return f'{self.__class__.__name__}("{self.token[:12]}..."{env_desc})'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataAPIClient):
            return all(
                [
                    self.token == other.token,
                    self.environment == other.environment,
                    self._caller_name == other._caller_name,
                    self._caller_version == other._caller_version,
                ]
            )
        else:
            return False

    def __getitem__(self, database_id_or_api_endpoint: str) -> Database:
        if re.match(database_id_matcher, database_id_or_api_endpoint):
            return self.get_database(database_id_or_api_endpoint)
        elif re.match(api_endpoint_parser, database_id_or_api_endpoint):
            return self.get_database_by_api_endpoint(database_id_or_api_endpoint)
        else:
            raise ValueError(
                "The provided input does not look like either a database ID "
                f"or an API endpoint ('{database_id_or_api_endpoint}')."
            )

    def _copy(
        self,
        *,
        token: Optional[str] = None,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> DataAPIClient:
        return DataAPIClient(
            token=token or self.token,
            environment=environment or self.environment,
            caller_name=caller_name or self._caller_name,
            caller_version=caller_version or self._caller_version,
        )

    def with_options(
        self,
        *,
        token: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> DataAPIClient:
        """
        Create a clone of this DataAPIClient with some changed attributes.

        Args:
            token: an Access Token to the database. Example: `"AstraCS:xyz..."`.
            caller_name: name of the application, or framework, on behalf of which
                the Data API and DevOps API calls are performed. This ends up in
                the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new DataAPIClient instance.

        Example:
            >>> another_client = my_client.with_options(
            ...     caller_name="caller_identity",
            ...     caller_version="1.2.0",
            ... )
        """

        return self._copy(
            token=token,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def set_caller(
        self,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> None:
        """
        Set a new identity for the application/framework on behalf of which
        the API calls will be performed (the "caller").

        New objects spawned from this client afterwards will inherit the new settings.

        Args:
            caller_name: name of the application, or framework, on behalf of which
                the API API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Example:
            >>> my_client.set_caller(caller_name="the_caller", caller_version="0.1.0")
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._caller_name = caller_name
        self._caller_version = caller_version

    def get_database(
        self,
        id: str,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        region: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> Database:
        """
        Get a Database object from this client, for doing data-related work.

        Args:
            id: the target database ID. The database must exist for the resulting
                object to be effectively used; in other words, this invocation
                does not create the database, just the object instance.
                Actual admin work can be achieved by using the AstraDBAdmin object.
            token: if supplied, is passed to the Database instead of the client token.
            namespace: if provided, is passed to the Database
                (it is left to the default otherwise).
            region: the region to use for connecting to the database. The
                database must be located in that region.
                Note that if this parameter is not passed, an additional
                DevOps API request is made to determine the default region
                and use it subsequently.
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".
            max_time_ms: a timeout, in milliseconds, for the DevOps API
                HTTP request should it be necessary (see the `region` argument).

        Returns:
            a Database object with which to work on Data API collections.

        Example:
            >>> my_db0 = my_client.get_database("01234567-...")
            >>> my_db1 = my_client.get_database("01234567-...", token="AstraCS:...")
            >>> my_db2 = my_client.get_database("01234567-...", region="us-west1")
            >>> my_coll = my_db0.create_collection("movies", dimension=512)
            >>> my_coll.insert_one({"title": "The Title"}, vector=...)

        Note:
            This method does not perform any admin-level operation through
            the DevOps API. For actual creation of a database, see the
            `create_database` method of class AstraDBAdmin.
        """

        # lazy importing here to avoid circular dependency
        from astrapy import Database

        # need to inspect for values?
        this_db_info: Optional[Dict[str, Any]] = None
        # handle overrides. Only region is needed (namespace can stay empty)
        if region:
            _region = region
        else:
            if this_db_info is None:
                logger.info(f"fetching raw database info for {id}")
                this_db_info = fetch_raw_database_info_from_id_token(
                    id=id,
                    token=self.token,
                    environment=self.environment,
                    max_time_ms=max_time_ms,
                )
                logger.info(f"finished fetching raw database info for {id}")
            _region = this_db_info["info"]["region"]

        _token = token or self.token
        _api_endpoint = build_api_endpoint(
            environment=self.environment,
            database_id=id,
            region=_region,
        )
        return Database(
            api_endpoint=_api_endpoint,
            token=_token,
            namespace=namespace,
            caller_name=self._caller_name,
            caller_version=self._caller_version,
            api_path=api_path,
            api_version=api_version,
        )

    def get_async_database(
        self,
        id: str,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        region: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> AsyncDatabase:
        """
        Get an AsyncDatabase object from this client.

        This method has identical behavior and signature as the sync
        counterpart `get_database`: please see that one for more details.
        """

        return self.get_database(
            id=id,
            token=token,
            namespace=namespace,
            region=region,
            api_path=api_path,
            api_version=api_version,
            max_time_ms=max_time_ms,
        ).to_async()

    def get_database_by_api_endpoint(
        self,
        api_endpoint: str,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> Database:
        """
        Get a Database object from this client, for doing data-related work.
        The Database is specified by an API Endpoint instead of the ID and a region.

        Args:
            api_endpoint: the full "API Endpoint" string used to reach the Data API.
                Example: "https://DATABASE_ID-REGION.apps.astra.datastax.com"
            token: if supplied, is passed to the Database instead of the client token.
            namespace: if provided, is passed to the Database
                (it is left to the default otherwise).
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".

        Returns:
            a Database object with which to work on Data API collections.

        Example:
            >>> my_db0 = my_client.get_database_by_api_endpoint("01234567-...")
            >>> my_db1 = my_client.get_database_by_api_endpoint(
            ...     "https://01234567-....apps.astra.datastax.com",
            ...     token="AstraCS:...",
            ... )
            >>> my_db2 = my_client.get_database_by_api_endpoint(
            ...     "https://01234567-....apps.astra.datastax.com",
            ...     namespace="the_other_namespace",
            ... )
            >>> my_coll = my_db0.create_collection("movies", dimension=512)
            >>> my_coll.insert_one({"title": "The Title"}, vector=...)

        Note:
            This method does not perform any admin-level operation through
            the DevOps API. For actual creation of a database, see the
            `create_database` method of class AstraDBAdmin.
        """

        # lazy importing here to avoid circular dependency
        from astrapy import Database

        parsed_api_endpoint = parse_api_endpoint(api_endpoint)
        if parsed_api_endpoint is not None:
            if parsed_api_endpoint.environment != self.environment:
                raise ValueError(
                    "Environment mismatch between client and provided "
                    "API endpoint. You can try adding "
                    f'`environment="{parsed_api_endpoint.environment}"` '
                    "to the DataAPIClient creation statement."
                )
            _token = token or self.token
            return Database(
                api_endpoint=api_endpoint,
                token=_token,
                namespace=namespace,
                caller_name=self._caller_name,
                caller_version=self._caller_version,
                api_path=api_path,
                api_version=api_version,
            )
        else:
            raise ValueError("Cannot parse the provided API endpoint.")

    def get_async_database_by_api_endpoint(
        self,
        api_endpoint: str,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> AsyncDatabase:
        """
        Get an AsyncDatabase object from this client, for doing data-related work.
        The Database is specified by an API Endpoint instead of the ID and a region.

        This method has identical behavior and signature as the sync
        counterpart `get_database_by_api_endpoint`: please see that one
        for more details.
        """

        return self.get_database_by_api_endpoint(
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace,
            api_path=api_path,
            api_version=api_version,
        ).to_async()

    def get_admin(
        self,
        *,
        token: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBAdmin:
        """
        Get an AstraDBAdmin instance corresponding to this client, for
        admin work such as managing databases.

        Args:
            token: if supplied, is passed to the Astra DB Admin instead of the
                client token. This may be useful when switching to a more powerful,
                admin-capable permission set.
            dev_ops_url: in case of custom deployments, this can be used to specify
                the URL to the DevOps API, such as "https://api.astra.datastax.com".
                Generally it can be omitted. The environment (prod/dev/...) is
                determined from the API Endpoint.
            dev_ops_api_version: this can specify a custom version of the DevOps API
                (such as "v2"). Generally not needed.

        Returns:
            An AstraDBAdmin instance, wich which to perform management at the
            database level.

        Example:
            >>> my_adm0 = my_client.get_admin()
            >>> my_adm1 = my_client.get_admin(token=more_powerful_token_override)
            >>> database_list = my_adm0.list_databases()
            >>> my_db_admin = my_adm0.create_database(
            ...     "the_other_database",
            ...     cloud_provider="AWS",
            ...     region="eu-west-1",
            ... )
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'that_other_one']
        """

        # lazy importing here to avoid circular dependency
        from astrapy.admin import AstraDBAdmin

        return AstraDBAdmin(
            token=token or self.token,
            environment=self.environment,
            caller_name=self._caller_name,
            caller_version=self._caller_version,
            dev_ops_url=dev_ops_url,
            dev_ops_api_version=dev_ops_api_version,
        )
