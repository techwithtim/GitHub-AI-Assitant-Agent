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
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

import httpx

from astrapy.core.ops import AstraDBOps
from astrapy.cursors import CommandCursor
from astrapy.info import AdminDatabaseInfo, DatabaseInfo
from astrapy.exceptions import (
    DevOpsAPIException,
    MultiCallTimeoutManager,
    base_timeout_info,
    to_dataapi_timeout_exception,
    ops_recast_method_sync,
    ops_recast_method_async,
)


if TYPE_CHECKING:
    from astrapy import AsyncDatabase, Database


logger = logging.getLogger(__name__)


DATABASE_POLL_NAMESPACE_SLEEP_TIME = 2
DATABASE_POLL_SLEEP_TIME = 15

STATUS_MAINTENANCE = "MAINTENANCE"
STATUS_ACTIVE = "ACTIVE"
STATUS_PENDING = "PENDING"
STATUS_INITIALIZING = "INITIALIZING"
STATUS_ERROR = "ERROR"
STATUS_TERMINATING = "TERMINATING"


class Environment:
    """
    Admitted values for `environment` property, such as the one denoting databases.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    PROD = "prod"
    DEV = "dev"
    TEST = "test"


database_id_matcher = re.compile(
    "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

api_endpoint_parser = re.compile(
    r"https://"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"-"
    r"([a-z0-9\-]+)"
    r".apps.astra[\-]{0,1}"
    r"(dev|test)?"
    r".datastax.com"
)


DEV_OPS_URL_MAP = {
    Environment.PROD: "https://api.astra.datastax.com",
    Environment.DEV: "https://api.dev.cloud.datastax.com",
    Environment.TEST: "https://api.test.cloud.datastax.com",
}

API_ENDPOINT_TEMPLATE_MAP = {
    Environment.PROD: "https://{database_id}-{region}.apps.astra.datastax.com",
    Environment.DEV: "https://{database_id}-{region}.apps.astra-dev.datastax.com",
    Environment.TEST: "https://{database_id}-{region}.apps.astra-test.datastax.com",
}


@dataclass
class ParsedAPIEndpoint:
    """
    The results of successfully parsing an Astra DB API endpoint, for internal
    by database metadata-related functions.

    Attributes:
        database_id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
        region: a region ID, such as "us-west1".
        environment: a label, whose value is one of Environment.PROD,
            Environment.DEV or Environment.TEST.
    """

    database_id: str
    region: str
    environment: str


def parse_api_endpoint(api_endpoint: str) -> Optional[ParsedAPIEndpoint]:
    """
    Parse an API Endpoint into a ParsedAPIEndpoint structure.

    Args:
        api_endpoint: a full API endpoint for the Data Api.

    Returns:
        The parsed ParsedAPIEndpoint. If parsing fails, return None.
    """

    match = api_endpoint_parser.match(api_endpoint)
    if match and match.groups():
        d_id, d_re, d_en_x = match.groups()
        return ParsedAPIEndpoint(
            database_id=d_id,
            region=d_re,
            environment=d_en_x if d_en_x else "prod",
        )
    else:
        return None


def build_api_endpoint(environment: str, database_id: str, region: str) -> str:
    """
    Build the API Endpoint full strings from database parameters.

    Args:
        environment: a label, whose value is one of Environment.PROD,
            Environment.DEV or Environment.TEST.
        database_id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
        region: a region ID, such as "us-west1".

    Returns:
        the endpoint string, such as "https://01234567-...-eu-west1.apps.datastax.com"
    """

    return API_ENDPOINT_TEMPLATE_MAP[environment].format(
        database_id=database_id,
        region=region,
    )


def fetch_raw_database_info_from_id_token(
    id: str,
    *,
    token: str,
    environment: str = Environment.PROD,
    max_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch database information through the DevOps API and return it in
    full, exactly like the API gives it back.

    Args:
        id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
        token: a valid token to access the database information.
        max_time_ms: a timeout, in milliseconds, for waiting on a response.

    Returns:
        The full response from the DevOps API about the database.
    """

    astra_db_ops = AstraDBOps(
        token=token,
        dev_ops_url=DEV_OPS_URL_MAP[environment],
    )
    try:
        gd_response = astra_db_ops.get_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        return gd_response
    except httpx.TimeoutException as texc:
        raise to_dataapi_timeout_exception(texc)


async def async_fetch_raw_database_info_from_id_token(
    id: str,
    *,
    token: str,
    environment: str = Environment.PROD,
    max_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch database information through the DevOps API and return it in
    full, exactly like the API gives it back.
    Async version of the function, for use in an asyncio context.

    Args:
        id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
        token: a valid token to access the database information.
        max_time_ms: a timeout, in milliseconds, for waiting on a response.

    Returns:
        The full response from the DevOps API about the database.
    """

    astra_db_ops = AstraDBOps(
        token=token,
        dev_ops_url=DEV_OPS_URL_MAP[environment],
    )
    try:
        gd_response = await astra_db_ops.async_get_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        return gd_response
    except httpx.TimeoutException as texc:
        raise to_dataapi_timeout_exception(texc)


def fetch_database_info(
    api_endpoint: str, token: str, namespace: str, max_time_ms: Optional[int] = None
) -> Optional[DatabaseInfo]:
    """
    Fetch database information through the DevOps API.

    Args:
        api_endpoint: a full API endpoint for the Data Api.
        token: a valid token to access the database information.
        namespace: the desired namespace that will be used in the result.
        max_time_ms: a timeout, in milliseconds, for waiting on a response.

    Returns:
        A DatabaseInfo object.
        If the API endpoint fails to be parsed, None is returned.
        For valid-looking endpoints, if something goes wrong an exception is raised.
    """

    parsed_endpoint = parse_api_endpoint(api_endpoint)
    if parsed_endpoint:
        gd_response = fetch_raw_database_info_from_id_token(
            id=parsed_endpoint.database_id,
            token=token,
            environment=parsed_endpoint.environment,
            max_time_ms=max_time_ms,
        )
        raw_info = gd_response["info"]
        if namespace not in raw_info["keyspaces"]:
            raise DevOpsAPIException(f"Namespace {namespace} not found on DB.")
        else:
            return DatabaseInfo(
                id=parsed_endpoint.database_id,
                region=parsed_endpoint.region,
                namespace=namespace,
                name=raw_info["name"],
                environment=parsed_endpoint.environment,
                raw_info=raw_info,
            )
    else:
        return None


async def async_fetch_database_info(
    api_endpoint: str, token: str, namespace: str, max_time_ms: Optional[int] = None
) -> Optional[DatabaseInfo]:
    """
    Fetch database information through the DevOps API.
    Async version of the function, for use in an asyncio context.

    Args:
        api_endpoint: a full API endpoint for the Data Api.
        token: a valid token to access the database information.
        namespace: the desired namespace that will be used in the result.
        max_time_ms: a timeout, in milliseconds, for waiting on a response.

    Returns:
        A DatabaseInfo object.
        If the API endpoint fails to be parsed, None is returned.
        For valid-looking endpoints, if something goes wrong an exception is raised.
    """

    parsed_endpoint = parse_api_endpoint(api_endpoint)
    if parsed_endpoint:
        gd_response = await async_fetch_raw_database_info_from_id_token(
            id=parsed_endpoint.database_id,
            token=token,
            environment=parsed_endpoint.environment,
            max_time_ms=max_time_ms,
        )
        raw_info = gd_response["info"]
        if namespace not in raw_info["keyspaces"]:
            raise DevOpsAPIException(f"Namespace {namespace} not found on DB.")
        else:
            return DatabaseInfo(
                id=parsed_endpoint.database_id,
                region=parsed_endpoint.region,
                namespace=namespace,
                name=raw_info["name"],
                environment=parsed_endpoint.environment,
                raw_info=raw_info,
            )
    else:
        return None


def _recast_as_admin_database_info(
    admin_database_info_dict: Dict[str, Any],
    *,
    environment: str,
) -> AdminDatabaseInfo:
    return AdminDatabaseInfo(
        info=DatabaseInfo(
            id=admin_database_info_dict["id"],
            region=admin_database_info_dict["info"]["region"],
            namespace=admin_database_info_dict["info"]["keyspace"],
            name=admin_database_info_dict["info"]["name"],
            environment=environment,
            raw_info=admin_database_info_dict["info"],
        ),
        available_actions=admin_database_info_dict.get("availableActions"),
        cost=admin_database_info_dict["cost"],
        cqlsh_url=admin_database_info_dict["cqlshUrl"],
        creation_time=admin_database_info_dict["creationTime"],
        data_endpoint_url=admin_database_info_dict["dataEndpointUrl"],
        grafana_url=admin_database_info_dict["grafanaUrl"],
        graphql_url=admin_database_info_dict["graphqlUrl"],
        id=admin_database_info_dict["id"],
        last_usage_time=admin_database_info_dict["lastUsageTime"],
        metrics=admin_database_info_dict["metrics"],
        observed_status=admin_database_info_dict["observedStatus"],
        org_id=admin_database_info_dict["orgId"],
        owner_id=admin_database_info_dict["ownerId"],
        status=admin_database_info_dict["status"],
        storage=admin_database_info_dict["storage"],
        termination_time=admin_database_info_dict["terminationTime"],
        raw_info=admin_database_info_dict,
    )


class AstraDBAdmin:
    """
    An "admin" object, able to perform administrative tasks at the databases
    level, such as creating, listing or dropping databases.

    Args:
        token: an access token with enough permission to perform admin tasks.
        environment: a label, whose value is one of Environment.PROD (default),
            Environment.DEV or Environment.TEST.
        caller_name: name of the application, or framework, on behalf of which
            the DevOps API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.
        dev_ops_url: in case of custom deployments, this can be used to specify
            the URL to the DevOps API, such as "https://api.astra.datastax.com".
            Generally it can be omitted. The environment (prod/dev/...) is
            determined from the API Endpoint.
        dev_ops_api_version: this can specify a custom version of the DevOps API
            (such as "v2"). Generally not needed.

    Example:
        >>> from astrapy import DataAPIClient
        >>> my_client = DataAPIClient("AstraCS:...")
        >>> my_astra_db_admin = my_client.get_admin()
        >>> database_list = my_astra_db_admin.list_databases()
        >>> len(database_list)
        3
        >>> database_list[2].id
        '01234567-...'
        >>> my_db_admin = my_astra_db_admin.get_database_admin("01234567-...")
        >>> my_db_admin.list_namespaces()
        ['default_keyspace', 'staging_namespace']
    """

    def __init__(
        self,
        token: str,
        *,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> None:
        self.token = token
        self.environment = environment or Environment.PROD
        if dev_ops_url is None:
            self.dev_ops_url = DEV_OPS_URL_MAP[self.environment]
        else:
            self.dev_ops_url = dev_ops_url
        self._caller_name = caller_name
        self._caller_version = caller_version
        self._dev_ops_url = dev_ops_url
        self._dev_ops_api_version = dev_ops_api_version
        self._astra_db_ops = AstraDBOps(
            token=self.token,
            dev_ops_url=self.dev_ops_url,
            dev_ops_api_version=dev_ops_api_version,
            caller_name=caller_name,
            caller_version=caller_version,
        )

    def __repr__(self) -> str:
        env_desc: str
        if self.environment == Environment.PROD:
            env_desc = ""
        else:
            env_desc = f', environment="{self.environment}"'
        return f'{self.__class__.__name__}("{self.token[:12]}..."{env_desc})'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AstraDBAdmin):
            return all(
                [
                    self.token == other.token,
                    self.environment == other.environment,
                    self.dev_ops_url == other.dev_ops_url,
                    self.dev_ops_url == other.dev_ops_url,
                    self._caller_name == other._caller_name,
                    self._caller_version == other._caller_version,
                    self._dev_ops_url == other._dev_ops_url,
                    self._dev_ops_api_version == other._dev_ops_api_version,
                    self._astra_db_ops == other._astra_db_ops,
                ]
            )
        else:
            return False

    def _copy(
        self,
        *,
        token: Optional[str] = None,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBAdmin:
        return AstraDBAdmin(
            token=token or self.token,
            environment=environment or self.environment,
            caller_name=caller_name or self._caller_name,
            caller_version=caller_version or self._caller_version,
            dev_ops_url=dev_ops_url or self._dev_ops_url,
            dev_ops_api_version=dev_ops_api_version or self._dev_ops_api_version,
        )

    def with_options(
        self,
        *,
        token: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AstraDBAdmin:
        """
        Create a clone of this AstraDBAdmin with some changed attributes.

        Args:
            token: an Access Token to the database. Example: `"AstraCS:xyz..."`.
            caller_name: name of the application, or framework, on behalf of which
                the Data API and DevOps API calls are performed. This ends up in
                the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new AstraDBAdmin instance.

        Example:
            >>> another_astra_db_admin = my_astra_db_admin.with_options(
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
        the DevOps API calls will be performed (the "caller").

        New objects spawned from this client afterwards will inherit the new settings.

        Args:
            caller_name: name of the application, or framework, on behalf of which
                the DevOps API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Example:
            >>> my_astra_db_admin.set_caller(
            ...     caller_name="the_caller",
            ...     caller_version="0.1.0",
            ... )
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._caller_name = caller_name
        self._caller_version = caller_version
        self._astra_db_ops.set_caller(caller_name, caller_version)

    @ops_recast_method_sync
    def list_databases(
        self,
        *,
        max_time_ms: Optional[int] = None,
    ) -> CommandCursor[AdminDatabaseInfo]:
        """
        Get the list of databases, as obtained with a request to the DevOps API.

        Args:
            max_time_ms: a timeout, in milliseconds, for the API request.

        Returns:
            A CommandCursor to iterate over the detected databases,
            represented as AdminDatabaseInfo objects.

        Example:
            >>> database_cursor = my_astra_db_admin.list_databases()
            >>> database_list = list(database_cursor)
            >>> len(database_list)
            3
            >>> database_list[2].id
            '01234567-...'
            >>> database_list[2].status
            'ACTIVE'
            >>> database_list[2].info.region
            'eu-west-1'
        """

        logger.info("getting databases")
        gd_list_response = self._astra_db_ops.get_databases(
            timeout_info=base_timeout_info(max_time_ms)
        )
        logger.info("finished getting databases")
        if not isinstance(gd_list_response, list):
            raise DevOpsAPIException(
                "Faulty response from get-databases DevOps API command.",
            )
        else:
            # we know this is a list of dicts which need a little adjusting
            return CommandCursor(
                address=self._astra_db_ops.base_url,
                items=[
                    _recast_as_admin_database_info(
                        db_dict,
                        environment=self.environment,
                    )
                    for db_dict in gd_list_response
                ],
            )

    @ops_recast_method_async
    async def async_list_databases(
        self,
        *,
        max_time_ms: Optional[int] = None,
    ) -> CommandCursor[AdminDatabaseInfo]:
        """
        Get the list of databases, as obtained with a request to the DevOps API.
        Async version of the method, for use in an asyncio context.

        Args:
            max_time_ms: a timeout, in milliseconds, for the API request.

        Returns:
            A CommandCursor to iterate over the detected databases,
            represented as AdminDatabaseInfo objects.
            Note that the return type is not an awaitable, rather
            a regular iterable, e.g. for use in ordinary "for" loops.

        Example:
            >>> async def check_if_db_exists(db_id: str) -> bool:
            ...     db_cursor = await my_astra_db_admin.async_list_databases()
            ...     db_list = list(dd_cursor)
            ...     return db_id in db_list
            ...
            >>> asyncio.run(check_if_db_exists("xyz"))
            True
            >>> asyncio.run(check_if_db_exists("01234567-..."))
            False
        """

        logger.info("getting databases, async")
        gd_list_response = await self._astra_db_ops.async_get_databases(
            timeout_info=base_timeout_info(max_time_ms)
        )
        logger.info("finished getting databases, async")
        if not isinstance(gd_list_response, list):
            raise DevOpsAPIException(
                "Faulty response from get-databases DevOps API command.",
            )
        else:
            # we know this is a list of dicts which need a little adjusting
            return CommandCursor(
                address=self._astra_db_ops.base_url,
                items=[
                    _recast_as_admin_database_info(
                        db_dict,
                        environment=self.environment,
                    )
                    for db_dict in gd_list_response
                ],
            )

    @ops_recast_method_sync
    def database_info(
        self, id: str, *, max_time_ms: Optional[int] = None
    ) -> AdminDatabaseInfo:
        """
        Get the full information on a given database, through a request to the DevOps API.

        Args:
            id: the ID of the target database, e. g.
                "01234567-89ab-cdef-0123-456789abcdef".
            max_time_ms: a timeout, in milliseconds, for the API request.

        Returns:
            An AdminDatabaseInfo object.

        Example:
            >>> details_of_my_db = my_astra_db_admin.database_info("01234567-...")
            >>> details_of_my_db.id
            '01234567-...'
            >>> details_of_my_db.status
            'ACTIVE'
            >>> details_of_my_db.info.region
            'eu-west-1'
        """

        logger.info(f"getting database info for '{id}'")
        gd_response = self._astra_db_ops.get_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished getting database info for '{id}'")
        if not isinstance(gd_response, dict):
            raise DevOpsAPIException(
                "Faulty response from get-database DevOps API command.",
            )
        else:
            return _recast_as_admin_database_info(
                gd_response,
                environment=self.environment,
            )

    @ops_recast_method_async
    async def async_database_info(
        self, id: str, *, max_time_ms: Optional[int] = None
    ) -> AdminDatabaseInfo:
        """
        Get the full information on a given database, through a request to the DevOps API.
        This is an awaitable method suitable for use within an asyncio event loop.

        Args:
            id: the ID of the target database, e. g.
                "01234567-89ab-cdef-0123-456789abcdef".
            max_time_ms: a timeout, in milliseconds, for the API request.

        Returns:
            An AdminDatabaseInfo object.

        Example:
            >>> async def check_if_db_active(db_id: str) -> bool:
            ...     db_info = await my_astra_db_admin.async_database_info(db_id)
            ...     return db_info.status == "ACTIVE"
            ...
            >>> asyncio.run(check_if_db_active("01234567-..."))
            True
        """

        logger.info(f"getting database info for '{id}', async")
        gd_response = await self._astra_db_ops.async_get_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"finished getting database info for '{id}', async")
        if not isinstance(gd_response, dict):
            raise DevOpsAPIException(
                "Faulty response from get-database DevOps API command.",
            )
        else:
            return _recast_as_admin_database_info(
                gd_response,
                environment=self.environment,
            )

    @ops_recast_method_sync
    def create_database(
        self,
        name: str,
        *,
        cloud_provider: str,
        region: str,
        namespace: Optional[str] = None,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Create a database as requested, optionally waiting for it to be ready.

        Args:
            name: the desired name for the database.
            cloud_provider: one of 'aws', 'gcp' or 'azure'.
            region: any of the available cloud regions.
            namespace: name for the one namespace the database starts with.
                If omitted, DevOps API will use its default.
            wait_until_active: if True (default), the method returns only after
                the newly-created database is in ACTIVE state (a few minutes,
                usually). If False, it will return right after issuing the
                creation request to the DevOps API, and it will be responsibility
                of the caller to check the database status before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the creation request
                has not reached the API server.

        Returns:
            An AstraDBDatabaseAdmin instance.

        Example:
            >>> my_new_db_admin = my_astra_db_admin.create_database(
            ...     "new_database",
            ...     cloud_provider="aws",
            ...     region="ap-south-1",
            ... )
            >>> my_new_db = my_new_db_admin.get_database()
            >>> my_coll = my_new_db.create_collection("movies", dimension=512)
            >>> my_coll.insert_one({"title": "The Title"}, vector=...)
        """

        database_definition = {
            k: v
            for k, v in {
                "name": name,
                "tier": "serverless",
                "cloudProvider": cloud_provider,
                "region": region,
                "capacityUnits": 1,
                "dbType": "vector",
                "keyspace": namespace,
            }.items()
            if v is not None
        }
        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"creating database {name}/({cloud_provider}, {region})")
        cd_response = self._astra_db_ops.create_database(
            database_definition=database_definition,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            "devops api returned from creating database "
            f"{name}/({cloud_provider}, {region})"
        )
        if cd_response is not None and "id" in cd_response:
            new_database_id = cd_response["id"]
            if wait_until_active:
                last_status_seen = STATUS_PENDING
                while last_status_seen in {STATUS_PENDING, STATUS_INITIALIZING}:
                    logger.info(f"sleeping to poll for status of '{new_database_id}'")
                    time.sleep(DATABASE_POLL_SLEEP_TIME)
                    last_db_info = self.database_info(
                        id=new_database_id,
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    last_status_seen = last_db_info.status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database {name} entered unexpected status {last_status_seen} after PENDING"
                    )
            # return the database instance
            logger.info(
                f"finished creating database '{new_database_id}' = "
                f"{name}/({cloud_provider}, {region})"
            )
            return AstraDBDatabaseAdmin.from_astra_db_admin(
                id=new_database_id,
                astra_db_admin=self,
            )
        else:
            raise DevOpsAPIException("Could not create the database.")

    @ops_recast_method_async
    async def async_create_database(
        self,
        name: str,
        *,
        cloud_provider: str,
        region: str,
        namespace: Optional[str] = None,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Create a database as requested, optionally waiting for it to be ready.
        This is an awaitable method suitable for use within an asyncio event loop.

        Args:
            name: the desired name for the database.
            cloud_provider: one of 'aws', 'gcp' or 'azure'.
            region: any of the available cloud regions.
            namespace: name for the one namespace the database starts with.
                If omitted, DevOps API will use its default.
            wait_until_active: if True (default), the method returns only after
                the newly-created database is in ACTIVE state (a few minutes,
                usually). If False, it will return right after issuing the
                creation request to the DevOps API, and it will be responsibility
                of the caller to check the database status before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the creation request
                has not reached the API server.

        Returns:
            An AstraDBDatabaseAdmin instance.

        Example:
            >>> asyncio.run(
            ...     my_astra_db_admin.async_create_database(
            ...         "new_database",
            ...         cloud_provider="aws",
            ...         region="ap-south-1",
            ....    )
            ... )
            AstraDBDatabaseAdmin(id=...)
        """

        database_definition = {
            k: v
            for k, v in {
                "name": name,
                "tier": "serverless",
                "cloudProvider": cloud_provider,
                "region": region,
                "capacityUnits": 1,
                "dbType": "vector",
                "keyspace": namespace,
            }.items()
            if v is not None
        }
        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"creating database {name}/({cloud_provider}, {region}), async")
        cd_response = await self._astra_db_ops.async_create_database(
            database_definition=database_definition,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            "devops api returned from creating database "
            f"{name}/({cloud_provider}, {region}), async"
        )
        if cd_response is not None and "id" in cd_response:
            new_database_id = cd_response["id"]
            if wait_until_active:
                last_status_seen = STATUS_PENDING
                while last_status_seen in {STATUS_PENDING, STATUS_INITIALIZING}:
                    logger.info(
                        f"sleeping to poll for status of '{new_database_id}', async"
                    )
                    await asyncio.sleep(DATABASE_POLL_SLEEP_TIME)
                    last_db_info = await self.async_database_info(
                        id=new_database_id,
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    last_status_seen = last_db_info.status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database {name} entered unexpected status {last_status_seen} after PENDING"
                    )
            # return the database instance
            logger.info(
                f"finished creating database '{new_database_id}' = "
                f"{name}/({cloud_provider}, {region}), async"
            )
            return AstraDBDatabaseAdmin.from_astra_db_admin(
                id=new_database_id,
                astra_db_admin=self,
            )
        else:
            raise DevOpsAPIException("Could not create the database.")

    @ops_recast_method_sync
    def drop_database(
        self,
        id: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop a database, i.e. delete it completely and permanently with all its data.

        Args:
            id: The ID of the database to drop, e. g.
                "01234567-89ab-cdef-0123-456789abcdef".
            wait_until_active: if True (default), the method returns only after
                the database has actually been deleted (generally a few minutes).
                If False, it will return right after issuing the
                drop request to the DevOps API, and it will be responsibility
                of the caller to check the database status/availability
                after that, if desired.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> database_list_pre = my_astra_db_admin.list_databases()
            >>> len(database_list_pre)
            3
            >>> my_astra_db_admin.drop_database("01234567-...")
            {'ok': 1}
            >>> database_list_post = my_astra_db_admin.list_databases()
            >>> len(database_list_post)
            2
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"dropping database '{id}'")
        te_response = self._astra_db_ops.terminate_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"devops api returned from dropping database '{id}'")
        if te_response == id:
            if wait_until_active:
                last_status_seen: Optional[str] = STATUS_TERMINATING
                _db_name: Optional[str] = None
                while last_status_seen == STATUS_TERMINATING:
                    logger.info(f"sleeping to poll for status of '{id}'")
                    time.sleep(DATABASE_POLL_SLEEP_TIME)
                    #
                    detected_databases = [
                        a_db_info
                        for a_db_info in self.list_databases(
                            max_time_ms=timeout_manager.remaining_timeout_ms(),
                        )
                        if a_db_info.id == id
                    ]
                    if detected_databases:
                        last_status_seen = detected_databases[0].status
                        _db_name = detected_databases[0].info.name
                    else:
                        last_status_seen = None
                if last_status_seen is not None:
                    _name_desc = f" ({_db_name})" if _db_name else ""
                    raise DevOpsAPIException(
                        f"Database {id}{_name_desc} entered unexpected status {last_status_seen} after PENDING"
                    )
            logger.info(f"finished dropping database '{id}'")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful terminate-database DevOps API request for {id}."
            )

    @ops_recast_method_async
    async def async_drop_database(
        self,
        id: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop a database, i.e. delete it completely and permanently with all its data.
        Async version of the method, for use in an asyncio context.

        Args:
            id: The ID of the database to drop, e. g.
                "01234567-89ab-cdef-0123-456789abcdef".
            wait_until_active: if True (default), the method returns only after
                the database has actually been deleted (generally a few minutes).
                If False, it will return right after issuing the
                drop request to the DevOps API, and it will be responsibility
                of the caller to check the database status/availability
                after that, if desired.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> asyncio.run(
            ...     my_astra_db_admin.async_drop_database("01234567-...")
            ... )
            {'ok': 1}
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"dropping database '{id}', async")
        te_response = await self._astra_db_ops.async_terminate_database(
            database=id,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(f"devops api returned from dropping database '{id}', async")
        if te_response == id:
            if wait_until_active:
                last_status_seen: Optional[str] = STATUS_TERMINATING
                _db_name: Optional[str] = None
                while last_status_seen == STATUS_TERMINATING:
                    logger.info(f"sleeping to poll for status of '{id}', async")
                    await asyncio.sleep(DATABASE_POLL_SLEEP_TIME)
                    #
                    detected_databases = [
                        a_db_info
                        for a_db_info in await self.async_list_databases(
                            max_time_ms=timeout_manager.remaining_timeout_ms(),
                        )
                        if a_db_info.id == id
                    ]
                    if detected_databases:
                        last_status_seen = detected_databases[0].status
                        _db_name = detected_databases[0].info.name
                    else:
                        last_status_seen = None
                if last_status_seen is not None:
                    _name_desc = f" ({_db_name})" if _db_name else ""
                    raise DevOpsAPIException(
                        f"Database {id}{_name_desc} entered unexpected status {last_status_seen} after PENDING"
                    )
            logger.info(f"finished dropping database '{id}', async")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful terminate-database DevOps API request for {id}."
            )

    def get_database_admin(self, id: str) -> AstraDBDatabaseAdmin:
        """
        Create an AstraDBDatabaseAdmin object for admin work within a certain database.

        Args:
            id: the ID of the target database, e. g. "01234567-89ab-cdef-0123-456789abcdef".

        Returns:
            An AstraDBDatabaseAdmin instance representing the requested database.

        Example:
            >>> my_db_admin = my_astra_db_admin.get_database_admin("01234567-...")
            >>> my_db_admin.list_namespaces()
            ['default_keyspace']
            >>> my_db_admin.create_namespace("that_other_one")
            {'ok': 1}
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'that_other_one']

        Note:
            This method does not perform any admin-level operation through
            the DevOps API. For actual creation of a database, see the
            `create_database` method.
        """

        return AstraDBDatabaseAdmin.from_astra_db_admin(
            id=id,
            astra_db_admin=self,
        )

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
        Create a Database instance for a specific database, to be used
        when doing data-level work (such as creating/managing collections).

        Args:
            id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
            token: if supplied, is passed to the Database instead of
                the one set for this object.
            namespace: used to specify a certain namespace the resulting
                Database will primarily work on. If not specified, similar
                as for `region`, an additional DevOps API call reveals
                the default namespace for the target database.
            region: the region to use for connecting to the database. The
                database must be located in that region.
                Note that if this parameter is not passed, an additional
                DevOps API request is made to determine the default region
                and use it subsequently.
                If both `namespace` and `region` are missing, a single
                DevOps API request is made.
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".
            max_time_ms: a timeout, in milliseconds, for the DevOps API
                HTTP request should it be necessary (see the `region` argument).

        Returns:
            A Database object ready to be used.

        Example:
            >>> my_db = my_astra_db_admin.get_database(
            ...     "01234567-...",
            ...     region="us-east1",
            ... )
            >>> coll = my_db.create_collection("movies", dimension=512)
            >>> my_coll.insert_one({"title": "The Title"}, vector=...)

        Note:
            This method does not perform any admin-level operation through
            the DevOps API. For actual creation of a database, see the
            `create_database` method of class AstraDBAdmin.
        """

        # lazy importing here to avoid circular dependency
        from astrapy import Database

        # need to inspect for values?
        this_db_info: Optional[AdminDatabaseInfo] = None
        # handle overrides
        _token = token or self.token
        if namespace:
            _namespace = namespace
        else:
            if this_db_info is None:
                this_db_info = self.database_info(id, max_time_ms=max_time_ms)
            _namespace = this_db_info.info.namespace
        if region:
            _region = region
        else:
            if this_db_info is None:
                this_db_info = self.database_info(id, max_time_ms=max_time_ms)
            _region = this_db_info.info.region

        _api_endpoint = build_api_endpoint(
            environment=self.environment,
            database_id=id,
            region=_region,
        )
        return Database(
            api_endpoint=_api_endpoint,
            token=_token,
            namespace=_namespace,
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
    ) -> AsyncDatabase:
        """
        Create an AsyncDatabase instance for a specific database, to be used
        when doing data-level work (such as creating/managing collections).

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
        ).to_async()


class DatabaseAdmin(ABC):
    """
    An abstract class defining the interface for a database admin object.
    This supports generic namespace crud, as well as spawning databases,
    without committing to a specific database architecture (e.g. Astra DB).
    """

    @abstractmethod
    def list_namespaces(self, *pargs: Any, **kwargs: Any) -> List[str]:
        """Get a list of namespaces for the database."""
        ...

    @abstractmethod
    def create_namespace(self, name: str, *pargs: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Create a namespace in the database, returning {'ok': 1} if successful.
        """
        ...

    @abstractmethod
    def drop_namespace(self, name: str, *pargs: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Drop (delete) a namespace from the database, returning {'ok': 1} if successful.
        """
        ...

    @abstractmethod
    async def async_list_namespaces(self, *pargs: Any, **kwargs: Any) -> List[str]:
        """
        Get a list of namespaces for the database.
        (Async version of the method.)
        """
        ...

    @abstractmethod
    async def async_create_namespace(
        self, name: str, *pargs: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create a namespace in the database, returning {'ok': 1} if successful.
        (Async version of the method.)
        """
        ...

    @abstractmethod
    async def async_drop_namespace(
        self, name: str, *pargs: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Drop (delete) a namespace from the database, returning {'ok': 1} if successful.
        (Async version of the method.)
        """
        ...

    @abstractmethod
    def get_database(self, *pargs: Any, **kwargs: Any) -> Database:
        """Get a Database object from this database admin."""
        ...

    @abstractmethod
    def get_async_database(self, *pargs: Any, **kwargs: Any) -> AsyncDatabase:
        """Get an AsyncDatabase object from this database admin."""
        ...


class AstraDBDatabaseAdmin(DatabaseAdmin):
    """
    An "admin" object, able to perform administrative tasks at the namespaces level
    (i.e. within a certani database), such as creating/listing/dropping namespaces.

    This is one layer below the AstraDBAdmin concept, in that it is tied to
    a single database and enables admin work within it. As such, it is generally
    created by a method call on an AstraDBAdmin.

    Args:
        id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
        token: an access token with enough permission to perform admin tasks.
        environment: a label, whose value is one of Environment.PROD (default),
            Environment.DEV or Environment.TEST.
        caller_name: name of the application, or framework, on behalf of which
            the DevOps API calls are performed. This ends up in the request user-agent.
        caller_version: version of the caller.
        dev_ops_url: in case of custom deployments, this can be used to specify
            the URL to the DevOps API, such as "https://api.astra.datastax.com".
            Generally it can be omitted. The environment (prod/dev/...) is
            determined from the API Endpoint.
        dev_ops_api_version: this can specify a custom version of the DevOps API
            (such as "v2"). Generally not needed.

    Example:
        >>> from astrapy import DataAPIClient
        >>> my_client = DataAPIClient("AstraCS:...")
        >>> admin_for_my_db = my_client.get_admin().get_database_admin("01234567-...")
        >>> admin_for_my_db.list_namespaces()
        ['default_keyspace', 'staging_namespace']
        >>> admin_for_my_db.info().status
        'ACTIVE'

    Note:
        creating an instance of AstraDBDatabaseAdmin does not trigger actual creation
        of the database itself, which should exist beforehand. To create databases,
        see the AstraDBAdmin class.
    """

    def __init__(
        self,
        id: str,
        *,
        token: str,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> None:
        self.id = id
        self.token = token
        self.environment = environment or Environment.PROD
        self._astra_db_admin = AstraDBAdmin(
            token=self.token,
            environment=self.environment,
            caller_name=caller_name,
            caller_version=caller_version,
            dev_ops_url=dev_ops_url,
            dev_ops_api_version=dev_ops_api_version,
        )

    def __repr__(self) -> str:
        env_desc: str
        if self.environment == Environment.PROD:
            env_desc = ""
        else:
            env_desc = f', environment="{self.environment}"'
        return (
            f'{self.__class__.__name__}(id="{self.id}", '
            f'"{self.token[:12]}..."{env_desc})'
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AstraDBDatabaseAdmin):
            return all(
                [
                    self.id == other.id,
                    self.token == other.token,
                    self.environment == other.environment,
                    self._astra_db_admin == other._astra_db_admin,
                ]
            )
        else:
            return False

    def _copy(
        self,
        id: Optional[str] = None,
        token: Optional[str] = None,
        environment: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBDatabaseAdmin:
        return AstraDBDatabaseAdmin(
            id=id or self.id,
            token=token or self.token,
            environment=environment or self.environment,
            caller_name=caller_name or self._astra_db_admin._caller_name,
            caller_version=caller_version or self._astra_db_admin._caller_version,
            dev_ops_url=dev_ops_url or self._astra_db_admin._dev_ops_url,
            dev_ops_api_version=dev_ops_api_version
            or self._astra_db_admin._dev_ops_api_version,
        )

    def with_options(
        self,
        *,
        id: Optional[str] = None,
        token: Optional[str] = None,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Create a clone of this AstraDBDatabaseAdmin with some changed attributes.

        Args:
            id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
            token: an Access Token to the database. Example: `"AstraCS:xyz..."`.
            caller_name: name of the application, or framework, on behalf of which
                the Data API and DevOps API calls are performed. This ends up in
                the request user-agent.
            caller_version: version of the caller.

        Returns:
            a new AstraDBDatabaseAdmin instance.

        Example:
            >>> admin_for_my_other_db = admin_for_my_db.with_options(
            ...     id="abababab-0101-2323-4545-6789abcdef01",
            ... )
        """

        return self._copy(
            id=id,
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
        the DevOps API calls will be performed (the "caller").

        New objects spawned from this client afterwards will inherit the new settings.

        Args:
            caller_name: name of the application, or framework, on behalf of which
                the DevOps API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.

        Example:
            >>> admin_for_my_db.set_caller(
            ...     caller_name="the_caller",
            ...     caller_version="0.1.0",
            ... )
        """

        logger.info(f"setting caller to {caller_name}/{caller_version}")
        self._astra_db_admin.set_caller(caller_name, caller_version)

    @staticmethod
    def from_astra_db_admin(
        id: str, *, astra_db_admin: AstraDBAdmin
    ) -> AstraDBDatabaseAdmin:
        """
        Create an AstraDBDatabaseAdmin from an AstraDBAdmin and a database ID.

        Args:
            id: e. g. "01234567-89ab-cdef-0123-456789abcdef".
            astra_db_admin: an AstraDBAdmin object that has visibility over
                the target database.

        Returns:
            An AstraDBDatabaseAdmin object, for admin work within the database.

        Example:
            >>> from astrapy import DataAPIClient, AstraDBDatabaseAdmin
            >>> admin_for_my_db = AstraDBDatabaseAdmin.from_astra_db_admin(
            ...     id="01234567-...",
            ...     astra_db_admin=DataAPIClient("AstraCS:...").get_admin(),
            ... )
            >>> admin_for_my_db.list_namespaces()
            ['default_keyspace', 'staging_namespace']
            >>> admin_for_my_db.info().status
            'ACTIVE'

        Note:
            Creating an instance of AstraDBDatabaseAdmin does not trigger actual creation
            of the database itself, which should exist beforehand. To create databases,
            see the AstraDBAdmin class.
        """

        return AstraDBDatabaseAdmin(
            id=id,
            token=astra_db_admin.token,
            environment=astra_db_admin.environment,
            caller_name=astra_db_admin._caller_name,
            caller_version=astra_db_admin._caller_version,
            dev_ops_url=astra_db_admin._dev_ops_url,
            dev_ops_api_version=astra_db_admin._dev_ops_api_version,
        )

    @staticmethod
    def from_api_endpoint(
        api_endpoint: str,
        *,
        token: str,
        caller_name: Optional[str] = None,
        caller_version: Optional[str] = None,
        dev_ops_url: Optional[str] = None,
        dev_ops_api_version: Optional[str] = None,
    ) -> AstraDBDatabaseAdmin:
        """
        Create an AstraDBDatabaseAdmin from an API Endpoint and optionally a token.

        Args:
            api_endpoint: a full API endpoint for the Data Api.
            token: an access token with enough permissions to do admin work.
            caller_name: name of the application, or framework, on behalf of which
                the DevOps API calls are performed. This ends up in the request user-agent.
            caller_version: version of the caller.
            dev_ops_url: in case of custom deployments, this can be used to specify
                the URL to the DevOps API, such as "https://api.astra.datastax.com".
                Generally it can be omitted. The environment (prod/dev/...) is
                determined from the API Endpoint.
            dev_ops_api_version: this can specify a custom version of the DevOps API
                (such as "v2"). Generally not needed.

        Returns:
            An AstraDBDatabaseAdmin object, for admin work within the database.

        Example:
            >>> from astrapy import AstraDBDatabaseAdmin
            >>> admin_for_my_db = AstraDBDatabaseAdmin.from_api_endpoint(
            ...     api_endpoint="https://01234567-....apps.astra.datastax.com",
            ...     token="AstraCS:...",
            ... )
            >>> admin_for_my_db.list_namespaces()
            ['default_keyspace', 'another_namespace']
            >>> admin_for_my_db.info().status
            'ACTIVE'

        Note:
            Creating an instance of AstraDBDatabaseAdmin does not trigger actual creation
            of the database itself, which should exist beforehand. To create databases,
            see the AstraDBAdmin class.
        """

        parsed_api_endpoint = parse_api_endpoint(api_endpoint)
        if parsed_api_endpoint:
            return AstraDBDatabaseAdmin(
                id=parsed_api_endpoint.database_id,
                token=token,
                environment=parsed_api_endpoint.environment,
                caller_name=caller_name,
                caller_version=caller_version,
                dev_ops_url=dev_ops_url,
                dev_ops_api_version=dev_ops_api_version,
            )
        else:
            raise ValueError("Cannot parse the provided API endpoint.")

    def info(self, *, max_time_ms: Optional[int] = None) -> AdminDatabaseInfo:
        """
        Query the DevOps API for the full info on this database.

        Args:
            max_time_ms: a timeout, in milliseconds, for the DevOps API request.

        Returns:
            An AdminDatabaseInfo object.

        Example:
            >>> my_db_info = admin_for_my_db.info()
            >>> my_db_info.status
            'ACTIVE'
            >>> my_db_info.info.region
            'us-east1'
        """

        logger.info(f"getting info ('{self.id}')")
        req_response = self._astra_db_admin.database_info(
            id=self.id,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished getting info ('{self.id}')")
        return req_response  # type: ignore[no-any-return]

    async def async_info(
        self, *, max_time_ms: Optional[int] = None
    ) -> AdminDatabaseInfo:
        """
        Query the DevOps API for the full info on this database.
        Async version of the method, for use in an asyncio context.

        Args:
            max_time_ms: a timeout, in milliseconds, for the DevOps API request.

        Returns:
            An AdminDatabaseInfo object.

        Example:
            >>> async def wait_until_active(db_admin: AstraDBDatabaseAdmin) -> None:
            ...     while True:
            ...         info = await db_admin.async_info()
            ...         if info.status == "ACTIVE":
            ...             return
            ...
            >>> asyncio.run(wait_until_active(admin_for_my_db))
        """

        logger.info(f"getting info ('{self.id}'), async")
        req_response = await self._astra_db_admin.async_database_info(
            id=self.id,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished getting info ('{self.id}'), async")
        return req_response  # type: ignore[no-any-return]

    def list_namespaces(self, *, max_time_ms: Optional[int] = None) -> List[str]:
        """
        Query the DevOps API for a list of the namespaces in the database.

        Args:
            max_time_ms: a timeout, in milliseconds, for the DevOps API request.

        Returns:
            A list of the namespaces, each a string, in no particular order.

        Example:
            >>> admin_for_my_db.list_namespaces()
            ['default_keyspace', 'staging_namespace']
        """

        logger.info(f"getting namespaces ('{self.id}')")
        info = self.info(max_time_ms=max_time_ms)
        logger.info(f"finished getting namespaces ('{self.id}')")
        if info.raw_info is None:
            raise DevOpsAPIException("Could not get the namespace list.")
        else:
            return info.raw_info["info"]["keyspaces"]  # type: ignore[no-any-return]

    async def async_list_namespaces(
        self, *, max_time_ms: Optional[int] = None
    ) -> List[str]:
        """
        Query the DevOps API for a list of the namespaces in the database.
        Async version of the method, for use in an asyncio context.

        Args:
            max_time_ms: a timeout, in milliseconds, for the DevOps API request.

        Returns:
            A list of the namespaces, each a string, in no particular order.

        Example:
            >>> async def check_if_ns_exists(
            ...     db_admin: AstraDBDatabaseAdmin, namespace: str
            ... ) -> bool:
            ...     ns_list = await db_admin.async_list_namespaces()
            ...     return namespace in ns_list
            ...
            >>> asyncio.run(check_if_ns_exists(admin_for_my_db, "dragons"))
            False
            >>> asyncio.run(check_if_db_exists(admin_for_my_db, "app_namespace"))
            True
        """

        logger.info(f"getting namespaces ('{self.id}'), async")
        info = await self.async_info(max_time_ms=max_time_ms)
        logger.info(f"finished getting namespaces ('{self.id}'), async")
        if info.raw_info is None:
            raise DevOpsAPIException("Could not get the namespace list.")
        else:
            return info.raw_info["info"]["keyspaces"]  # type: ignore[no-any-return]

    @ops_recast_method_sync
    def create_namespace(
        self,
        name: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a namespace in this database as requested,
        optionally waiting for it to be ready.

        Args:
            name: the namespace name. If supplying a namespace that exists
                already, the method call proceeds as usual, no errors are
                raised, and the whole invocation is a no-op.
            wait_until_active: if True (default), the method returns only after
                the target database is in ACTIVE state again (a few
                seconds, usually). If False, it will return right after issuing the
                creation request to the DevOps API, and it will be responsibility
                of the caller to check the database status/namespace availability
                before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the creation request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> my_db_admin.list_namespaces()
            ['default_keyspace']
            >>> my_db_admin.create_namespace("that_other_one")
            {'ok': 1}
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'that_other_one']
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"creating namespace '{name}' on '{self.id}'")
        cn_response = self._astra_db_admin._astra_db_ops.create_keyspace(
            database=self.id,
            keyspace=name,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            f"devops api returned from creating namespace '{name}' on '{self.id}'"
        )
        if cn_response is not None and name == cn_response.get("name"):
            if wait_until_active:
                last_status_seen = STATUS_MAINTENANCE
                while last_status_seen == STATUS_MAINTENANCE:
                    logger.info(f"sleeping to poll for status of '{self.id}'")
                    time.sleep(DATABASE_POLL_NAMESPACE_SLEEP_TIME)
                    last_status_seen = self.info(
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    ).status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database entered unexpected status {last_status_seen} after MAINTENANCE."
                    )
                # is the namespace found?
                if name not in self.list_namespaces():
                    raise DevOpsAPIException("Could not create the namespace.")
            logger.info(f"finished creating namespace '{name}' on '{self.id}'")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful create-namespace DevOps API request for {name}."
            )

    # the 'override' is because the error-recast decorator washes out the signature
    @ops_recast_method_async
    async def async_create_namespace(  # type: ignore[override]
        self,
        name: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a namespace in this database as requested,
        optionally waiting for it to be ready.
        Async version of the method, for use in an asyncio context.

        Args:
            name: the namespace name. If supplying a namespace that exists
                already, the method call proceeds as usual, no errors are
                raised, and the whole invocation is a no-op.
            wait_until_active: if True (default), the method returns only after
                the target database is in ACTIVE state again (a few
                seconds, usually). If False, it will return right after issuing the
                creation request to the DevOps API, and it will be responsibility
                of the caller to check the database status/namespace availability
                before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the creation request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> asyncio.run(
            ...     my_db_admin.async_create_namespace("app_namespace")
            ... )
            {'ok': 1}
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"creating namespace '{name}' on '{self.id}', async")
        cn_response = await self._astra_db_admin._astra_db_ops.async_create_keyspace(
            database=self.id,
            keyspace=name,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            f"devops api returned from creating namespace "
            f"'{name}' on '{self.id}', async"
        )
        if cn_response is not None and name == cn_response.get("name"):
            if wait_until_active:
                last_status_seen = STATUS_MAINTENANCE
                while last_status_seen == STATUS_MAINTENANCE:
                    logger.info(f"sleeping to poll for status of '{self.id}', async")
                    await asyncio.sleep(DATABASE_POLL_NAMESPACE_SLEEP_TIME)
                    last_db_info = await self.async_info(
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    last_status_seen = last_db_info.status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database entered unexpected status {last_status_seen} after MAINTENANCE."
                    )
                # is the namespace found?
                if name not in await self.async_list_namespaces():
                    raise DevOpsAPIException("Could not create the namespace.")
            logger.info(f"finished creating namespace '{name}' on '{self.id}', async")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful create-namespace DevOps API request for {name}."
            )

    @ops_recast_method_sync
    def drop_namespace(
        self,
        name: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delete a namespace from the database, optionally waiting for it
        to become active again.

        Args:
            name: the namespace to delete. If it does not exist in this database,
                an error is raised.
            wait_until_active: if True (default), the method returns only after
                the target database is in ACTIVE state again (a few
                seconds, usually). If False, it will return right after issuing the
                deletion request to the DevOps API, and it will be responsibility
                of the caller to check the database status/namespace availability
                before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'that_other_one']
            >>> my_db_admin.drop_namespace("that_other_one")
            {'ok': 1}
            >>> my_db_admin.list_namespaces()
            ['default_keyspace']
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"dropping namespace '{name}' on '{self.id}'")
        dk_response = self._astra_db_admin._astra_db_ops.delete_keyspace(
            database=self.id,
            keyspace=name,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            f"devops api returned from dropping namespace '{name}' on '{self.id}'"
        )
        if dk_response == name:
            if wait_until_active:
                last_status_seen = STATUS_MAINTENANCE
                while last_status_seen == STATUS_MAINTENANCE:
                    logger.info(f"sleeping to poll for status of '{self.id}'")
                    time.sleep(DATABASE_POLL_NAMESPACE_SLEEP_TIME)
                    last_status_seen = self.info(
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    ).status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database entered unexpected status {last_status_seen} after MAINTENANCE."
                    )
                # is the namespace found?
                if name in self.list_namespaces():
                    raise DevOpsAPIException("Could not drop the namespace.")
            logger.info(f"finished dropping namespace '{name}' on '{self.id}'")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful delete-namespace DevOps API request for {name}."
            )

    # the 'override' is because the error-recast decorator washes out the signature
    @ops_recast_method_async
    async def async_drop_namespace(  # type: ignore[override]
        self,
        name: str,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delete a namespace from the database, optionally waiting for it
        to become active again.
        Async version of the method, for use in an asyncio context.

        Args:
            name: the namespace to delete. If it does not exist in this database,
                an error is raised.
            wait_until_active: if True (default), the method returns only after
                the target database is in ACTIVE state again (a few
                seconds, usually). If False, it will return right after issuing the
                deletion request to the DevOps API, and it will be responsibility
                of the caller to check the database status/namespace availability
                before working with it.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> asyncio.run(
            ...     my_db_admin.async_drop_namespace("app_namespace")
            ... )
            {'ok': 1}
        """

        timeout_manager = MultiCallTimeoutManager(
            overall_max_time_ms=max_time_ms, exception_type="devops_api"
        )
        logger.info(f"dropping namespace '{name}' on '{self.id}', async")
        dk_response = await self._astra_db_admin._astra_db_ops.async_delete_keyspace(
            database=self.id,
            keyspace=name,
            timeout_info=base_timeout_info(max_time_ms),
        )
        logger.info(
            f"devops api returned from dropping namespace "
            f"'{name}' on '{self.id}', async"
        )
        if dk_response == name:
            if wait_until_active:
                last_status_seen = STATUS_MAINTENANCE
                while last_status_seen == STATUS_MAINTENANCE:
                    logger.info(f"sleeping to poll for status of '{self.id}', async")
                    await asyncio.sleep(DATABASE_POLL_NAMESPACE_SLEEP_TIME)
                    last_db_info = await self.async_info(
                        max_time_ms=timeout_manager.remaining_timeout_ms(),
                    )
                    last_status_seen = last_db_info.status
                if last_status_seen != STATUS_ACTIVE:
                    raise DevOpsAPIException(
                        f"Database entered unexpected status {last_status_seen} after MAINTENANCE."
                    )
                # is the namespace found?
                if name in await self.async_list_namespaces():
                    raise DevOpsAPIException("Could not drop the namespace.")
            logger.info(f"finished dropping namespace '{name}' on '{self.id}', async")
            return {"ok": 1}
        else:
            raise DevOpsAPIException(
                f"Could not issue a successful delete-namespace DevOps API request for {name}."
            )

    def drop(
        self,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop this database, i.e. delete it completely and permanently with all its data.

        This method wraps the `drop_database` method of the AstraDBAdmin class,
        where more information may be found.

        Args:
            wait_until_active: if True (default), the method returns only after
                the database has actually been deleted (generally a few minutes).
                If False, it will return right after issuing the
                drop request to the DevOps API, and it will be responsibility
                of the caller to check the database status/availability
                after that, if desired.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> my_db_admin.list_namespaces()
            ['default_keyspace', 'that_other_one']
            >>> my_db_admin.drop()
            {'ok': 1}
            >>> my_db_admin.list_namespaces()  # raises a 404 Not Found http error

        Note:
            Once the method succeeds, methods on this object -- such as `info()`,
            or `list_namespaces()` -- can still be invoked: however, this hardly
            makes sense as the underlying actual database is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased database any further.
        """

        logger.info(f"dropping this database ('{self.id}')")
        return self._astra_db_admin.drop_database(  # type: ignore[no-any-return]
            id=self.id,
            wait_until_active=wait_until_active,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished dropping this database ('{self.id}')")

    async def async_drop(
        self,
        *,
        wait_until_active: bool = True,
        max_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Drop this database, i.e. delete it completely and permanently with all its data.
        Async version of the method, for use in an asyncio context.

        This method wraps the `drop_database` method of the AstraDBAdmin class,
        where more information may be found.

        Args:
            wait_until_active: if True (default), the method returns only after
                the database has actually been deleted (generally a few minutes).
                If False, it will return right after issuing the
                drop request to the DevOps API, and it will be responsibility
                of the caller to check the database status/availability
                after that, if desired.
            max_time_ms: a timeout, in milliseconds, for the whole requested
                operation to complete.
                Note that a timeout is no guarantee that the deletion request
                has not reached the API server.

        Returns:
            A dictionary of the form {"ok": 1} in case of success.
            Otherwise, an exception is raised.

        Example:
            >>> asyncio.run(my_db_admin.async_drop())
            {'ok': 1}

        Note:
            Once the method succeeds, methods on this object -- such as `info()`,
            or `list_namespaces()` -- can still be invoked: however, this hardly
            makes sense as the underlying actual database is no more.
            It is responsibility of the developer to design a correct flow
            which avoids using a deceased database any further.
        """

        logger.info(f"dropping this database ('{self.id}'), async")
        return await self._astra_db_admin.async_drop_database(  # type: ignore[no-any-return]
            id=self.id,
            wait_until_active=wait_until_active,
            max_time_ms=max_time_ms,
        )
        logger.info(f"finished dropping this database ('{self.id}'), async")

    def get_database(
        self,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        region: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> Database:
        """
        Create a Database instance out of this class for working with the data in it.

        Args:
            token: if supplied, is passed to the Database instead of
                the one set for this object. Useful if one wants to work in
                a least-privilege manner, limiting the permissions for non-admin work.
            namespace: an optional namespace to set in the resulting Database.
                The same default logic as for `AstraDBAdmin.get_database` applies.
            region: an optional region for connecting to the database Data API endpoint.
                The same default logic as for `AstraDBAdmin.get_database` applies.
            api_path: path to append to the API Endpoint. In typical usage, this
                should be left to its default of "/api/json".
            api_version: version specifier to append to the API path. In typical
                usage, this should be left to its default of "v1".

        Returns:
            A Database object, ready to be used for working with data and collections.

        Example:
            >>> my_db = my_db_admin.get_database()
            >>> my_db.list_collection_names()
            ['movies', 'another_collection']

        Note:
            creating an instance of Database does not trigger actual creation
            of the database itself, which should exist beforehand. To create databases,
            see the AstraDBAdmin class.
        """

        return self._astra_db_admin.get_database(
            id=self.id,
            token=token,
            namespace=namespace,
            region=region,
            api_path=api_path,
            api_version=api_version,
            max_time_ms=max_time_ms,
        )

    def get_async_database(
        self,
        *,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        region: Optional[str] = None,
        api_path: Optional[str] = None,
        api_version: Optional[str] = None,
        max_time_ms: Optional[int] = None,
    ) -> AsyncDatabase:
        """
        Create an AsyncDatabase instance out of this class for working
        with the data in it.

        This method has identical behavior and signature as the sync
        counterpart `get_database`: please see that one for more details.
        """

        return self.get_database(
            token=token,
            namespace=namespace,
            region=region,
            api_path=api_path,
            api_version=api_version,
            max_time_ms=max_time_ms,
        ).to_async()
