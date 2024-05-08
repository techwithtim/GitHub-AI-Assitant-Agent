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
import httpx
from typing import Any, Dict, Optional, Union, cast

from astrapy.core.core_types import API_RESPONSE
from astrapy.core.utils import amake_request, make_request

logger = logging.getLogger(__name__)


class APIRequestError(ValueError):
    def __init__(
        self, response: httpx.Response, payload: Optional[Dict[str, Any]]
    ) -> None:
        super().__init__(response.text)

        self.response = response
        self.payload = payload

    def __repr__(self) -> str:
        return f"{self.response}"


def raw_api_request(
    client: httpx.Client,
    base_url: str,
    auth_header: str,
    token: str,
    method: str,
    json_data: Optional[Dict[str, Any]],
    url_params: Optional[Dict[str, Any]],
    path: Optional[str],
    caller_name: Optional[str],
    caller_version: Optional[str],
    timeout: Optional[Union[httpx.Timeout, float]],
) -> httpx.Response:
    return make_request(
        client=client,
        base_url=base_url,
        auth_header=auth_header,
        token=token,
        method=method,
        json_data=json_data,
        url_params=url_params,
        path=path,
        caller_name=caller_name,
        caller_version=caller_version,
        timeout=timeout,
    )


def process_raw_api_response(
    raw_response: httpx.Response,
    skip_error_check: bool,
    json_data: Optional[Dict[str, Any]],
) -> API_RESPONSE:
    # In case of other successful responses, parse the JSON body.
    try:
        # Cast the response to the expected type.
        response_body: API_RESPONSE = cast(API_RESPONSE, raw_response.json())

        # If the API produced an error, warn and raise it as an Exception
        if "errors" in response_body and not skip_error_check:
            logger.debug(response_body["errors"])

            raise APIRequestError(raw_response, payload=json_data)

        # Otherwise, set the response body
        return response_body
    except ValueError:
        # Handle cases where json() parsing fails (e.g., empty body)
        raise APIRequestError(raw_response, payload=json_data)


def api_request(
    client: httpx.Client,
    base_url: str,
    auth_header: str,
    token: str,
    method: str,
    json_data: Optional[Dict[str, Any]],
    url_params: Optional[Dict[str, Any]],
    path: Optional[str],
    skip_error_check: bool,
    caller_name: Optional[str],
    caller_version: Optional[str],
    timeout: Optional[Union[httpx.Timeout, float]],
) -> API_RESPONSE:
    raw_response = raw_api_request(
        client=client,
        base_url=base_url,
        auth_header=auth_header,
        token=token,
        method=method,
        json_data=json_data,
        url_params=url_params,
        path=path,
        caller_name=caller_name,
        caller_version=caller_version,
        timeout=timeout,
    )
    raw_response.raise_for_status()
    return process_raw_api_response(
        raw_response, skip_error_check=skip_error_check, json_data=json_data
    )


###
async def async_raw_api_request(
    client: httpx.AsyncClient,
    base_url: str,
    auth_header: str,
    token: str,
    method: str,
    json_data: Optional[Dict[str, Any]],
    url_params: Optional[Dict[str, Any]],
    path: Optional[str],
    caller_name: Optional[str],
    caller_version: Optional[str],
    timeout: Optional[Union[httpx.Timeout, float]],
) -> httpx.Response:
    return await amake_request(
        client=client,
        base_url=base_url,
        auth_header=auth_header,
        token=token,
        method=method,
        json_data=json_data,
        url_params=url_params,
        path=path,
        caller_name=caller_name,
        caller_version=caller_version,
        timeout=timeout,
    )


async def async_process_raw_api_response(
    raw_response: httpx.Response,
    skip_error_check: bool,
    json_data: Optional[Dict[str, Any]],
) -> API_RESPONSE:
    # In case of other successful responses, parse the JSON body.
    try:
        # Cast the response to the expected type.
        response_body: API_RESPONSE = cast(API_RESPONSE, raw_response.json())

        # If the API produced an error, warn and return the API request error class
        if "errors" in response_body and not skip_error_check:
            logger.debug(response_body["errors"])

            raise APIRequestError(raw_response, payload=json_data)

        # Otherwise, set the response body
        return response_body
    except ValueError:
        # Handle cases where json() parsing fails (e.g., empty body)
        raise APIRequestError(raw_response, payload=json_data)


async def async_api_request(
    client: httpx.AsyncClient,
    base_url: str,
    auth_header: str,
    token: str,
    method: str,
    json_data: Optional[Dict[str, Any]],
    url_params: Optional[Dict[str, Any]],
    path: Optional[str],
    skip_error_check: bool,
    caller_name: Optional[str],
    caller_version: Optional[str],
    timeout: Optional[Union[httpx.Timeout, float]],
) -> API_RESPONSE:
    raw_response = await async_raw_api_request(
        client=client,
        base_url=base_url,
        auth_header=auth_header,
        token=token,
        method=method,
        json_data=json_data,
        url_params=url_params,
        path=path,
        caller_name=caller_name,
        caller_version=caller_version,
        timeout=timeout,
    )
    raw_response.raise_for_status()
    return await async_process_raw_api_response(
        raw_response,
        skip_error_check=skip_error_check,
        json_data=json_data,
    )
