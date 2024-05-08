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
from typing import Any, Dict, List, Protocol, Union

# A type for:
#     "dict from parsing a JSON from the API responses"
# This is not exactly == the JSON specs,
# (e.g. 'null' is valid JSON), but the JSON API are committed to always
# return JSON objects with a mapping as top-level.
API_RESPONSE = Dict[str, Any]

# The DevOps API has a broader return type in that some calls return
# a list at top level (e.g. "get databases")
OPS_API_RESPONSE = Union[API_RESPONSE, List[Any]]

# A type for:
#     "document stored on the collections"
# Identical to the above in its nature, but preferrably marked as
# 'a disting thing, conceptually, from the returned JSONs'
API_DOC = Dict[str, Any]


# This is for the (partialed, if necessary) functions that can be "paginated".
class PaginableRequestMethod(Protocol):
    def __call__(self, options: Dict[str, Any]) -> API_RESPONSE: ...


# This is for the (partialed, if necessary) async functions that can be "paginated".
class AsyncPaginableRequestMethod(Protocol):
    async def __call__(self, options: Dict[str, Any]) -> API_RESPONSE: ...
