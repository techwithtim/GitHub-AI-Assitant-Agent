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
from typing import Any, Dict, List, Optional


@dataclass
class OperationResult(ABC):
    """
    Class that represents the generic result of a single mutation operation.

    Attributes:
        raw_results: response/responses from the Data API call.
            Depending on the exact delete method being used, this
            list of raw responses can contain exactly one or a number of items.
    """

    raw_results: List[Dict[str, Any]]

    @abstractmethod
    def to_bulk_write_result(self, index_in_bulk_write: int) -> BulkWriteResult: ...


@dataclass
class DeleteResult(OperationResult):
    """
    Class that represents the result of delete operations.

    Attributes:
        deleted_count: number of deleted documents
        raw_results: response/responses from the Data API call.
            Depending on the exact delete method being used, this
            list of raw responses can contain exactly one or a number of items.
    """

    deleted_count: Optional[int]

    def to_bulk_write_result(self, index_in_bulk_write: int) -> BulkWriteResult:
        return BulkWriteResult(
            bulk_api_results={index_in_bulk_write: self.raw_results},
            deleted_count=self.deleted_count,
            inserted_count=0,
            matched_count=0,
            modified_count=0,
            upserted_count=0,
            upserted_ids={},
        )


@dataclass
class InsertOneResult(OperationResult):
    """
    Class that represents the result of insert_one operations.

    Attributes:
        raw_results: one-item list with the response from the Data API call
        inserted_id: the ID of the inserted document
    """

    inserted_id: Any

    def to_bulk_write_result(self, index_in_bulk_write: int) -> BulkWriteResult:
        return BulkWriteResult(
            bulk_api_results={index_in_bulk_write: self.raw_results},
            deleted_count=0,
            inserted_count=1,
            matched_count=0,
            modified_count=0,
            upserted_count=0,
            upserted_ids={},
        )


@dataclass
class InsertManyResult(OperationResult):
    """
    Class that represents the result of insert_many operations.

    Attributes:
        raw_results: responses from the Data API calls
        inserted_ids: list of the IDs of the inserted documents
    """

    inserted_ids: List[Any]

    def to_bulk_write_result(self, index_in_bulk_write: int) -> BulkWriteResult:
        return BulkWriteResult(
            bulk_api_results={index_in_bulk_write: self.raw_results},
            deleted_count=0,
            inserted_count=len(self.inserted_ids),
            matched_count=0,
            modified_count=0,
            upserted_count=0,
            upserted_ids={},
        )


@dataclass
class UpdateResult(OperationResult):
    """
    Class that represents the result of any update operation.

    Attributes:
        raw_results: responses from the Data API calls
        update_info: a dictionary reporting about the update

    Note:
        the "update_info" field has the following fields: "n" (int),
        "updatedExisting" (bool), "ok" (float), "nModified" (int)
        and optionally "upserted" containing the ID of an upserted document.

    """

    update_info: Dict[str, Any]

    def to_bulk_write_result(self, index_in_bulk_write: int) -> BulkWriteResult:
        inserted_count = 1 if "upserted" in self.update_info else 0
        matched_count = (self.update_info.get("n") or 0) - inserted_count
        if "upserted" in self.update_info:
            upserted_ids = {index_in_bulk_write: self.update_info["upserted"]}
        else:
            upserted_ids = {}
        return BulkWriteResult(
            bulk_api_results={index_in_bulk_write: self.raw_results},
            deleted_count=0,
            inserted_count=inserted_count,
            matched_count=matched_count,
            modified_count=self.update_info.get("nModified") or 0,
            upserted_count=1 if "upserted" in self.update_info else 0,
            upserted_ids=upserted_ids,
        )


@dataclass
class BulkWriteResult:
    """
    Class that represents the result of a bulk write operations.

    Indices in the maps below refer to the position of each write operation
    in the list of operations passed to the bulk_write command.

    The numeric counts refer to the whole of the bulk write.

    Attributes:
        bulk_api_results: a map from indices to the corresponding raw responses
        deleted_count: number of deleted documents
        inserted_count: number of inserted documents
        matched_count: number of matched documents
        modified_count: number of modified documents
        upserted_count: number of upserted documents
        upserted_ids: a (sparse) map from indices to ID of the upserted document
    """

    bulk_api_results: Dict[int, List[Dict[str, Any]]]
    deleted_count: Optional[int]
    inserted_count: int
    matched_count: int
    modified_count: int
    upserted_count: int
    upserted_ids: Dict[int, Any]

    @staticmethod
    def zero() -> BulkWriteResult:
        """
        Return an empty BulkWriteResult, for use in no-ops and list reductions.
        """

        return BulkWriteResult(
            bulk_api_results={},
            deleted_count=0,
            inserted_count=0,
            matched_count=0,
            modified_count=0,
            upserted_count=0,
            upserted_ids={},
        )
