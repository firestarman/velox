/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

namespace {

using namespace facebook::velox;

bool isSupportedWindowType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::TIMESTAMP:
    case TypeKind::DATE:
    case TypeKind::SHORT_DECIMAL:
    case TypeKind::LONG_DECIMAL:
      return true;
    case TypeKind::ROW:
      for (auto i = 0; i < type->size(); ++i) {
        if (!isSupportedWindowType(type->childAt(i))) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

bool isDefaultRankFrame(const core::WindowNode::Frame& frame) {
  return frame.startType == core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kCurrentRow &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isFieldAccessExpr(const core::TypedExprPtr& expr) {
  return std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr) !=
      nullptr;
}

bool isSupportedRankLikeWindowFunction(const core::WindowNode::Function& function) {
  const auto& functionName = function.functionCall->name();
  if ((functionName != "row_number" && functionName != "rank" &&
       functionName != "dense_rank") ||
      function.ignoreNulls) {
    return false;
  }

  if (!function.functionCall->inputs().empty()) {
    return false;
  }

  if (!isDefaultRankFrame(function.frame)) {
    return false;
  }

  auto const resultKind = function.functionCall->type()->kind();
  if (resultKind != TypeKind::INTEGER && resultKind != TypeKind::BIGINT) {
    return false;
  }

  return true;
}

std::vector<cudf::column_view> selectColumns(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& channels) {
  std::vector<cudf::column_view> columns;
  columns.reserve(channels.size());
  for (auto channel : channels) {
    columns.push_back(table.column(channel));
  }
  return columns;
}

bool isAscending(core::SortOrder const& sortOrder) {
  return sortOrder.isAscending();
}

cudf::null_order toCudfNullOrder(core::SortOrder const& sortOrder) {
  return (sortOrder.isNullsFirst() ^ !sortOrder.isAscending())
      ? cudf::null_order::BEFORE
      : cudf::null_order::AFTER;
}

cudf::rank_method toRankMethod(
    std::shared_ptr<const core::WindowNode> const& windowNode) {
  VELOX_CHECK_EQ(
      windowNode->windowFunctions().size(),
      1,
      "CudfWindow supports one rank-like window function");
  auto const& functionName = windowNode->windowFunctions()[0].functionCall->name();
  if (functionName == "row_number") {
    return cudf::rank_method::FIRST;
  }
  if (functionName == "dense_rank") {
    return cudf::rank_method::DENSE;
  }
  VELOX_CHECK_EQ(functionName, "rank");
  return cudf::rank_method::MIN;
}

} // namespace

namespace facebook::velox::cudf_velox {

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node) {
  if (!node || !isSupportedWindowType(node->inputType()) ||
      !isSupportedWindowType(node->outputType())) {
    return false;
  }

  const auto& windowFunctions = node->windowFunctions();
  if (windowFunctions.size() != 1 ||
      !isSupportedRankLikeWindowFunction(windowFunctions[0])) {
    return false;
  }

  if (node->sortingKeys().size() != 1 || node->sortingOrders().size() != 1) {
    return false;
  }

  if (!isFieldAccessExpr(node->sortingKeys()[0]) ||
      !isSupportedWindowType(node->sortingKeys()[0]->type())) {
    return false;
  }

  for (const auto& partitionKey : node->partitionKeys()) {
    if (!isFieldAccessExpr(partitionKey) ||
        !isSupportedWindowType(partitionKey->type())) {
      return false;
    }
  }

  return true;
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : exec::Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "CudfWindow"),
      NvtxHelper(
          nvtx3::rgb{123, 104, 238}, // MediumSlateBlue
          operatorId,
          fmt::format("[{}]", windowNode->id())),
      windowNode_(windowNode),
      inputType_(windowNode->sources()[0]->outputType()),
      partitionKeyChannels_([&]() {
        std::vector<cudf::size_type> channels;
        channels.reserve(windowNode->partitionKeys().size());
        for (auto const& key : windowNode->partitionKeys()) {
          auto const channel = exec::exprToChannel(key.get(), inputType_);
          VELOX_CHECK_NE(
              channel, kConstantChannel, "Window partition keys must be columns");
          channels.push_back(channel);
        }
        return channels;
      }()),
      sortKeyChannel_([&]() {
        auto const channel =
            exec::exprToChannel(windowNode->sortingKeys()[0].get(), inputType_);
        VELOX_CHECK_NE(
            channel, kConstantChannel, "Window sorting key must be a column");
        return static_cast<cudf::size_type>(channel);
      }()),
      sortOrder_(
          isAscending(windowNode->sortingOrders()[0]) ? cudf::order::ASCENDING
                                                      : cudf::order::DESCENDING),
      nullOrder_(toCudfNullOrder(windowNode->sortingOrders()[0])),
      rankMethod_(toRankMethod(windowNode)) {
  VELOX_CHECK_EQ(
      windowNode->windowFunctions().size(),
      1,
      "CudfWindow supports one rank-like window function");
}

void CudfWindow::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

void CudfWindow::noMoreInput() {
  exec::Operator::noMoreInput();
}

RowVectorPtr CudfWindow::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (!output_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto inputTable = getConcatenatedTable(inputs_, inputType_, stream);
    inputs_.clear();

    auto outputTable = computeOutputTable(std::move(inputTable), stream);
    output_ = std::make_shared<CudfVector>(
        pool(),
        outputType_,
        outputTable->num_rows(),
        std::move(outputTable),
        stream);
  }

  finished_ = true;
  return output_;
}

void CudfWindow::close() {
  exec::Operator::close();
  inputs_.clear();
  output_.reset();
}

std::unique_ptr<cudf::column> CudfWindow::computeRankLikeColumn(
    cudf::table_view const& sortedInput,
    rmm::cuda_stream_view stream) const {
  auto const mr = cudf::get_current_device_resource_ref();

  if (partitionKeyChannels_.empty()) {
    auto aggregation = cudf::make_rank_aggregation<cudf::scan_aggregation>(
        rankMethod_, sortOrder_, cudf::null_policy::INCLUDE, nullOrder_);
    return cudf::scan(
        sortedInput.column(sortKeyChannel_),
        *aggregation,
        cudf::scan_type::INCLUSIVE,
        cudf::null_policy::INCLUDE,
        stream,
        mr);
  }

  auto partitionColumns = selectColumns(sortedInput, partitionKeyChannels_);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = sortedInput.column(sortKeyChannel_);
  requests[0].aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
          rankMethod_, sortOrder_, cudf::null_policy::INCLUDE, nullOrder_));

  cudf::groupby::groupby grouper(
      cudf::table_view(partitionColumns),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyChannels_.size(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyChannels_.size(), cudf::null_order::BEFORE));
  auto scanResult = grouper.scan(requests, stream, mr);
  auto& aggregationResults = scanResult.second;
  VELOX_CHECK_EQ(aggregationResults.size(), 1);
  VELOX_CHECK_EQ(aggregationResults[0].results.size(), 1);
  return std::move(aggregationResults[0].results[0]);
}

std::unique_ptr<cudf::table> CudfWindow::computeOutputTable(
    std::unique_ptr<cudf::table> input,
    rmm::cuda_stream_view stream) const {
  auto const mr = cudf::get_current_device_resource_ref();
  auto const rankType = cudf::data_type(
      cudf_velox::veloxToCudfTypeId(outputType_->childAt(outputType_->size() - 1)));

  if (input->num_rows() == 0) {
    auto columns = input->release();
    columns.push_back(cudf::make_empty_column(rankType));
    return std::make_unique<cudf::table>(std::move(columns));
  }

  auto const inputView = input->view();
  if (windowNode_->inputsSorted()) {
    auto rankColumn = computeRankLikeColumn(inputView, stream);
    if (rankColumn->type() != rankType) {
      rankColumn = cudf::cast(rankColumn->view(), rankType, stream);
    }

    auto columns = input->release();
    columns.push_back(std::move(rankColumn));
    return std::make_unique<cudf::table>(std::move(columns));
  }

  auto sortKeyViews = selectColumns(inputView, partitionKeyChannels_);
  sortKeyViews.push_back(inputView.column(sortKeyChannel_));

  std::vector<cudf::order> columnOrder(partitionKeyChannels_.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> keyNullOrder(
      partitionKeyChannels_.size(), cudf::null_order::BEFORE);
  columnOrder.push_back(sortOrder_);
  keyNullOrder.push_back(nullOrder_);

  auto sortedOrder =
      cudf::stable_sorted_order(cudf::table_view(sortKeyViews), columnOrder, keyNullOrder, stream, mr);
  auto sortedTable =
      cudf::gather(inputView, sortedOrder->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  auto rankColumn = computeRankLikeColumn(sortedTable->view(), stream);

  if (rankColumn->type() != rankType) {
    rankColumn = cudf::cast(rankColumn->view(), rankType, stream);
  }

  auto scatteredRankTarget =
      cudf::allocate_like(rankColumn->view(), rankColumn->size(), cudf::mask_allocation_policy::RETAIN, stream, mr);
  auto scatteredRankTable = cudf::scatter(
      cudf::table_view{{rankColumn->view()}},
      sortedOrder->view(),
      cudf::table_view{{scatteredRankTarget->view()}},
      stream,
      mr);

  auto rankColumnsOut = scatteredRankTable->release();
  auto columns = input->release();
  columns.push_back(std::move(rankColumnsOut[0]));
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace facebook::velox::cudf_velox
