/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Concat.cpp - Lowering Concat Op -------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXFlattenOp(A) is mainly implemented using MHLO reshapeOp
struct ONNXFlattenOpLoweringToMhlo : public ConversionPattern {
  ONNXFlattenOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXFlattenOpAdaptor operandAdaptor(operands);
    ONNXFlattenOp flattenOp = llvm::cast<ONNXFlattenOp>(op);

    Value input = operandAdaptor.input();
    assert(isRankedShapedType(input.getType()) && "Expected Ranked ShapedType");
    ShapedType inputType = input.getType().cast<ShapedType>();
    int64_t rank = inputType.getRank();
    int64_t axis = flattenOp.axis();
    assert(axis >= -rank && axis <= rank - 1);
    axis = axis >= 0 ? axis : rank + axis;

    Value flattenDimFirst = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, input);
    for (int64_t i = 0; i < axis; i++) {
      Value dim = rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
      flattenDimFirst =
          rewriter.create<shape::MulOp>(loc, flattenDimFirst, dim);
    }
    Value flattenDimSecond = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (int64_t i = axis; i < rank; i++) {
      Value dim = rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
      flattenDimSecond =
          rewriter.create<shape::MulOp>(loc, flattenDimSecond, dim);
    }
    SmallVector<Value> dims{flattenDimFirst, flattenDimSecond};
    Type outputShapeType = RankedTensorType::get({2}, rewriter.getIndexType());
    Value outputShape = rewriter.create<shape::FromExtentsOp>(loc, dims);
    outputShape = rewriter.create<shape::ToExtentTensorOp>(
        loc, outputShapeType, outputShape);
    auto result = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, *op->result_type_begin(), input, outputShape);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLoweringToMhlo>(ctx);
}

} // namespace onnx_mlir
