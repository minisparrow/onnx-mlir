/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- SpaceToDepth.cpp - Lowering SpaceToDepthOp ----------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SpaceToDepth Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "llvm/Support/Debug.h"

using namespace mlir;
using llvm::dbgs;

#define DEBUG_TYPE "space_to_depth_onnx_to_krnl"

class ONNXSpaceToDepthOpLowering : public ConversionPattern {
public:
  ONNXSpaceToDepthOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXSpaceToDepthOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto spaceToDepthOp = dyn_cast_or_null<ONNXSpaceToDepthOp>(op);
    assert(spaceToDepthOp && "Expecting op to have type ONNXSpaceToDepthOp");

    // Ensure we can compute the operator output shape.
    ONNXSpaceToDepthOpShapeHelper shapeHelper(&spaceToDepthOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    ONNXSpaceToDepthOpAdaptor operandAdaptor(operands);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    Value input = spaceToDepthOp.input();
    int64_t bs = spaceToDepthOp.blocksize();
    Location loc = spaceToDepthOp.getLoc();

    // Compute the new dimensions.
    MemRefBoundsIndexCapture bounds(input);
    assert(bounds.getRank() == 4 && "Input tensor should have rank equal to 4");

    DimIndexExpr B(bounds.getDim(0));
    DimIndexExpr C(bounds.getDim(1));
    DimIndexExpr H(bounds.getDim(2));
    DimIndexExpr W(bounds.getDim(3));
    DimIndexExpr newC = create<MulIOp>(C, (bs * bs), rewriter, loc);
    DimIndexExpr newH = create<SignedFloorDivIOp>(H, bs, rewriter, loc);
    DimIndexExpr newW = create<SignedFloorDivIOp>(W, bs, rewriter, loc);

    // Reshape input tensor to shape [B, C, H/bs, bs, W/bs, bs].
    LiteralIndexExpr bsLit(bs);
    DimsExpr outputDims1({B, C, newH, bsLit, newW, bsLit});
    Value reshapeRes1 = reshape(input, outputDims1, rewriter, loc);
    LLVM_DEBUG(dbgs() << "reshapeRes1: " << reshapeRes1 << "\n");

    // Transpose the reshape result into shape [B, C, bs, bs, H/bs, W/bs].
    DimsExpr outputDims2({B, C, bsLit, bsLit, newH, newW});
    ArrayRef<int64_t> perm = {0, 1, 3, 5, 2, 4};
    Value transposeRes = transpose(
        reshapeRes1, outputDims2, perm, rewriter, spaceToDepthOp, shapeHelper);
    LLVM_DEBUG(dbgs() << "transposeRes: " << transposeRes << "\n");

    // Reshape the transpose result into shape [B, C*bs*bs, H/bs, W/bs].
    DimsExpr outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = reshape(transposeRes, outputDims3, rewriter, loc);
    LLVM_DEBUG(dbgs() << "reshapeRes2: " << reshapeRes2 << "\n");

    rewriter.replaceOp(op, reshapeRes2);

    return success();
  }

private:
  template <typename Op>
  DimIndexExpr create(const DimIndexExpr &indexExpr,
      const LiteralIndexExpr &constant, ConversionPatternRewriter &rewriter,
      const Location &loc) const {
    static_assert(Op::template hasTrait<OpTrait::NOperands<2>::Impl>(),
        "expected binary operation");

    if (!indexExpr.isLiteral())
      return DimIndexExpr(
          rewriter.create<Op>(loc, indexExpr.getValue(), constant.getValue()));

    if (std::is_same<Op, SignedFloorDivIOp>::value)
      return LiteralIndexExpr(indexExpr.getLiteral() / constant.getLiteral());
    else if (std::is_same<Op, MulIOp>::value)
      return LiteralIndexExpr(indexExpr.getLiteral() * constant.getLiteral());

    llvm_unreachable("Unexpected operator type");
  }

  // Reshape the 'input' tensor to the shape prodided by 'outputDims'.
  Value reshape(const Value &input, const DimsExpr &outputDims,
      ConversionPatternRewriter &rewriter, const Location &loc) const {
    assert(!outputDims.empty() && "Output dimensions should not be empty");

    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : outputDims)
      shape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    MemRefType memRefType = MemRefType::get(shape, elementType);

    return emitMemRefReinterpretCastOp(
        rewriter, loc, input, memRefType, outputDims);
  }

  // Transpose the 'input' tensor.
  Value transpose(Value &input, const DimsExpr &outputDims,
      const ArrayRef<int64_t> &perm, ConversionPatternRewriter &rewriter,
      ONNXSpaceToDepthOp &op,
      ONNXSpaceToDepthOpShapeHelper &shapeHelper) const {
    assert(!outputDims.empty() && "Output dimensions should not be empty");
    assert(!perm.empty() && perm.size() == outputDims.size() &&
           "Expecitng valid permutation array");

    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : outputDims)
      shape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    MemRefType transposeMemRefType = MemRefType::get(shape, elementType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, transposeMemRefType, op.getLoc(), outputDims);

    // Create loop.
    int64_t rank = transposeMemRefType.getShape().size();
    BuildKrnlLoop inputLoops(rewriter, op.getLoc(), rank);
    inputLoops.createDefineAndIterateOp(input);

    rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());
    {
      // Get a child IndexExpr context.
      IndexExprScope childScope(&rewriter, shapeHelper.scope);
      KrnlBuilder createKrnl(rewriter, op.getLoc());

      // Get read/write indices.
      SmallVector<IndexExpr, 4> readIndices;
      SmallVector<IndexExpr, 4> writeIndices;
      for (int64_t i = 0; i < rank; ++i) {
        Value readVal = inputLoops.getInductionVar(i);
        Value writeVal = inputLoops.getInductionVar(perm[i]);
        readIndices.emplace_back(DimIndexExpr(readVal));
        writeIndices.emplace_back(DimIndexExpr(writeVal));
      }

      // Copy data.
      Value loadData = createKrnl.loadIE(input, readIndices);
      createKrnl.storeIE(loadData, alloc, writeIndices);
    }

    rewriter.setInsertionPoint(op);

    return alloc;
  }
};

void populateLoweringONNXSpaceToDepthOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSpaceToDepthOpLowering>(ctx);
}
