export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
MLIR_DIR=$(pwd)/build-llvm/lib/cmake/mlir
mkdir -p build-onnx-mlir 
cd build-onnx-mlir

if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi

bash ./utils/install-protobuf.sh

cmake --build . -j 2
