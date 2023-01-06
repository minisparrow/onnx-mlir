mkdir -p build-llvm 

# build llvm and mlir 
cmake -G Ninja -S third_party/llvm-project/llvm -B build-llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON
cmake --build build-llvm/ -- ${MAKEFLAGS} -j 2
