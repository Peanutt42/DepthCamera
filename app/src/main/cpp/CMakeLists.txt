cmake_minimum_required(VERSION 3.10)

project(NativeLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LITERT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/include/external/org_tensorflow"
)
set(LITERT_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-gpu-1.2.0/include")
set(LITERT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/lib/${ANDROID_ABI}/libtensorflowlite_jni.so")
set(LITERT_GPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-gpu-1.2.0/lib/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so")


add_library(
    NativeLib
    SHARED
    NativeLib.cpp
    Log.hpp
    NativeJavaScopes.hpp
    PerformanceScope.hpp
    ImageUtils.hpp
    ImageUtils.cpp
    TfLiteRuntime.hpp
    TfLiteRuntime.cpp
    DepthEstimation.hpp
    DepthEstimation.cpp
)

target_compile_features(NativeLib PUBLIC cxx_std_20)

target_include_directories(NativeLib PUBLIC
    ${LITERT_INCLUDE_DIRS}
    ${LITERT_GPU_INCLUDE_DIR}
	
	# just so that vscode knows where to look for headers
	"${CMAKE_SYSROOT}/usr/include/"
	"${CMAKE_SYSROOT}/usr/include/c++/v1"
	"${CMAKE_SYSROOT}/usr/include/${ANDROID_TOOLCHAIN_NAME}"
)

target_link_libraries(
    NativeLib
    android
    log
    jnigraphics
    ${LITERT_LIB}
    ${LITERT_GPU_LIB}
)
