# Fase 6: build nativo de la app. El AAR oficial de onnxruntime-android 1.30.0 NO publica prefab
# (confirmado inspeccionando el AAR extraido: solo AndroidManifest.xml/classes.jar/headers/jni/
# META-INF/proguard.txt/R.txt, sin carpeta prefab/) -- por eso no se linkea via find_package()
# desde una dependencia Gradle, sino contra el .so y headers ya copiados a este arbol de fuentes
# (mismos que se usaron para compilar y verificar synth_cli en el dispositivo real, ver
# decisions.md). Gradle empaqueta jniLibs/arm64-v8a/libonnxruntime.so en el APK por su cuenta.
cmake_minimum_required(VERSION 3.22.1)
project(xtts_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")

add_library(xtts_native SHARED
    jni_bridge.cpp
    xtts_engine.cpp)

target_include_directories(xtts_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_headers")

target_link_libraries(xtts_native
    onnxruntime
    log)
