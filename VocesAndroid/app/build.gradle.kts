plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "voces.tcu748"
    // 34 es la plataforma mas nueva instalada localmente (ver decisions.md, Fase 6): el
    // dispositivo real reporta API 36, pero compileSdk/targetSdk no necesitan igualar la
    // version del dispositivo -- solo minSdk <= API del dispositivo.
    compileSdk = 34
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "voces.tcu748"
        // minSdk 26 (Android 8.0): da AudioTrack MODE_STREAM con los overloads de escritura por
        // float/short array que se van a usar para el streaming de audio -- nada en el resto del
        // stack (NDK, ONNX Runtime 1.30.0) exige algo mas nuevo.
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1-fase6"

        ndk {
            // Redmi Note 14 Pro+ 5G (Snapdragon 7s Gen 3) es arm64-v8a puro -- ver Fase 0/1.
            // No se compila para ningun otro ABI: reduce tamano del APK y evita cargar/probar
            // codigo nativo para arquitecturas que este dispositivo nunca va a usar.
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    packaging {
        // libonnxruntime.so ya esta en jniLibs/arm64-v8a/; nada mas que empaquetar de fuentes
        // externas (no se usa la dependencia Maven de onnxruntime-android -- ver CMakeLists.txt).
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}
