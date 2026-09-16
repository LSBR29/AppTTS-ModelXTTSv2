// Fase 6: puente JNI entre Kotlin (voces.tcu748.NativeBridge) y xtts_engine (motor ya
// verificado byte-identico contra la version CLI monolitica que primero produjo audio real en el
// dispositivo -- ver .claude-work/state/decisions.md). No tiene logica de sintesis propia, solo
// convierte tipos JNI <-> C++ y administra el puntero opaco (jlong) al motor.
#include <jni.h>

#include <memory>
#include <string>

#include "sentence_split.hpp"
#include "xtts_engine.hpp"

extern "C" {

JNIEXPORT jlong JNICALL
Java_voces_tcu748_NativeBridge_nativeInit(
    JNIEnv* env, jobject, jstring model_dir, jint threads_prefill, jint threads_decode, jint threads_hifigan) {
  const char* dir_chars = env->GetStringUTFChars(model_dir, nullptr);
  std::string dir(dir_chars);
  env->ReleaseStringUTFChars(model_dir, dir_chars);
  try {
    auto* engine = new xtts_engine::XttsEngine(dir, threads_prefill, threads_decode, threads_hifigan);
    return reinterpret_cast<jlong>(engine);
  } catch (const std::exception& e) {
    jclass ex_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(ex_class, e.what());
    return 0;
  }
}

JNIEXPORT jobjectArray JNICALL
Java_voces_tcu748_NativeBridge_nativeSplitSentences(JNIEnv* env, jobject, jstring text) {
  const char* t_chars = env->GetStringUTFChars(text, nullptr);
  // char_limit mas chico que el default (239, el tope duro del tokenizador real): trocitos mas
  // cortos generan mas rapido cada uno -- primer audio antes, y huecos de silencio entre
  // fragmentos mas cortos cuando ocurren (RTF>1 sigue significando que tarde o temprano la
  // reproduccion alcanza a la generacion, pero con fragmentos chicos el hueco individual es
  // chico en vez de la oracion entera). Reutiliza el mismo separador ya existente (punto/coma/
  // espacio) -- SplitSentences() ya partia oraciones largas asi, esto solo hace que aplique mas
  // seguido. Ver MainActivity.synthesizeAndPlay() para el otro lado del ajuste (buffer de 2
  // fragmentos antes de empezar a reproducir).
  std::vector<std::string> sentences = xtts_tokenizer::SplitSentences(std::string(t_chars), 100);
  env->ReleaseStringUTFChars(text, t_chars);

  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(sentences.size()), string_class, nullptr);
  for (size_t i = 0; i < sentences.size(); ++i) {
    jstring s = env->NewStringUTF(sentences[i].c_str());
    env->SetObjectArrayElement(result, static_cast<jsize>(i), s);
    env->DeleteLocalRef(s);
  }
  return result;
}

// Devuelve las muestras de audio (float32, 24kHz mono, [-1,1]) de una oracion. El booleano
// "parado en stop token real" (vs. tope de seguridad) se pasa via un jintArray de 2 elementos
// [n_audio_tokens, stopped_naturally] -- evita crear una clase Kotlin extra para un resultado
// tan chico, y `out_info` la llena directamente (idioma comun en JNI para "out parameters").
JNIEXPORT jfloatArray JNICALL
Java_voces_tcu748_NativeBridge_nativeSynthSentence(
    JNIEnv* env, jobject, jlong handle, jstring text, jint seed,
    jfloat temperature, jfloat top_p, jint top_k, jfloat repetition_penalty, jintArray out_info) {
  auto* engine = reinterpret_cast<xtts_engine::XttsEngine*>(handle);
  const char* t_chars = env->GetStringUTFChars(text, nullptr);
  std::string t(t_chars);
  env->ReleaseStringUTFChars(text, t_chars);

  xtts_engine::SamplingParams params;
  params.temperature = temperature;
  params.top_p = top_p;
  params.top_k = top_k;
  params.repetition_penalty = repetition_penalty;

  try {
    xtts_engine::SynthResult result = engine->SynthSentence(t, static_cast<unsigned int>(seed), params);

    if (out_info != nullptr) {
      jint info[2] = {result.n_audio_tokens, result.stopped_naturally ? 1 : 0};
      env->SetIntArrayRegion(out_info, 0, 2, info);
    }

    jfloatArray arr = env->NewFloatArray(static_cast<jsize>(result.wav.size()));
    env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(result.wav.size()), result.wav.data());
    return arr;
  } catch (const std::exception& e) {
    jclass ex_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(ex_class, e.what());
    return nullptr;
  }
}

JNIEXPORT void JNICALL
Java_voces_tcu748_NativeBridge_nativeReleaseGpt(JNIEnv*, jobject, jlong handle) {
  if (handle != 0) reinterpret_cast<xtts_engine::XttsEngine*>(handle)->ReleaseGpt();
}

JNIEXPORT void JNICALL
Java_voces_tcu748_NativeBridge_nativeDestroy(JNIEnv*, jobject, jlong handle) {
  delete reinterpret_cast<xtts_engine::XttsEngine*>(handle);
}

}  // extern "C"
