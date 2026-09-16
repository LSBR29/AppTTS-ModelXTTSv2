package voces.tcu748

/**
 * Puente delgado a xtts_engine (C++, ya verificado byte-identico contra la version CLI que
 * primero produjo audio real en el dispositivo). No hay logica aca, solo las declaraciones
 * `external fun` que JNI resuelve contra jni_bridge.cpp.
 */
object NativeBridge {
    init {
        System.loadLibrary("xtts_native")
    }

    /**
     * threadsPrefill/threadsHifigan=4, threadsDecode=1: medido en el dispositivo real -- el
     * bucle de decode es memory-bound y EMPEORA con mas threads (143ms/paso a 1 thread vs
     * 272ms/paso a 4), mientras que prefill e HiFiGAN si se benefician de mas threads.
     */
    external fun nativeInit(modelDir: String, threadsPrefill: Int, threadsDecode: Int, threadsHifigan: Int): Long

    external fun nativeSplitSentences(text: String): Array<String>

    /**
     * temperature/topP/topK/repetitionPenalty: parametros de muestreo del GPT (ver pantalla de
     * ajustes en MainActivity.kt) -- valores de producción: 0.85/0.85/50/2.0.
     * out_info[0]=n_audio_tokens, out_info[1]=1 si paro en stop token real (0 si tope de seguridad).
     */
    external fun nativeSynthSentence(
        handle: Long, text: String, seed: Int,
        temperature: Float, topP: Float, topK: Int, repetitionPenalty: Float,
        outInfo: IntArray,
    ): FloatArray

    /** Libera la sesion de decode (~640MB, la mas pesada) -- llamar al pasar a segundo plano. */
    external fun nativeReleaseGpt(handle: Long)

    external fun nativeDestroy(handle: Long)
}
