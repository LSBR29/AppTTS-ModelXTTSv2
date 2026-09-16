// Fase 6: motor de sintesis reutilizable, con sesiones ONNX persistentes -- extraido de
// synth_cli.cpp (que ya funciono de punta a punta en el dispositivo real, ver decisions.md) para
// que una app real (via JNI) pueda:
//   - crear el motor UNA vez (carga tokenizador, latentes de condicionamiento, sesion de
//     HiFiGAN, y NO crea la sesion de decode todavia -- se crea perezosamente en el primer
//     SynthSentence()),
//   - sintetizar muchas oraciones sin recargar el tokenizador/latentes/HiFiGAN en cada una
//     (el prefill SI se crea/corre/destruye por oracion, tal como se midio que da RSS pico
//     1320MB en vez de 2500MB -- ver decisions.md, entrada de RAM),
//   - liberar la sesion de decode (la mas pesada, ~1.27GB) cuando la app pasa a segundo plano,
//     sin perder tokenizador/latentes/HiFiGAN, y recrearla la proxima vez que haga falta.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace xtts_engine {

struct SynthResult {
  std::vector<float> wav;       // 24kHz mono, [-1,1] aprox
  int n_audio_tokens = 0;
  bool stopped_naturally = false;  // true si paro en stop_audio_token, false si tope de seguridad
};

// Valores por defecto = los de produccion (ver synth_cli.cpp, citas de infer_repl.py/gpt.py).
// Ajustables desde la app (ver pantalla de ajustes en MainActivity.kt).
struct SamplingParams {
  float temperature = 0.85f;
  float top_p = 0.85f;
  int top_k = 50;
  float repetition_penalty = 2.0f;
};

class XttsEngine {
 public:
  // threads_prefill/decode/hifigan: ver decisions.md, entrada "barrido de threads" -- decode
  // NO se beneficia de mas threads (memory-bound), prefill e hifigan si.
  XttsEngine(std::string model_dir, int threads_prefill = 4, int threads_decode = 1,
             int threads_hifigan = 4);
  ~XttsEngine();

  SynthResult SynthSentence(const std::string& raw_text, unsigned int seed,
                             const SamplingParams& params = SamplingParams{});

  // Libera la sesion de decode (la mas grande). Tokenizador/latentes/HiFiGAN quedan vivos.
  // La proxima llamada a SynthSentence la vuelve a crear (perezoso), pagando de nuevo el costo
  // de carga medido (~5-9s en el dispositivo real, ver decisions.md) -- aceptable al volver de
  // segundo plano, no en medio de una sintesis.
  void ReleaseGpt();

 private:
  std::string model_dir_;
  int threads_prefill_, threads_decode_, threads_hifigan_;

  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> decode_sess_;
  std::unique_ptr<Ort::Session> hifigan_sess_;

  std::vector<float> cond_latent_;       // [1,32,1024]
  std::vector<float> speaker_embedding_;  // [1,512,1]

  void* tok_;  // xtts_tokenizer::BpeTokenizer*, tipo oculto para no exponer el header en la API publica

  Ort::SessionOptions MakeOpts(int threads) const;
  void EnsureDecodeSession();
  std::vector<int64_t> BuildTextIdsPadded(const std::string& raw_text) const;
};

}  // namespace xtts_engine
