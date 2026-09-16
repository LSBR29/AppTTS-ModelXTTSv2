#include "xtts_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "bpe_tokenizer.hpp"
#include "normalize_es.hpp"

namespace xtts_engine {

static const int N_LAYER = 25;
static const int N_EMBD = 1024;
static const int START_TEXT_TOKEN = 261;
static const int STOP_TEXT_TOKEN = 0;
static const int START_AUDIO_TOKEN = 1024;
static const int STOP_AUDIO_TOKEN = 1025;
static const int N_AUDIO_TOKENS = 1026;
static const int MAX_GEN_MEL_TOKENS = 602;
static const int COND_LEN = 32;

// Misma logica de muestreo que synth_cli.cpp (ver ese archivo para la justificacion completa,
// citas de infer_repl.py/gpt.py). Valores por defecto en SamplingParams; ajustables desde la app.
static int SampleToken(std::vector<float> logits, const std::vector<bool>& seen, std::mt19937& rng,
                        const SamplingParams& p) {
  const float penalty = p.repetition_penalty, temperature = p.temperature, top_p = p.top_p;
  const int top_k = std::clamp(p.top_k, 1, N_AUDIO_TOKENS);

  for (int i = 0; i < N_AUDIO_TOKENS; ++i) {
    if (seen[i]) logits[i] = (logits[i] > 0.0f) ? logits[i] / penalty : logits[i] * penalty;
  }
  for (float& v : logits) v /= temperature;

  std::vector<int> idx(N_AUDIO_TOKENS);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
  idx.resize(top_k);

  float max_logit = logits[idx[0]];
  std::vector<float> probs(top_k);
  float sum = 0.0f;
  for (int i = 0; i < top_k; ++i) { probs[i] = std::exp(logits[idx[i]] - max_logit); sum += probs[i]; }
  for (float& p : probs) p /= sum;

  float cum = 0.0f;
  int keep = 0;
  for (; keep < top_k; ++keep) {
    cum += probs[keep];
    if (cum > top_p) { keep++; break; }
  }
  keep = std::max(keep, 1);

  float keep_sum = 0.0f;
  for (int i = 0; i < keep; ++i) keep_sum += probs[i];
  std::vector<float> final_probs(keep);
  for (int i = 0; i < keep; ++i) final_probs[i] = probs[i] / keep_sum;

  std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
  return idx[dist(rng)];
}

static std::vector<float> ReadBinFloat(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("no se pudo abrir " + path);
  std::vector<float> data(expected_count);
  f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
  if (!f) throw std::runtime_error("archivo mas corto de lo esperado: " + path);
  return data;
}

static Ort::Value MakeI64(Ort::MemoryInfo& mem, std::vector<int64_t>& data, std::vector<int64_t> shape) {
  return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(), shape.data(), shape.size());
}
static Ort::Value MakeF32(Ort::MemoryInfo& mem, std::vector<float>& data, std::vector<int64_t> shape) {
  return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
}

XttsEngine::XttsEngine(std::string model_dir, int threads_prefill, int threads_decode, int threads_hifigan)
    : model_dir_(std::move(model_dir)),
      threads_prefill_(threads_prefill),
      threads_decode_(threads_decode),
      threads_hifigan_(threads_hifigan),
      env_(ORT_LOGGING_LEVEL_WARNING, "xtts_engine"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  auto* tok = new xtts_tokenizer::BpeTokenizer();
  tok->LoadVocab(model_dir_ + "/vocab.tsv");
  tok->LoadMerges(model_dir_ + "/merges.tsv");
  tok->LoadSpecialTokens(model_dir_ + "/special_tokens.tsv");
  tok_ = tok;

  cond_latent_ = ReadBinFloat(model_dir_ + "/cond_latent.bin", 1 * COND_LEN * N_EMBD);
  speaker_embedding_ = ReadBinFloat(model_dir_ + "/speaker_embedding.bin", 1 * 512 * 1);

  hifigan_sess_ = std::make_unique<Ort::Session>(
      env_, (model_dir_ + "/hifigan_decoder.onnx").c_str(), MakeOpts(threads_hifigan_));
  // decode_sess_ se crea perezosamente en el primer SynthSentence()/EnsureDecodeSession().
}

XttsEngine::~XttsEngine() {
  delete static_cast<xtts_tokenizer::BpeTokenizer*>(tok_);
}

Ort::SessionOptions XttsEngine::MakeOpts(int threads) const {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return opts;
}

void XttsEngine::EnsureDecodeSession() {
  if (!decode_sess_) {
    decode_sess_ = std::make_unique<Ort::Session>(
        env_, (model_dir_ + "/gpt_decode_step.onnx").c_str(), MakeOpts(threads_decode_));
  }
}

void XttsEngine::ReleaseGpt() {
  decode_sess_.reset();
}

std::vector<int64_t> XttsEngine::BuildTextIdsPadded(const std::string& raw_text) const {
  auto* tok = static_cast<xtts_tokenizer::BpeTokenizer*>(tok_);
  std::string normalized = xtts_tokenizer::NormalizeEs(raw_text);
  std::string wrapped = "[es]" + normalized;
  std::string final_text;
  for (char c : wrapped) final_text += (c == ' ') ? "[SPACE]" : std::string(1, c);
  std::vector<int> text_tokens = tok->Encode(final_text);

  std::vector<int64_t> text_ids_padded;
  text_ids_padded.push_back(START_TEXT_TOKEN);
  for (int t : text_tokens) text_ids_padded.push_back(t);
  text_ids_padded.push_back(STOP_TEXT_TOKEN);
  return text_ids_padded;
}

SynthResult XttsEngine::SynthSentence(const std::string& raw_text, unsigned int seed,
                                       const SamplingParams& params) {
  std::vector<int64_t> text_ids_padded = BuildTextIdsPadded(raw_text);
  int T_wrapped = static_cast<int>(text_ids_padded.size());

  static const std::vector<std::string> out_names_s = [] {
    std::vector<std::string> v = {"logits", "hidden_state"};
    for (int i = 0; i < N_LAYER; ++i) v.push_back("present_k_" + std::to_string(i));
    for (int i = 0; i < N_LAYER; ++i) v.push_back("present_v_" + std::to_string(i));
    return v;
  }();
  std::vector<const char*> out_names;
  for (auto& s : out_names_s) out_names.push_back(s.c_str());

  // ---- prefill: sesion propia, se crea/corre/DESTRUYE (RSS medido, ver decisions.md) ----
  std::vector<float> logits0(N_AUDIO_TOKENS), hidden0(N_EMBD);
  std::vector<std::vector<float>> ks(N_LAYER), vs(N_LAYER);
  int64_t kv_len = 0;
  {
    Ort::Session prefill_sess(env_, (model_dir_ + "/gpt_prefill.onnx").c_str(), MakeOpts(threads_prefill_));
    std::vector<float> cond_copy = cond_latent_;
    std::vector<int64_t> text_copy = text_ids_padded;
    std::vector<Ort::Value> inputs;
    inputs.push_back(MakeF32(mem_, cond_copy, {1, COND_LEN, N_EMBD}));
    inputs.push_back(MakeI64(mem_, text_copy, {1, T_wrapped}));
    const char* in_names[] = {"cond_latent", "text_ids_padded"};

    auto outputs = prefill_sess.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(),
                                     out_names.data(), out_names.size());
    std::memcpy(logits0.data(), outputs[0].GetTensorMutableData<float>(), N_AUDIO_TOKENS * sizeof(float));
    std::memcpy(hidden0.data(), outputs[1].GetTensorMutableData<float>(), N_EMBD * sizeof(float));

    auto kshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    kv_len = kshape[2];
    size_t kv_elems = static_cast<size_t>(kshape[0] * kshape[1] * kshape[2] * kshape[3]);
    for (int i = 0; i < N_LAYER; ++i) {
      ks[i].resize(kv_elems);
      vs[i].resize(kv_elems);
      std::memcpy(ks[i].data(), outputs[2 + i].GetTensorMutableData<float>(), kv_elems * sizeof(float));
      std::memcpy(vs[i].data(), outputs[2 + N_LAYER + i].GetTensorMutableData<float>(), kv_elems * sizeof(float));
    }
  }  // prefill_sess destruida aqui

  // repetition_penalty: historial inicial = placeholder id=1 (relleno cond+texto, gpt.py:499-508)
  // + start_audio_token. Ver synth_cli.cpp para la cita completa del hallazgo.
  std::vector<bool> seen(N_AUDIO_TOKENS, false);
  seen[1] = true;
  seen[START_AUDIO_TOKEN] = true;
  std::mt19937 rng(seed);

  std::vector<int> chosen;
  std::vector<std::vector<float>> hidden_seq;
  chosen.push_back(SampleToken(logits0, seen, rng, params));
  hidden_seq.push_back(hidden0);
  seen[chosen[0]] = true;

  // ---- decode: sesion persistente (creada perezosamente, sobrevive entre oraciones) ----
  EnsureDecodeSession();
  {
    std::vector<std::string> in_names_s = {"token_id", "mel_position_id"};
    for (int i = 0; i < N_LAYER; ++i) in_names_s.push_back("past_k_" + std::to_string(i));
    for (int i = 0; i < N_LAYER; ++i) in_names_s.push_back("past_v_" + std::to_string(i));
    std::vector<const char*> in_names;
    for (auto& s : in_names_s) in_names.push_back(s.c_str());

    std::vector<int64_t> kv_shape = {1, 16, kv_len, 64};
    int step = 0;
    while (chosen.back() != STOP_AUDIO_TOKEN && step < MAX_GEN_MEL_TOKENS) {
      std::vector<int64_t> tok_data = {chosen.back()};
      std::vector<int64_t> pos_data = {step + 1};
      std::vector<Ort::Value> inputs;
      inputs.push_back(MakeI64(mem_, tok_data, {1, 1}));
      inputs.push_back(MakeI64(mem_, pos_data, {1, 1}));
      kv_shape[2] = kv_len;
      for (int i = 0; i < N_LAYER; ++i) inputs.push_back(MakeF32(mem_, ks[i], kv_shape));
      for (int i = 0; i < N_LAYER; ++i) inputs.push_back(MakeF32(mem_, vs[i], kv_shape));

      auto outputs = decode_sess_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                        inputs.size(), out_names.data(), out_names.size());
      std::vector<float> logits(N_AUDIO_TOKENS);
      std::memcpy(logits.data(), outputs[0].GetTensorMutableData<float>(), N_AUDIO_TOKENS * sizeof(float));
      std::vector<float> hidden(N_EMBD);
      std::memcpy(hidden.data(), outputs[1].GetTensorMutableData<float>(), N_EMBD * sizeof(float));

      auto kshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
      kv_len = kshape[2];
      size_t kv_elems = static_cast<size_t>(kshape[0] * kshape[1] * kshape[2] * kshape[3]);
      for (int i = 0; i < N_LAYER; ++i) {
        ks[i].resize(kv_elems);
        vs[i].resize(kv_elems);
        std::memcpy(ks[i].data(), outputs[2 + i].GetTensorMutableData<float>(), kv_elems * sizeof(float));
        std::memcpy(vs[i].data(), outputs[2 + N_LAYER + i].GetTensorMutableData<float>(), kv_elems * sizeof(float));
      }

      int next_tok = SampleToken(logits, seen, rng, params);
      seen[next_tok] = true;
      chosen.push_back(next_tok);
      hidden_seq.push_back(hidden);
      step++;
    }
  }

  // ---- HiFiGAN: sesion persistente, una pasada sobre toda la secuencia de esta oracion ----
  int T_audio = static_cast<int>(hidden_seq.size());
  std::vector<float> latents_flat(static_cast<size_t>(T_audio) * N_EMBD);
  for (int t = 0; t < T_audio; ++t)
    std::memcpy(latents_flat.data() + static_cast<size_t>(t) * N_EMBD, hidden_seq[t].data(), N_EMBD * sizeof(float));

  SynthResult result;
  result.n_audio_tokens = T_audio;
  result.stopped_naturally = (chosen.back() == STOP_AUDIO_TOKEN);
  {
    std::vector<float> latents_copy = latents_flat;
    std::vector<float> g_copy = speaker_embedding_;
    std::vector<Ort::Value> inputs;
    inputs.push_back(MakeF32(mem_, latents_copy, {1, T_audio, N_EMBD}));
    inputs.push_back(MakeF32(mem_, g_copy, {1, 512, 1}));
    const char* hg_in_names[] = {"latents", "g"};
    const char* hg_out_names[] = {"wav"};
    auto outputs = hifigan_sess_->Run(Ort::RunOptions{nullptr}, hg_in_names, inputs.data(), inputs.size(),
                                       hg_out_names, 1);
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    result.wav.resize(n);
    std::memcpy(result.wav.data(), outputs[0].GetTensorMutableData<float>(), n * sizeof(float));
  }
  return result;
}

}  // namespace xtts_engine
