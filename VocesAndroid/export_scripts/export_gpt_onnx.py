"""
Exporta el GPT autorregresivo de un checkpoint XTTS-v2 (formato VocesWindows/VocesLinux:
best_model.pth + config.json + vocab.json + ref.wav) a dos grafos ONNX listos para Android:

  gpt_prefill.onnx      -- procesa condicionamiento + texto en una sola pasada causal. fp32.
  gpt_decode_step.onnx  -- un token de audio nuevo contra la cache KV, un paso por llamada.
                           Precisión mixta: los pesos de las proyecciones de atención y del MLP
                           corren en fp16 (más rápido en este runtime para este tamaño de
                           entrada), pero el stream residual y el cálculo de atención en sí se
                           mantienen en fp32 -- necesario porque el residual crece a magnitudes
                           donde la mantisa de fp16 (~3 dígitos decimales) ya no alcanza para la
                           suma acumulada de 25 capas autorregresivas. El prefill se deja en fp32
                           completo porque, medido en el hardware real, fp16 resulta MÁS LENTO
                           que fp32 quando la entrada tiene varios tokens a la vez (el prefill
                           procesa todo el texto de una pasada) -- solo el paso de decode (un
                           token a la vez) se beneficia de fp16 en este runtime.

Ambos grafos tienen TRES salidas: logits, hidden_state y la cache KV por capa (present_k_i/
present_v_i). El `hidden_state` es el tensor que consume el vocoder (HiFiGAN) -- no basta con
`logits`, que solo sirve para elegir el token, no para la calidad del audio de ese token. La cache
KV siempre es fp32 en la interfaz del grafo (entrada y salida), independientemente de la
precisión interna del cálculo -- así la app no necesita saber ni le importa qué precisión usa el
grafo por dentro.

Además calcula y vuelca los "conditioning latents" (huella de la voz de referencia, `ref.wav`)
como los dos binarios crudos que la app Android carga directamente: cond_latent.bin
(1x32x1024 float32) y speaker_embedding.bin (1x512x1 float32).

Uso:
    python export_gpt_onnx.py \\
        --checkpoint ../../VocesWindows/models/2025acv02/best_model.pth \\
        --config     ../../VocesWindows/models/2025acv02/config.json \\
        --vocab      ../../VocesWindows/models/2025acv02/vocab.json \\
        --ref-wav    ../../VocesWindows/models/2025acv02/ref.wav \\
        --tts-root   ../../VocesWindows \\
        --outdir     ../generated_models

Requiere el Python portable + `TTS` que ya trae VocesWindows/VocesLinux (torch, transformers,
onnxruntime, coqui-tts) -- correr este script CON ESE python, no con un Python del sistema:

    ..\\VocesWindows\\python\\python.exe export_gpt_onnx.py [argumentos]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LAYER = 25
N_EMBD = 1024
N_HEAD = 16
HEAD_DIM = N_EMBD // N_HEAD
N_AUDIO_TOKENS = 1026


def gelu_new(x):
    # NewGELUActivation de HF (tanh-approx) -- NO la nn.GELU() por defecto de PyTorch.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GptCore(nn.Module):
    """Contenedor de los pesos reales del GPT (25 capas). GptPrefill/GptDecodeStep lo usan por
    composición, compartiendo los mismos tensores (no se cargan dos veces)."""

    def __init__(self, sd):
        super().__init__()
        prefix = "xtts.gpt."

        def w(key):
            return sd[prefix + key]

        self.mel_embedding = nn.Embedding.from_pretrained(w("mel_embedding.weight"), freeze=True)
        self.mel_pos_embedding = nn.Embedding.from_pretrained(w("mel_pos_embedding.emb.weight"), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(w("text_embedding.weight"), freeze=True)
        self.text_pos_embedding = nn.Embedding.from_pretrained(w("text_pos_embedding.emb.weight"), freeze=True)
        self.mel_head = nn.Linear(N_EMBD, N_AUDIO_TOKENS)
        with torch.no_grad():
            self.mel_head.weight.copy_(w("mel_head.weight"))
            self.mel_head.bias.copy_(w("mel_head.bias"))
        self.ln_f = nn.LayerNorm(N_EMBD)
        with torch.no_grad():
            self.ln_f.weight.copy_(w("gpt.ln_f.weight"))
            self.ln_f.bias.copy_(w("gpt.ln_f.bias"))
        self.final_norm = nn.LayerNorm(N_EMBD)
        with torch.no_grad():
            self.final_norm.weight.copy_(w("final_norm.weight"))
            self.final_norm.bias.copy_(w("final_norm.bias"))

        self.ln_1 = nn.ModuleList()
        self.c_attn_w = nn.ParameterList()
        self.c_attn_b = nn.ParameterList()
        self.c_proj_w = nn.ParameterList()
        self.c_proj_b = nn.ParameterList()
        self.ln_2 = nn.ModuleList()
        self.mlp_fc_w = nn.ParameterList()
        self.mlp_fc_b = nn.ParameterList()
        self.mlp_proj_w = nn.ParameterList()
        self.mlp_proj_b = nn.ParameterList()

        for i in range(N_LAYER):
            p = f"gpt.h.{i}."
            ln1 = nn.LayerNorm(N_EMBD)
            with torch.no_grad():
                ln1.weight.copy_(w(p + "ln_1.weight"))
                ln1.bias.copy_(w(p + "ln_1.bias"))
            self.ln_1.append(ln1)
            ln2 = nn.LayerNorm(N_EMBD)
            with torch.no_grad():
                ln2.weight.copy_(w(p + "ln_2.weight"))
                ln2.bias.copy_(w(p + "ln_2.bias"))
            self.ln_2.append(ln2)
            # Conv1D de HF guarda weight=[in,out] y hace x @ weight + bias directo (verificado en
            # transformers/pytorch_utils.py) -- NO transponer, a diferencia de nn.Linear.
            self.c_attn_w.append(nn.Parameter(w(p + "attn.c_attn.weight").clone(), requires_grad=False))
            self.c_attn_b.append(nn.Parameter(w(p + "attn.c_attn.bias").clone(), requires_grad=False))
            self.c_proj_w.append(nn.Parameter(w(p + "attn.c_proj.weight").clone(), requires_grad=False))
            self.c_proj_b.append(nn.Parameter(w(p + "attn.c_proj.bias").clone(), requires_grad=False))
            self.mlp_fc_w.append(nn.Parameter(w(p + "mlp.c_fc.weight").clone(), requires_grad=False))
            self.mlp_fc_b.append(nn.Parameter(w(p + "mlp.c_fc.bias").clone(), requires_grad=False))
            self.mlp_proj_w.append(nn.Parameter(w(p + "mlp.c_proj.weight").clone(), requires_grad=False))
            self.mlp_proj_b.append(nn.Parameter(w(p + "mlp.c_proj.bias").clone(), requires_grad=False))

    def block_forward(self, i, x, past_k, past_v):
        B, S, _ = x.shape
        residual = x
        h = self.ln_1[i](x)
        qkv = h @ self.c_attn_w[i] + self.c_attn_b[i]
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        new_k = torch.cat([past_k, k], dim=2)
        new_v = torch.cat([past_v, v], dim=2)
        att = torch.matmul(q, new_k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        Lpast = past_k.shape[2]
        if S > 1:
            idx = torch.arange(S, device=x.device)
            allow_past = torch.ones(1, 1, S, Lpast, dtype=torch.bool, device=x.device)
            new_cols = idx.view(1, 1, S, 1) >= idx.view(1, 1, 1, S)
            mask = torch.cat([allow_past, new_cols], dim=3)
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, new_v)
        out = out.transpose(1, 2).reshape(B, S, N_EMBD)
        out = out @ self.c_proj_w[i] + self.c_proj_b[i]
        x = residual + out
        residual = x
        h = self.ln_2[i](x)
        h = gelu_new(h @ self.mlp_fc_w[i] + self.mlp_fc_b[i])
        h = h @ self.mlp_proj_w[i] + self.mlp_proj_b[i]
        x = residual + h
        return x, new_k, new_v

    def head(self, x):
        # hidden = final_norm(ln_f(x)) es EXACTAMENTE el tensor que la produccion real entrega
        # al vocoder por paso (ver gpt_inference.py / stream_generator.py del repo TTS real).
        x = self.ln_f(x)
        hidden = self.final_norm(x)
        logits = self.mel_head(hidden)
        return hidden, logits


class GptCoreFp16(GptCore):
    """Variante de GptCore para el grafo de decode: las proyecciones de atención y del MLP (las
    matrices grandes) corren en fp16, LayerNorm y las tablas de embeddings se quedan en fp32.
    El stream residual (`x`) NUNCA se guarda en fp16 -- solo se castea a fp16 inmediatamente
    antes de cada multiplicación de matriz grande y se vuelve a fp32 apenas termina (igual que
    mixed-precision/autocast real). La atención en sí (`q@k^T`, `softmax@v`) también se queda en
    fp32: son las matrices más grandes del grafo (la cache KV completa) pero representan una
    fracción mínima del cómputo de la capa, así que castearlas a fp16 cuesta más de lo que
    ahorra."""

    def __init__(self, sd):
        super().__init__(sd)
        for plist_name in ["c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b",
                            "mlp_fc_w", "mlp_fc_b", "mlp_proj_w", "mlp_proj_b"]:
            plist = getattr(self, plist_name)
            for i in range(len(plist)):
                plist[i] = nn.Parameter(plist[i].data.half(), requires_grad=False)
        self.mel_head.half()

    def block_forward(self, i, x, past_k, past_v):
        x = x.float()
        B, S, _ = x.shape
        residual = x
        h = self.ln_1[i](x)
        qkv = (h.half() @ self.c_attn_w[i] + self.c_attn_b[i]).float()
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, N_HEAD, HEAD_DIM).transpose(1, 2)
        new_k = torch.cat([past_k, k], dim=2)
        new_v = torch.cat([past_v, v], dim=2)
        att = torch.matmul(q, new_k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        Lpast = past_k.shape[2]
        if S > 1:
            idx = torch.arange(S, device=x.device)
            allow_past = torch.ones(1, 1, S, Lpast, dtype=torch.bool, device=x.device)
            new_cols = idx.view(1, 1, S, 1) >= idx.view(1, 1, 1, S)
            mask = torch.cat([allow_past, new_cols], dim=3)
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, new_v)
        out = out.transpose(1, 2).reshape(B, S, N_EMBD)
        out = (out.half() @ self.c_proj_w[i] + self.c_proj_b[i]).float()
        x = residual + out
        residual = x
        h = self.ln_2[i](x)
        h = (h.half() @ self.mlp_fc_w[i] + self.mlp_fc_b[i]).float()
        h = gelu_new(h)
        h = (h.half() @ self.mlp_proj_w[i] + self.mlp_proj_b[i]).float()
        x = residual + h
        return x, new_k, new_v

    def head(self, x):
        x = self.ln_f(x.float())
        hidden = self.final_norm(x)
        logits = self.mel_head(hidden.half()).float()
        return hidden, logits


class GptPrefill(nn.Module):
    def __init__(self, core: GptCore, start_audio_token: int):
        super().__init__()
        self.core = core
        self.start_audio_token = start_audio_token

    def forward(self, cond_latent, text_ids_padded):
        B, T = cond_latent.shape[0], text_ids_padded.shape[1]
        text_pos_ids = torch.arange(T, device=text_ids_padded.device).unsqueeze(0).expand(B, T)
        text_emb = self.core.text_embedding(text_ids_padded) + self.core.text_pos_embedding(text_pos_ids)
        prefix_emb = torch.cat([cond_latent, text_emb], dim=1)
        start_tok = torch.full((B, 1), self.start_audio_token, dtype=torch.long, device=cond_latent.device)
        gen_emb = self.core.mel_embedding(start_tok) + self.core.mel_pos_embedding(
            torch.zeros(B, 1, dtype=torch.long, device=cond_latent.device)
        )
        x = torch.cat([prefix_emb, gen_emb], dim=1)
        zero_past = torch.zeros(B, N_HEAD, 0, HEAD_DIM, dtype=x.dtype, device=x.device)
        present_ks, present_vs = [], []
        for i in range(N_LAYER):
            x, k, v = self.core.block_forward(i, x, zero_past, zero_past)
            present_ks.append(k)
            present_vs.append(v)
        hidden, logits = self.core.head(x[:, -1:, :])
        return (logits, hidden, *present_ks, *present_vs)


class GptDecodeStep(nn.Module):
    def __init__(self, core: GptCore):
        super().__init__()
        self.core = core

    def forward(self, token_id, mel_position_id, *past_kv_flat):
        x = self.core.mel_embedding(token_id) + self.core.mel_pos_embedding(mel_position_id)
        present_ks, present_vs = [], []
        for i in range(N_LAYER):
            past_k = past_kv_flat[2 * i]
            past_v = past_kv_flat[2 * i + 1]
            x, k, v = self.core.block_forward(i, x, past_k, past_v)
            present_ks.append(k)
            present_vs.append(v)
        hidden, logits = self.core.head(x)
        return (logits, hidden, *present_ks, *present_vs)


def load_model(tts_root, checkpoint, config, vocab):
    sys.path.insert(0, tts_root)
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    xtts_config = XttsConfig()
    xtts_config.load_json(config)
    model = Xtts(xtts_config)
    model.load_checkpoint(xtts_config, checkpoint_path=checkpoint, vocab_path=vocab, use_deepspeed=False)
    model.eval()
    model.gpt.init_gpt_for_inference(kv_cache=True)
    return model


def validate_core(core, label, gpt_cond_latent, text_ids_padded, real_logits0, real_hidden0, start_audio_token):
    """Corre una pasada de prefill con `core` (fp32 o fp16) y la compara contra la salida real del
    modelo de producción -- válido para cualquier precisión porque GptPrefill/block_forward tienen
    la misma interfaz sin importar en qué precisión corran los pesos por dentro."""
    prefill = GptPrefill(core, start_audio_token).eval()
    pre_out = prefill(gpt_cond_latent, text_ids_padded)
    my_logits0, my_hidden0 = pre_out[0].float(), pre_out[1].float()
    max_err = float((my_logits0 - real_logits0).abs().max())
    max_err_h = float((my_hidden0 - real_hidden0).abs().max())
    tol = 1e-2
    status = "OK" if max_err < tol and max_err_h < tol else "FALLO"
    print(f"Validacion {label} vs. modelo real: max_abs_err logits={max_err:.2e} "
          f"hidden={max_err_h:.2e} -> {status}")
    if status == "FALLO":
        raise RuntimeError(f"La validacion del export ({label}) fallo -- revisar antes de usar estos grafos.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--checkpoint", default=os.path.join(here, "..", "..", "VocesWindows", "models", "2025acv02", "best_model.pth"))
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "VocesWindows", "models", "2025acv02", "config.json"))
    ap.add_argument("--vocab", default=os.path.join(here, "..", "..", "VocesWindows", "models", "2025acv02", "vocab.json"))
    ap.add_argument("--ref-wav", default=os.path.join(here, "..", "..", "VocesWindows", "models", "2025acv02", "ref.wav"))
    ap.add_argument("--tts-root", default=os.path.join(here, "..", "..", "VocesWindows"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "generated_models"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Cargando checkpoint ({args.checkpoint})...")
    model = load_model(args.tts_root, args.checkpoint, args.config, args.vocab)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    core_fp32 = GptCore(sd).eval()
    core_fp16 = GptCoreFp16(sd).eval()

    with torch.no_grad():
        print("Calculando conditioning latents desde ref.wav...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[args.ref_wav])
        gpt_cond_latent.numpy().astype(np.float32).tofile(os.path.join(args.outdir, "cond_latent.bin"))
        speaker_embedding.numpy().astype(np.float32).tofile(os.path.join(args.outdir, "speaker_embedding.bin"))
        print(f"  cond_latent.bin: {tuple(gpt_cond_latent.shape)}, speaker_embedding.bin: {tuple(speaker_embedding.shape)}")

        # Texto/tokens de prueba solo para trazar un ejemplo de exportacion y validar -- no forma
        # parte del modelo exportado (los grafos ONNX aceptan cualquier texto en tiempo de uso).
        text = "Hola, esto es una prueba de sintesis de voz en espanol."
        text_tokens = torch.tensor(model.tokenizer.encode(text, lang="es"), dtype=torch.long).unsqueeze(0)
        text_ids_padded = F.pad(text_tokens, (0, 1), value=model.gpt.stop_text_token)
        text_ids_padded = F.pad(text_ids_padded, (1, 0), value=model.gpt.start_text_token)

        # ---- Validacion rapida: comparar contra el modelo real de produccion (gpt_inference +
        # DynamicCache reales) en unos pocos pasos reales, antes de confiar en el export ----
        from transformers.cache_utils import DynamicCache

        gpt_inputs = model.gpt.compute_embeddings(gpt_cond_latent, text_tokens)
        real_cache = DynamicCache(config=model.gpt.gpt_inference.config)
        attn_mask = torch.ones(1, gpt_inputs.shape[1], dtype=torch.long)
        real_out = model.gpt.gpt_inference(
            input_ids=gpt_inputs, past_key_values=real_cache, attention_mask=attn_mask,
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        real_logits0 = real_out.logits[:, -1:, :]
        real_hidden0 = model.gpt.gpt_inference.final_norm(real_out.hidden_states[-1][:, -1:])

        validate_core(core_fp32, "prefill fp32", gpt_cond_latent, text_ids_padded, real_logits0, real_hidden0, model.gpt.start_audio_token)
        validate_core(core_fp16, "decode fp16", gpt_cond_latent, text_ids_padded, real_logits0, real_hidden0, model.gpt.start_audio_token)

        # ---- Export a ONNX ----
        # dynamo=False: fuerza el exportador clasico basado en TorchScript -- el nuevo exportador
        # de PyTorch (basado en torch.export) no maneja bien el patron de entradas usado aqui
        # (25 pares de tensores de cache pasados como *args). Necesario en versiones de PyTorch
        # donde el exportador nuevo ya es el default.
        prefill = GptPrefill(core_fp32, model.gpt.start_audio_token).eval()
        decode_step = GptDecodeStep(core_fp16).eval()

        prefill_path = os.path.join(args.outdir, "gpt_prefill.onnx")
        output_names = ["logits", "hidden_state"] + [f"present_k_{i}" for i in range(N_LAYER)] + [f"present_v_{i}" for i in range(N_LAYER)]
        dyn = {"cond_latent": {0: "batch"}, "text_ids_padded": {0: "batch", 1: "text_len"},
               "logits": {0: "batch"}, "hidden_state": {0: "batch"}}
        for i in range(N_LAYER):
            dyn[f"present_k_{i}"] = {0: "batch", 2: "prefill_len"}
            dyn[f"present_v_{i}"] = {0: "batch", 2: "prefill_len"}
        torch.onnx.export(
            prefill, (gpt_cond_latent, text_ids_padded), prefill_path,
            input_names=["cond_latent", "text_ids_padded"], output_names=output_names,
            dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False,
        )
        print(f"  {prefill_path} ({os.path.getsize(prefill_path)/1e6:.1f} MB)")

        decode_path = os.path.join(args.outdir, "gpt_decode_step.onnx")
        flat_names = []
        dyn_b = {"token_id": {0: "batch"}, "mel_position_id": {0: "batch"},
                 "logits": {0: "batch"}, "hidden_state": {0: "batch"}}
        for i in range(N_LAYER):
            flat_names += [f"past_k_{i}", f"past_v_{i}"]
            dyn_b[f"past_k_{i}"] = {0: "batch", 2: "past_len"}
            dyn_b[f"past_v_{i}"] = {0: "batch", 2: "past_len"}
            dyn_b[f"present_k_{i}"] = {0: "batch", 2: "present_len"}
            dyn_b[f"present_v_{i}"] = {0: "batch", 2: "present_len"}
        example_tok = torch.tensor([[100]], dtype=torch.long)
        example_pos = torch.tensor([[1]], dtype=torch.long)
        # La cache KV siempre se pasa en fp32, sin importar la precision interna del grafo.
        example_past = [torch.zeros(1, N_HEAD, 33, HEAD_DIM, dtype=torch.float32) for _ in range(2 * N_LAYER)]
        torch.onnx.export(
            decode_step, (example_tok, example_pos, *example_past), decode_path,
            input_names=["token_id", "mel_position_id"] + flat_names, output_names=output_names,
            dynamic_axes=dyn_b, opset_version=17, do_constant_folding=True, dynamo=False,
        )
        print(f"  {decode_path} ({os.path.getsize(decode_path)/1e6:.1f} MB)")

    print("\nOK -- export completo y validado.")


if __name__ == "__main__":
    main()
