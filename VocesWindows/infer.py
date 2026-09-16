import queue
import sys
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from scipy import signal as scipy_signal
from tqdm import tqdm

from interfaz import lanzar_interfaz

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models/2025acv02/best_model.pth"
CONFIG_PATH = ROOT / "models/2025acv02/config.json"
VOCAB_PATH = ROOT / "models/2025acv02/vocab.json"
SPEAKER_REFERENCE = [str(ROOT / "models/2025acv02/ref.wav")]
ONNX_VOCODER = ROOT / "onnx_models/2025acv02/hifigan_decoder.onnx"
SAMPLE_RATE = 24000
STREAM_CHUNK_SIZE = 60  # Punto medio: RTF 2.21x medido (vs 4.14x en 20 y 1.72x en 200), con primer sonido mucho mas rapido que chunk=200 (~10s)
PREBUFFER_CHUNKS = 1  # Cuantos chunks juntar antes de arrancar a reproducir
CPU_CORES = 6       # Adaptar al tipo de dispositivo

# Postprocesado
POST_GAIN = 1.6
LIMITER_THRESHOLD = 0.92
LIMITER_CEILING = 0.98

# Ecualizador de 3 bandas 
# Cada banda es independiente:
#   *_GAIN_DB en 0.0 = sin cambio en esa banda. Negativo = atenua, positivo = realza.
#   Ejemplos utiles:
#     - "mas suave/menos aspera": bajar HIGH_SHELF_GAIN_DB a -4.0 o -6.0
#     - "mas calida": subir LOW_SHELF_GAIN_DB a +2.0 y bajar HIGH_SHELF_GAIN_DB a -3.0
#     - "mas brillante/nitida": subir HIGH_SHELF_GAIN_DB a +2.0 o +3.0
#     - "voz mas fina/menos golpeada": bajar MID_GAIN_DB
LOW_SHELF_FREQ_HZ = 200.0
LOW_SHELF_GAIN_DB = -12.0
MID_FREQ_HZ = 1000.0
MID_GAIN_DB = 5.0
MID_Q = 1.0
HIGH_SHELF_FREQ_HZ = 6000.0
HIGH_SHELF_GAIN_DB = 4.0

# Semitonos a bajar (negativo = mas grave, positivo = mas agudo,
# 0.0 = desactivado por completo). -2.0 es un cambio moderado: -4.0 o -5.0 ya suena
# bastante mas grave.

PITCH_SHIFT_SEMITONES = -1.8

def _biquad_low_shelf(freq_hz, gain_db, fs, shelf_slope=1.0):
    """Formulas RBJ Audio EQ Cookbook (estandar, ver W3C Audio EQ Cookbook)."""
    a_amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((a_amp + 1 / a_amp) * (1 / shelf_slope - 1) + 2)
    sqrt_a = np.sqrt(a_amp)
    b0 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = 2 * a_amp * ((a_amp - 1) - (a_amp + 1) * cos_w0)
    b2 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a_amp + 1) + (a_amp - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = -2 * ((a_amp - 1) + (a_amp + 1) * cos_w0)
    a2 = (a_amp + 1) + (a_amp - 1) * cos_w0 - 2 * sqrt_a * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_high_shelf(freq_hz, gain_db, fs, shelf_slope=1.0):
    a_amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((a_amp + 1 / a_amp) * (1 / shelf_slope - 1) + 2)
    sqrt_a = np.sqrt(a_amp)
    b0 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a_amp * ((a_amp - 1) + (a_amp + 1) * cos_w0)
    b2 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a_amp + 1) - (a_amp - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a_amp - 1) - (a_amp + 1) * cos_w0)
    a2 = (a_amp + 1) - (a_amp - 1) * cos_w0 - 2 * sqrt_a * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_peaking(freq_hz, gain_db, q, fs):
    a_amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2 * q)
    b0 = 1 + alpha * a_amp
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a_amp
    a0 = 1 + alpha / a_amp
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a_amp
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


class ThreeBandEqualizer:
    """EQ de 3 bandas (low-shelf, peaking-mid, high-shelf) aplicado chunk a chunk con estado de
    filtro (zi) persistente entre chunks -- necesario para que no haya clicks en los bordes de
    chunk (filtrar cada chunk con estado inicial cero cada vez sonaria mal). Costo: 3 filtros IIR
    de segundo orden (biquad) por chunk, medido ~0.3-0.6ms para chunks de 50-67k samples,
    sigue siendo despreciable frente a los 2-4s de generacion por chunk."""

    def __init__(self, fs):
        self._filters = []
        for gain_db, coef_fn, extra in (
            (LOW_SHELF_GAIN_DB, _biquad_low_shelf, (LOW_SHELF_FREQ_HZ, fs)),
            (MID_GAIN_DB, _biquad_peaking, (MID_FREQ_HZ, MID_Q, fs)),
            (HIGH_SHELF_GAIN_DB, _biquad_high_shelf, (HIGH_SHELF_FREQ_HZ, fs)),
        ):
            if gain_db == 0.0:
                continue  # banda neutra -- no aplicar filtro, ahorra el costo por completo
            if coef_fn is _biquad_peaking:
                b, a = coef_fn(extra[0], gain_db, extra[1], extra[2])
            else:
                b, a = coef_fn(extra[0], gain_db, extra[1])
            self._filters.append((b, a, scipy_signal.lfilter_zi(b, a) * 0.0))

    def process(self, chunk_np):
        for i, (b, a, zi) in enumerate(self._filters):
            chunk_np, zi_out = scipy_signal.lfilter(b, a, chunk_np, zi=zi)
            self._filters[i] = (b, a, zi_out)
        return chunk_np.astype(np.float32)


class PitchShifter:
    """Baja (o sube) el tono fundamental de la voz via resampleo puro (sin STFT/vocoder de fase --
    ver comentario de PITCH_SHIFT_SEMITONES mas arriba para el por que). Se puede procesar cada
    chunk de forma independiente sin acumular estado -- medido: salto en los bordes de chunk MENOR
    al salto tipico del resto de la señal (0.00132 vs ~0.008), mejor que sin ningun postprocesado
    de pitch."""

    def __init__(self, semitones):
        self.semitones = semitones
        self._pitch_ratio = 2 ** (semitones / 12) if semitones != 0.0 else 1.0

    def process(self, chunk_np, sr):
        if self.semitones == 0.0:
            return chunk_np  # desactivado -- sin costo
        shifted = librosa.resample(
            chunk_np, orig_sr=sr * self._pitch_ratio, target_sr=sr, res_type="soxr_hq"
        )
        return shifted.astype(np.float32)


def warmup_pitch_shift():
    """Corre el resampleador una vez durante la carga del modelo (no en la primera sintesis real
    del usuario) -- soxr no tiene el costo de compilacion JIT que si tenia el vocoder de fase
    (numba), pero el warmup no cuesta nada y evita depender de esa suposicion sin verificarla en
    cada maquina. No hace nada si PITCH_SHIFT_SEMITONES == 0.0."""
    if PITCH_SHIFT_SEMITONES == 0.0:
        return
    dummy = np.zeros(4096, dtype=np.float32)
    pitch_ratio = 2 ** (PITCH_SHIFT_SEMITONES / 12)
    librosa.resample(dummy, orig_sr=SAMPLE_RATE * pitch_ratio, target_sr=SAMPLE_RATE, res_type="soxr_hq")


def postprocess_chunk(chunk_np, equalizer, pitch_shifter):
    chunk_np = pitch_shifter.process(chunk_np, SAMPLE_RATE)
    chunk_np = equalizer.process(chunk_np)
    chunk_np = chunk_np * POST_GAIN
    abs_chunk = np.abs(chunk_np)
    over = abs_chunk > LIMITER_THRESHOLD
    if np.any(over):
        excess = abs_chunk[over] - LIMITER_THRESHOLD
        headroom = LIMITER_CEILING - LIMITER_THRESHOLD
        compressed = LIMITER_THRESHOLD + headroom * excess / (excess + headroom)
        chunk_np[over] = np.sign(chunk_np[over]) * compressed
    return chunk_np


_SENTINEL = object()


def load_model():
    # El GPT autoregresivo genera un token a la vez (batch=1, seq_len=1 por paso), asi que cada matmul cuantizada int8
    # es "delgada" -- medido: paralelizar esas matmuls entre muchos threads cuesta mas en
    # sincronizacion que lo que ahorra en computo. 
    torch.set_num_threads(CPU_CORES)

    sys.path.insert(0, str(ROOT))
    from quantize_dynamic_int8 import build_int8_dynamic_model

    model, _ = build_int8_dynamic_model(CHECKPOINT_PATH, CONFIG_PATH, VOCAB_PATH)

    vocoder_sess_options = ort.SessionOptions()
    vocoder_sess_options.intra_op_num_threads = 12
    vocoder_sess_options.inter_op_num_threads = 1
    vocoder_sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    vocoder_sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess = ort.InferenceSession(
        str(ONNX_VOCODER), sess_options=vocoder_sess_options, providers=["CPUExecutionProvider"]
    )

    def onnx_hifigan_forward(latents, g=None):
        out = sess.run(None, {"latents": latents.detach().cpu().numpy(), "g": g.detach().cpu().numpy()})[0]
        return torch.from_numpy(out)

    model.hifigan_decoder.forward = onnx_hifigan_forward

    max_audio_tokens = model.args.gpt_max_audio_tokens
    return model, max_audio_tokens


def synthesize(model, text, max_audio_tokens, gpt_cond_latent, speaker_embedding,
                temperature=0.85, top_k=50, top_p=0.85, repetition_penalty=2.0, log=None):
    """Genera y reproduce en streaming con un hilo productor (generacion) separado del hilo
    consumidor (reproduccion), con pre-buffer chico y pausa limpia en subejecucion (underrun) en
    vez de cortes crudos. Devuelve (full_wav, stats) -- stats trae numeros medidos reales, no
    estimados, para poder verificar el comportamiento.

    `log`, si se pasa, es un callable(str) al que se le reportan los mismos datos que antes solo
    se veian en la consola (parametros usados, avance por chunk, primer sonido, underruns) -- lo
    usa la interfaz grafica para mostrarlos en la pestana de Ajustes."""
    import sounddevice as sd

    top_k = int(top_k)
    total_chunks_estimate = max(1, -(-max_audio_tokens // STREAM_CHUNK_SIZE))
    chunk_queue: "queue.Queue" = queue.Queue()
    producer_error = []

    if log:
        log(f"Parametros: temperature={temperature} top_k={top_k} top_p={top_p} "
            f"repetition_penalty={repetition_penalty}")

    pbar = tqdm(total=total_chunks_estimate, desc="Generando audio", unit="chunk", leave=False)
    equalizer = ThreeBandEqualizer(SAMPLE_RATE)  # estado nuevo por cada texto/llamada
    pitch_shifter = PitchShifter(PITCH_SHIFT_SEMITONES)  # idem -- acumulador propio por llamada

    def producer():
        try:
            with torch.no_grad():
                t_chunk = time.perf_counter()
                for i, wav_chunk in enumerate(model.inference_stream(
                    text, "es", gpt_cond_latent, speaker_embedding,
                    stream_chunk_size=STREAM_CHUNK_SIZE,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )):
                    chunk_np = wav_chunk.detach().cpu().numpy().astype(np.float32)
                    chunk_np = postprocess_chunk(chunk_np, equalizer, pitch_shifter)
                    chunk_queue.put(chunk_np)
                    pbar.update(1)
                    if log:
                        elapsed = time.perf_counter() - t_chunk
                        log(f"  chunk {i + 1}/~{total_chunks_estimate} generado ({elapsed:.1f}s)")
                        t_chunk = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            producer_error.append(e)
        finally:
            chunk_queue.put(_SENTINEL)

    producer_thread = threading.Thread(target=producer, daemon=True)
    t_start = time.perf_counter()
    producer_thread.start()

    # Pre-buffer: juntar unos chunks antes de arrancar a reproducir, para absorber jitter chico
    # sin sumar mucha espera (la generacion es mas lenta que tiempo real de por si, ver docstring
    # -- un pre-buffer grande solo demora el primer sonido sin evitar los cortes en textos largos).
    prebuffered = []
    producer_done = False
    while len(prebuffered) < PREBUFFER_CHUNKS:
        item = chunk_queue.get()
        if item is _SENTINEL:
            producer_done = True
            break
        prebuffered.append(item)

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    chunks_played = []
    t_first_chunk = None
    underrun_count = 0
    underrun_total_s = 0.0
    waiting_since = None

    def play_chunk(chunk_np):
        nonlocal t_first_chunk
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
        chunks_played.append(chunk_np)
        stream.write(chunk_np)

    try:
        for c in prebuffered:
            play_chunk(c)

        while not producer_done:
            try:
                item = chunk_queue.get(timeout=0.05)
            except queue.Empty:
                if waiting_since is None:
                    waiting_since = time.perf_counter()
                    underrun_count += 1
                continue
            if waiting_since is not None:
                underrun_total_s += time.perf_counter() - waiting_since
                waiting_since = None
            if item is _SENTINEL:
                producer_done = True
                break
            play_chunk(item)
    finally:
        stream.stop()
        stream.close()
        pbar.close()

    producer_thread.join(timeout=5)
    if producer_error:
        raise producer_error[0]

    t_end = time.perf_counter()
    full_wav = np.concatenate(chunks_played) if chunks_played else np.zeros(0, dtype=np.float32)
    audio_dur_s = len(full_wav) / SAMPLE_RATE
    stats = {
        "wall_s": t_end - t_start,
        "audio_dur_s": audio_dur_s,
        "first_chunk_s": (t_first_chunk - t_start) if t_first_chunk is not None else None,
        "underrun_count": underrun_count,
        "underrun_total_s": underrun_total_s,
    }
    if log:
        if stats["first_chunk_s"] is not None:
            log(f"Primer sonido a los {stats['first_chunk_s']:.1f}s")
        if stats["underrun_count"] > 0:
            log(f"Audio con {stats['underrun_count']} pausa(s) breve(s), "
                f"{stats['underrun_total_s']:.1f}s en total")
        log(f"Generado en {stats['wall_s']:.1f}s ({stats['audio_dur_s']:.1f}s de audio)")
    return full_wav, stats

def cargar(log=None):
    if log:
        log("Cargando modelo (cuantizando pesos)...")
    model, max_audio_tokens = load_model()
    if log:
        log("Modelo cargado. Cargando referencia de voz...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)
    warmup_pitch_shift()
    if log:
        log("Listo para generar.")
    return model, max_audio_tokens, gpt_cond_latent, speaker_embedding


def generar(ctx, texto, temperature=0.85, top_p=0.85, top_k=50, repetition_penalty=2.0, log=None):
    model, max_audio_tokens, gpt_cond_latent, speaker_embedding = ctx
    full_wav, _stats = synthesize(
        model, texto, max_audio_tokens, gpt_cond_latent, speaker_embedding,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, log=log,
    )

    return full_wav

"""
def main():
    print("\nCargando modelo, un momento...")
    model, max_audio_tokens = load_model()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)
    warmup_pitch_shift()  # paga el costo de compilacion JIT aca, no en la primera frase del usuario
    print(f"Escribe un texto y presiona Enter para generar audio.")
    print(f"(Enter vacio para terminar)\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            break

        try:
            _, stats = synthesize(model, text, max_audio_tokens, gpt_cond_latent, speaker_embedding)
            if stats["first_chunk_s"] is not None:
                print(f"  (primer sonido a los {stats['first_chunk_s']:.1f}s)")
            if stats["underrun_count"] > 0:
                print(
                    f"  (audio con {stats['underrun_count']} pausa(s) breve(s), "
                    f"{stats['underrun_total_s']:.1f}s en total -- el texto es largo para "
                    f"generarlo mas rapido que tiempo real en esta maquina)"
                )
        except Exception as e:  # noqa: BLE001
            print(f"  Error generando audio: {e}")

    print("\nCerrando.")
"""

if __name__ == "__main__":
    # main()
    lanzar_interfaz(cargar=cargar, generar=generar, sample_rate=SAMPLE_RATE)