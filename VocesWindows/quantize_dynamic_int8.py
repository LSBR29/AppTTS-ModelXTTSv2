"""
Cuantizacion dinamica int8 del GPT autoregresivo de XTTS (domina ~79% del tiempo de generacion).

El GPT2 de `transformers` implementa las proyecciones de atencion y MLP con
`transformers.pytorch_utils.Conv1D`, NO con `nn.Linear` -- `quantize_dynamic` solo cubre
`nn.Linear`/RNN, asi que primero se convierte cada `Conv1D` a un `nn.Linear` equivalente
in-place antes de cuantizar (ver OPTIMIZATION_LOG.md del proyecto original para el detalle
completo de esta decision).
"""

import time

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


def conv1d_to_linear(conv: Conv1D) -> nn.Linear:
    in_features, out_features = conv.weight.shape
    linear = nn.Linear(in_features, out_features, bias=conv.bias is not None)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.t())
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)
    return linear


def replace_conv1d_with_linear(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            setattr(module, name, conv1d_to_linear(child))
            count += 1
        else:
            count += replace_conv1d_with_linear(child)
    return count


def count_quantized_linears(module: nn.Module) -> int:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    return sum(1 for m in module.modules() if isinstance(m, DynamicQuantizedLinear))


def build_int8_dynamic_model(checkpoint_path, config_path, vocab_path):
    """Carga el modelo Xtts y cuantiza dinamicamente (int8) su GPT in-place. Devuelve (model, timings)."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    t0 = time.perf_counter()
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts(config)
    load_t0 = time.perf_counter()
    model.load_checkpoint(
        config, checkpoint_path=str(checkpoint_path), vocab_path=str(vocab_path), use_deepspeed=False
    )
    load_t1 = time.perf_counter()
    model.eval()

    same_object = model.gpt.gpt is model.gpt.gpt_inference.transformer
    if not same_object:
        raise RuntimeError(
            "El transformer usado en generate() no es el mismo objeto que se va a cuantizar; "
            "la cuantizacion no tendria efecto en la ruta de inferencia real. Abortando."
        )

    n_conv1d = replace_conv1d_with_linear(model.gpt.gpt)
    quant_t0 = time.perf_counter()
    torch.ao.quantization.quantize_dynamic(model.gpt, {nn.Linear}, dtype=torch.qint8, inplace=True)
    quant_t1 = time.perf_counter()

    n_quant = count_quantized_linears(model.gpt)
    if n_quant < n_conv1d:
        raise RuntimeError(
            f"Solo se cuantizaron {n_quant} de {n_conv1d} capas Linear esperadas -- "
            "la cuantizacion no se aplico como se esperaba, abortando antes de medir."
        )

    timings = {
        "config_init_s": load_t0 - t0,
        "checkpoint_load_s": load_t1 - load_t0,
        "conv1d_to_linear_count": n_conv1d,
        "quantized_linear_count": n_quant,
        "quantize_dynamic_s": quant_t1 - quant_t0,
    }
    return model, timings
