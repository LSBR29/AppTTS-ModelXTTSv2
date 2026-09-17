# AppTTS-ModelXTTSv2

Este es un proyecto de síntesis de voz (Text-to-Speech) basado en el modelo XTTS v2 (coqui), con diferentes instrucciones y el código fuente para el despliegue de: una aplicación de escritorio (Windows/Linux) y una aplicación móvil (Android) que la inferencia (generación de audio) con el modelo TTS de manera local.

## Estructura

- **`VocesWindows/`**
  Generador de voz para Windows. Incluye un Python portable con todas las dependencias (PyTorch, TTS, ONNX Runtime, etc.) y un las indicaciones para compilar un ejecutable (`SintensisVoz.exe`), listo para usar sin instalar nada en el sistema. Solo se deben guardar los checkpoints entrenados (`models/`) y sus versiones exportadas a ONNX (`onnx_models/`).

- **`VocesLinux/`**
  Equivalente a `VocesWindows/` pero para Linux: Python portable con las mismas dependencias, checkpoints y modelos ONNX.

- **`VocesAndroid/`**
  App Android (Kotlin + C++) que ejecuta el modelo XTTS v2 exportado a ONNX directamente en el teléfono. Incluye el proyecto Gradle completo del checkpoint en los archivos que consumela app, y los modelos ya generados para una voz probada.

## Preview de las Apps

<div align="center">
  
| Escritorio | Android |
|:--:|:--:|
| <img width="340" height="380" alt="Aplicación de Escritorio" src="https://github.com/user-attachments/assets/3c6b696f-ffc6-41bf-8109-120b3f10f037" /> | <img width="340" height="510" alt="Aplicación Móvil" src="https://github.com/user-attachments/assets/ceab3281-b716-4ecb-9c76-b93599de50ff" /> |

</div>

## Modelos ya entrenados

Puede encontrarlos en la siguiente carpeta: [OneDrive](https://6f33fa7f78ea46e2aaca-my.sharepoint.com/:f:/g/personal/luis_brenesruiz_ucr_ac_cr/IgA_fHN6DcnzTq30IUfhQH8CATcJdq6fODv1tks2GW-3qeI?e=JOUh1M)

# AI Disclaimer

Este proyecto fue desarrollado con asistencia de inteligencia artificial (código, documentación y configuración). Puede contener errores, decisiones no óptimas o partes sin verificar exhaustivamente.
