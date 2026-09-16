# AppTTS-ModelXTTSv2

Este es un proyecto de síntesis de voz (Text-to-Speech) basado en el modelo XTTS v2 (coqui), con diferentes instrucciones y el código fuente para el despliegue de: una aplicación de escritorio (Windows/Linux) y una aplicación móvil (Android) que la inferencia (generación de audio) con el modelo TTS de manera local.

## Estructura

- **`VocesWindows/`**
  Generador de voz para Windows. Incluye un Python portable con todas las dependencias (PyTorch, TTS, ONNX Runtime, etc.) y un las indicaciones para compilar un ejecutable (`SintensisVoz.exe`), listo para usar sin instalar nada en el sistema. Contiene también los checkpoints entrenados (`models/`) y sus versiones exportadas a ONNX (`onnx_models/`).

- **`VocesLinux/`**
  Equivalente a `VocesWindows/` pero para Linux: Python portable con las mismas dependencias, checkpoints y modelos ONNX.

- **`VocesAndroid/`**
  App Android (Kotlin + C++) que ejecuta el modelo XTTS v2 exportado a ONNX directamente en el teléfono. Incluye el proyecto Gradle complón del checkpoint a los archivos que consumela app, y los modelos ya generados para una voz probada.

# AI Disclaimer

Este proyecto fue desarrollado con asistencia de inteligencia artificial (código, documentación y configuración). Puede contener errores, decisiones no óptimas o partes sin verificar exhaustivamente.