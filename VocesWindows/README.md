# Generador de Voz -- Windows

Esta carpeta trae **todo incluido**: Python 3.10 portable (`python/`, standalone,
sin instalar nada de Python en el sistema) + todas las dependencias exactas
(torch, onnxruntime, customtkinter, etc., mismas versiones que la build de
Linux) ya bajadas dentro de `python/Lib/site-packages/`.

A diferencia de la build de Linux, esta se armo y se probo en una maquina
Windows real -- `SintensisVoz.exe` ya viene compilado en esta carpeta, listo
para ejecutar sin pasos previos.

## 1. Compilar el lanzador (opcional -- ya viene compilado)

Solo hace falta si modificas `launcher.c`. Se necesita un `gcc` que genere
ejecutables de Windows -- por ejemplo el de [MSYS2](https://www.msys2.org/)
(`pacman -S mingw-w64-ucrt-x86_64-gcc`, usando la consola "UCRT64").

Con `make` instalado:

```bash
make
```

Si no tenes `make` (no viene por defecto en la mayoria de instalaciones de
MSYS2), compila directo con el mismo comando que usa el `Makefile`:

```bash
gcc -O2 -mwindows launcher.c -o SintensisVoz.exe
```

Esto genera `SintensisVoz.exe` en esta carpeta, usando `launcher.c` (rama
`_WIN32`): ubica su propio directorio, redirige stdout/stderr del proceso
Python a `infer.log`, y ejecuta `python\pythonw.exe -u infer.py` sin dejar
ninguna consola abierta.

## 2. Ejecutar

Doble click en `SintensisVoz.exe`, o desde una terminal:

```
.\SintensisVoz.exe
```

No hace falta instalar PortAudio aparte (a diferencia de Linux) -- las DLLs
de PortAudio para Windows ya vienen dentro de `python\Lib\site-packages\_sounddevice_data\`.

La ventana "Generador de Voz" deberia abrir. La primera vez tarda unos
segundos mas: esta cuantizando y cargando el modelo. Los `print()` del
proceso Python quedan en `infer.log`, junto al ejecutable (no hay consola
visible porque el lanzador usa `pythonw.exe`).

## Requisitos del sistema

- Windows 10 o 11, 64 bits (x86_64). El Python portable y las DLLs son de esa
  arquitectura -- no arm64 (Windows on ARM).
- Nada de Python, PortAudio ni ninguna dependencia instalada aparte: todo
  viene empaquetado en `python/`.
- `gcc` (MinGW-w64) solo si vas a recompilar el lanzador (paso 1) -- no hace
  falta para solo ejecutar la app.