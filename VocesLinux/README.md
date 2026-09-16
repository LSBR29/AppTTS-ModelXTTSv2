# Generador de Voz -- Linux

Esta carpeta trae **todo incluido**: Python 3.10 portable (`python/`, standalone,
sin instalar nada de Python en el sistema) + todas las dependencias exactas
(torch, onnxruntime, customtkinter, etc., mismas versiones que la build de
Windows) ya bajadas dentro de `python/lib/python3.10/site-packages/`.

Se armo desde una maquina Windows (bajando los wheels manylinux para Linux x86_64
directamente, sin usar ningun Linux real), asi que **no pudo probarse ejecutando
de verdad** -- revisa la seccion "Si algo falla" mas abajo.

## 1. Compilar el lanzador (una sola vez)

No se pudo compilar el binario desde Windows (hace falta un compilador que genere
ELF de Linux, y esta maquina solo tiene uno para Windows). Con `gcc` instalado
(viene de fabrica en casi cualquier distro con herramientas de desarrollo, o se
instala con `sudo apt install build-essential` / `sudo dnf groupinstall "Development Tools"`):

```bash
make
```

Esto genera `SintesisVoz` (sin extension) en esta carpeta, usando `launcher.c`
(rama POSIX): ubica su propio directorio, redirige stdout/stderr del proceso
Python a `infer.log`, y ejecuta `python/bin/python3 -u infer.py`.

## 2. Instalar PortAudio del sistema (una sola vez)

`sounddevice` (la libreria que reproduce el audio generado) **no trae PortAudio
embebido en Linux** -- a diferencia de Windows, ahi el sistema operativo
siempre necesita la libreria instalada aparte:

```bash
sudo apt install libportaudio2       # Debian/Ubuntu
sudo dnf install portaudio           # Fedora
sudo pacman -S portaudio             # Arch
```

Sin esto, `import sounddevice` va a fallar apenas se intente generar audio.

## 3. Ejecutar

```bash
chmod +x SintesisVoz python/bin/python3.10 python/bin/python3 python/bin/python
./SintesisVoz
```

(El `chmod +x` es necesario la primera vez si la carpeta se transfirio por zip o
por un medio que no preserva permisos de Unix -- por ejemplo, viene de una
maquina Windows.)

La ventana "Generador de Voz" deberia abrir igual que en Windows. Los
`print()` del proceso quedan en `infer.log`, junto al ejecutable.

## Requisitos del sistema

- **glibc >= 2.28** (Debian 10+, Ubuntu 20.04+, Fedora 29+, y practicamente
  cualquier distro de 2019 en adelante). El Python portable se compilo contra
  esa base; en distros mas viejas puede no arrancar.
- x86_64 (Intel/AMD de 64 bits). No arm64 (Raspberry Pi, etc.) -- para eso
  habria que rearmar el bundle con wheels `manylinux_..._aarch64`.
- `libportaudio2` (ver paso 2).
- `gcc` para compilar el launcher (paso 1).
- Entorno grafico con soporte X11/Wayland (es una app de escritorio con tkinter).