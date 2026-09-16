# Guía: de VocesWindows a la app Android

Esta guía cubre el camino completo, de punta a punta: desde el checkpoint entrenado (los archivos
que ya están en `VocesWindows/models/<voz>/`, resultado del entrenamiento en Kaggle) hasta tener
la app funcionando en un teléfono Android real, generando voz.

No requiere conocimiento previo del resto del proyecto. Todo lo que hace falta está en esta
carpeta (`VocesAndroid/`) más lo que ya trae `VocesWindows/` (el Python portable con PyTorch/
TTS/ONNX Runtime ya instalados, y el checkpoint entrenado).

---

## 0. Qué hay en esta carpeta

```
VocesAndroid/
  GUIA_ANDROID.md          <- esta guía
  export_scripts/          <- scripts Python: checkpoint -> archivos que carga la app
  app/src/                 <- código fuente de la app (Kotlin + C++)
  build.gradle.kts, settings.gradle.kts, gradle.properties, gradlew(.bat), gradle/
                            <- proyecto Gradle completo (compila la app)
  generated_models/        <- YA CONTIENE los archivos exportados para la voz "2025acv02"
                             (se puede regenerar con export_scripts/, ver paso 2)
```

No se incluyen `TTS/` ni el Python portable: `export_scripts/` los usa directamente desde
`VocesWindows/` (o `VocesLinux/`), que ya los tienen. No hace falta duplicarlos.

---

## 1. Requisitos (una sola vez por máquina de desarrollo)

### 1.1 `VocesWindows/` (o `VocesLinux/`)

Ya presente como carpeta hermana de `VocesAndroid/` -- de ahí se toman el checkpoint entrenado y
el Python portable con PyTorch/TTS/ONNX Runtime. No hay que instalar nada para esto.

### 1.2 JDK 17

Descarga: **[Eclipse Temurin 17](https://adoptium.net/temurin/releases/?version=17)** (elegir el
instalador `.msi` de Windows, arquitectura x64). Instalar con las opciones por defecto -- queda
en `C:\Program Files\Eclipse Adoptium\jdk-17.<version>-hotspot\` (la ruta exacta depende de la
versión que descargue el instalador en ese momento).

**El `java` que ya viene en el PATH del sistema puede ser una versión vieja** (Java 8 es común) --
Gradle necesita 17 específicamente para este proyecto. Confirmar con `java -version` antes de
compilar; si no da 17, apuntar `JAVA_HOME` a la carpeta del JDK recién instalado ANTES de correr
`gradlew` (ver paso 3, donde se explica el error exacto que da si esto falta).

### 1.3 Android SDK (command-line tools, sin Android Studio)

Descarga: **[Command line tools](https://developer.android.com/studio#command-tools)** (en la
página de Android Studio, bajar hasta "Command line tools only" y tomar el `.zip` de Windows).

Instalación:
1. Crear la carpeta `%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\` (a mano).
2. Descomprimir el `.zip` descargado -- adentro hay una carpeta `cmdline-tools\`; su contenido
   (`bin\`, `lib\`, etc.) va DIRECTO dentro de `latest\` (no la carpeta `cmdline-tools` en sí,
   sino lo que tiene adentro -- si queda anidado como `...\latest\cmdline-tools\bin\` no
   funciona).
3. Con eso ya alcanza: **no hace falta instalar manualmente** la plataforma 34, build-tools, NDK
   ni CMake -- Gradle los descarga solos la primera vez que hacen falta (acepta las licencias
   automáticamente al correr `gradlew`, ver paso 3). Si se prefiere instalarlos a mano de
   antemano (por ejemplo, en una máquina sin conexión durante la compilación):
   ```bash
   cd %LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin
   sdkmanager --licenses
   sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;27.0.12077973" "cmake;3.22.1"
   ```
4. Agregar `%LOCALAPPDATA%\Android\Sdk\platform-tools\` al PATH del sistema -- de ahí sale
   `adb`, que se usa en toda esta guía (pasos 5 y 6). Confirmar con `adb version` en una terminal
   nueva.

### 1.4 `local.properties`

En esta carpeta (`VocesAndroid/`), con la ruta al SDK (crear a mano, no viene incluido porque es
específico de cada máquina):
```
sdk.dir=C\:\\Users\\<usuario>\\AppData\\Local\\Android\\Sdk
```
(En Windows, cada `\` va doblado -- `\\` -- y `:` escapado como `\:`. Es sintaxis de archivo
`.properties` de Java, no una ruta de Windows normal.)

### 1.5 Teléfono Android real, arm64

**Requisitos mínimos de hardware** (verificado contra el APK compilado y midiendo en el
dispositivo real, no supuesto):
- **CPU arm64-v8a.** El APK solo trae binarios nativos para esa arquitectura -- en un teléfono
  sin ese soporte la instalación falla directo con `INSTALL_FAILED_NO_MATCHING_ABIS`. Casi todo
  equipo Android desde ~2017 cumple esto, pero no está de más confirmarlo.
- **Android 8.0 (API 26) o superior.**
- **~2.1GB de almacenamiento libre** (modelos ~2.0GB + APK ~38MB).
- **~2GB de RAM *disponible* en el momento de sintetizar** (no de RAM total del equipo). Medido
  en el dispositivo real: el pico no ocurre en la primera oración (~1.3GB) sino de la segunda en
  adelante, cuando la sesión de decode ya está cargada en memoria y una nueva sesión de prefill se
  crea al mismo tiempo -- pico real medido ≈1.76GB de proceso nativo, ≈1.9-2.0GB contando el
  overhead normal de Android encima. En equipos de gama baja (4GB de RAM total o menos) el sistema
  suele dejar bastante menos que eso libre -- riesgo real de que el sistema mate la app a mitad de
  una síntesis larga. No hace falta modificar nada para equipos con RAM disponible cómoda (6GB+ de
  RAM total); en equipos más ajustados, liberar también la sesión de decode entre oraciones (no
  solo la de prefill) bajaría el pico a ~1.3-1.4GB a costa de volver a pagar el costo de recrearla
  en cada oración -- es un cambio de código en `app/src/main/cpp/xtts_engine.cpp`, no de
  configuración.

Fuera de RAM en equipos muy ajustados, no hace falta tocar nada más para correr en otro teléfono
arm64 -- los hilos por sesión (ver sección 7) son configuración en tiempo de ejecución, no algo
compilado para un chip específico.

Pasos para preparar el teléfono:
1. Activar **Opciones de desarrollador**: Ajustes → Acerca del teléfono → tocar 7 veces seguidas
   "Versión de MIUI" (o "Número de compilación" en Android estándar).
2. Dentro de Opciones de desarrollador, activar **"Depuración USB"** y, específicamente en
   Xiaomi/MIUI, también **"Instalar vía USB"** -- si falta esta segunda, la instalación falla con
   `INSTALL_FAILED_USER_RESTRICTED` aunque la depuración USB esté activa.
3. Conectar el teléfono por USB y aceptar el diálogo "¿Permitir depuración USB?" que aparece en
   pantalla la primera vez.
4. Confirmar con `adb devices` (requiere el PATH del paso 1.3.4) -- debe listar el teléfono como
   `device` (no `unauthorized`; si dice eso, revisar el diálogo del paso 3 en la pantalla del
   teléfono).

---

## 2. Del checkpoint a los archivos que carga la app

Dos scripts, ambos en `export_scripts/`. Por defecto apuntan a la voz `2025acv02` de
`VocesWindows/models/`; para otra voz, pasar las rutas explícitas (`--checkpoint`, `--config`,
`--vocab`, `--ref-wav`, `--outdir` -- ver `--help` de cada script).

### 2.1 Frontend de texto (tokenizador)

No necesita el checkpoint ni PyTorch, solo `vocab.json`:

```bash
cd export_scripts
python dump_text_frontend.py
```

Genera en `generated_models/`: `vocab.tsv`, `merges.tsv`, `special_tokens.tsv`.

### 2.2 GPT + conditioning latents

Este sí necesita el Python portable de VocesWindows (trae PyTorch/transformers/coqui-TTS ya
instalados con las versiones exactas que requiere el checkpoint):

```bash
cd export_scripts
..\..\VocesWindows\python\python.exe export_gpt_onnx.py
```

Tarda unos minutos (carga un checkpoint de ~1.8GB y exporta dos grafos ONNX: `gpt_prefill.onnx`,
~1.3GB, y `gpt_decode_step.onnx`, ~640MB -- ver sección 7 sobre por qué tienen precisión distinta).
Al final valida el resultado contra el modelo real y debe imprimir:

```
Validacion prefill fp32 vs. modelo real: max_abs_err logits=... hidden=... -> OK
Validacion decode fp16 vs. modelo real: max_abs_err logits=... hidden=... -> OK
```

Si alguna dice `FALLO`, el script termina con error -- no seguir con archivos de un export que no
validó. Genera en `generated_models/`: `gpt_prefill.onnx`, `gpt_decode_step.onnx`,
`cond_latent.bin`, `speaker_embedding.bin`.

### 2.3 Vocoder (HiFiGAN)

No hace falta exportarlo: ya existe, pre-exportado, en
`VocesWindows/onnx_models/<voz>/hifigan_decoder.onnx`. Copiarlo tal cual:

```bash
copy ..\..\VocesWindows\onnx_models\2025acv02\hifigan_decoder.onnx generated_models\
```

### 2.4 Resultado

`generated_models/` debe tener exactamente estos 8 archivos (los mismos 8 que la app busca en
el paso 5):

```
vocab.tsv  merges.tsv  special_tokens.tsv
cond_latent.bin  speaker_embedding.bin
gpt_prefill.onnx  gpt_decode_step.onnx  hifigan_decoder.onnx
```

Tamaño total: ~2.0GB (el GPT sin cuantizar domina el peso -- ver la sección 7 sobre la precisión
usada en cada grafo).

---

## 3. Compilar la app

Primero confirmar que `java -version` da 17 en esta terminal. Si no (por ejemplo, muestra una
versión 1.8.x), fijar `JAVA_HOME` al JDK 17 del paso 1.2 ANTES de compilar -- solo dura mientras
esa ventana de terminal esté abierta, hay que repetirlo en cada terminal nueva:

```bat
:: cmd.exe
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
gradlew.bat assembleDebug
```

```powershell
# PowerShell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot"
.\gradlew.bat assembleDebug
```

(`./gradlew assembleDebug` en Linux/macOS, si se arma la carpeta ahí -- ver el `README.md` de
`VocesLinux` para el patrón de compilar el lanzador nativo en Linux; para la app Android da lo
mismo el sistema operativo de escritorio, Gradle es multiplataforma.)

Genera `app/build/outputs/apk/debug/app-debug.apk` (~38MB -- los modelos NO van dentro del APK,
ver paso 4). `BUILD SUCCESSFUL` confirma que compiló.

**Si falla con algo como** `Could not resolve com.android.tools.build:gradle:... > ... compatible
with Java 8 ... consumer needed ... compatible with Java 8` (u otro mensaje de variantes
incompatibles mencionando "Java 8"): es exactamente el problema de arriba, `JAVA_HOME` sigue sin
apuntar al JDK 17 -- no es un error del código que se haya editado, es puramente de entorno. La
otra causa común de falla es `local.properties` con una ruta mal escapada (paso 1.4).

---

## 4. Por qué los modelos no van dentro del APK

`generated_models/` pesa ~2.0GB. Un APK no puede incluir eso como "asset" (Android los
extrae en la instalación, duplicando el tamaño en el teléfono) ni tiene sentido para una tienda
de aplicaciones. La app espera los modelos en su almacenamiento privado externo
(`Android/data/voces.tcu748/files/xtts_models/` dentro del almacenamiento del
teléfono), copiados aparte -- ver paso 5.2. Un mecanismo de descarga automática al primer
inicio (en vez de copiarlos a mano) es una mejora pendiente, no implementada todavía.

---

## 5. Instalar y cargar el modelo en el teléfono

### 5.1 Instalar la app

```bash
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Si falla con `INSTALL_FAILED_USER_RESTRICTED`: revisar que "Instalar vía USB" esté habilitado
en Opciones de desarrollador del teléfono (Xiaomi/MIUI lo pide aparte de la depuración USB).

### 5.2 Copiar los modelos

**Importante, en este orden exacto** -- si se empuja el modelo ANTES de que la app haya
arrancado al menos una vez, la app no lo va a poder leer (ver "Problemas conocidos" más abajo):

```bash
:: 1. Abrir la app una vez (para que cree su propia carpeta de almacenamiento externo)
adb shell am start -n voces.tcu748/.MainActivity

:: 2. Recién ahora, copiar los 8 archivos de generated_models/
adb push generated_models\vocab.tsv generated_models\merges.tsv generated_models\special_tokens.tsv generated_models\cond_latent.bin generated_models\speaker_embedding.bin generated_models\hifigan_decoder.onnx /storage/emulated/0/Android/data/voces.tcu748/files/xtts_models/
adb push generated_models\gpt_prefill.onnx /storage/emulated/0/Android/data/voces.tcu748/files/xtts_models/
adb push generated_models\gpt_decode_step.onnx /storage/emulated/0/Android/data/voces.tcu748/files/xtts_models/

:: 3. Reiniciar la app para que vuelva a chequear si los modelos ya estan
adb shell am force-stop voces.tcu748
adb shell am start -n voces.tcu748/.MainActivity
```

Los dos archivos del GPT son grandes (`gpt_prefill.onnx` ~1.3GB, `gpt_decode_step.onnx` ~640MB) y
tardan lo que tarde la transferencia USB/red -- normal que cada `adb push` de esos dos tome varios
segundos a medio minuto.

---

## 6. Usar la app

La pantalla tiene un título, una caja de texto (español), un botón "SINTETIZAR", y un área de
estado debajo.

1. Al abrir, si los modelos están en su lugar, la app empieza a cargarlos automáticamente en
   segundo plano (área de estado: **"Cargando modelo..."**, botón deshabilitado mientras tanto).
   Cuando termina, dice **"Modelo cargado. Listo para sintetizar."** y el botón se habilita. Si
   los modelos no están, dice dónde los espera y qué archivos faltan (no intenta cargar nada).
2. Escribir o editar el texto (puede tener varias oraciones).
3. Tocar **SINTETIZAR**. Como el modelo ya se cargó al abrir la app (paso 1), la inferencia
   empieza de inmediato -- no hay que esperar la carga en este momento. El área de estado va
   mostrando, oración por oración, cuántos tokens de audio generó cada una y cuánto tardó. El
   audio se reproduce por el parlante/audífonos a medida que cada oración termina de generarse --
   no hay que esperar a que termine todo el texto para escuchar la primera oración.
4. Mientras sintetiza, toda la inferencia corre en un hilo de fondo (la pantalla no se congela).
   Si se sale de la app (botón atrás/inicio) a mitad de una síntesis, se libera la sesión de
   decode (la que queda viva entre oraciones, ~640MB de RAM) -- al volver a abrir la app se
   vuelve a cargar automáticamente, igual que en el paso 1.

**Primera apertura esperable**: la carga automática de modelos (paso 1) tarda varios segundos
(lee ~2GB de almacenamiento y crea las sesiones de inferencia) -- es el momento en que conviene
esperar antes de una demostración, no después de tocar SINTETIZAR. Ver la sección de rendimiento
más abajo para los tiempos de generación en sí.

---

## 7. Decisiones importantes que hay que conocer para no romper nada

- **El GPT usa precisión distinta en cada uno de sus dos grafos: `gpt_prefill.onnx` en fp32
  completo, `gpt_decode_step.onnx` en precisión mixta (fp16 en las proyecciones de atención y del
  MLP, fp32 en el resto).** No es una elección arbitraria, sale de medir en el teléfono real:
  - Cuantización a int8 (cualquier variante) produce audio inutilizable -- rompe específicamente
    en el bloque MLP de cada capa, sin importar cómo se agrupe o dónde se aplique. Descartada por
    completo, no se usa en ningún grafo.
  - fp16 nativo (no una conversión posterior del grafo fp32, sino pesos en fp16 desde el
    entrenamiento hasta el export) sí funciona bien en calidad -- pero solo conviene en
    velocidad para el paso de decode (un token a la vez). Para el prefill (que procesa todo el
    texto de una sola pasada, varios tokens a la vez) fp16 resulta más LENTO que fp32 en este
    hardware -- por eso el prefill se deja en fp32 completo.
  - Dentro del grafo de decode, el cálculo de atención en sí (no las proyecciones de pesos) y el
    stream residual entre capas se mantienen siempre en fp32 -- necesario para no perder
    precisión de forma acumulada a través de las 25 capas y los muchos pasos autorregresivos de
    una síntesis real.
  - Resultado medido: el paso de decode es ~25% más rápido que la versión 100% fp32, sin pérdida
    de calidad perceptible.
- **La sesión de prefill (`gpt_prefill.onnx`, ~1.3GB de pesos) se crea, se usa una vez y se
  destruye por cada oración -- nunca queda viva de forma permanente.** Mantenerla cargada todo el
  tiempo junto con la de decode empuja el pico de RAM cerca del límite disponible en el teléfono
  (~1.8GB libres en uso normal). El código de la app (`xtts_engine.cpp`) ya hace esto
  correctamente -- no "optimizar" manteniendo ambas sesiones vivas para ahorrar el tiempo de
  recrear el prefill, el riesgo de memoria no vale la pena.
- **La sesión de decode usa 1 hilo (`intra_op_num_threads`), no 4.** Medido en el teléfono real
  con los pesos reales: el bucle de decodificación (un token por vez) es memory-bound y con más
  hilos anda MÁS LENTO (143ms/paso con 1 hilo vs. 272ms/paso con 4). El prefill y el HiFiGAN sí
  se benefician de más hilos (van con 4). Esto está en `MainActivity.kt`, en la llamada a
  `NativeBridge.nativeInit(...)` -- no cambiarlo sin volver a medir.
- **La segmentación de texto en oraciones es una decisión de UX, no un requisito de fidelidad**
  al modelo original -- el separador (`app/src/main/cpp/sentence_split.hpp`) es una
  implementación propia y simple (corta en `. ! ?`, con un tope duro de 239 caracteres), no una
  réplica de ningún componente del entrenamiento.

---

## 8. Rendimiento actual (medido) -- para calibrar expectativas

Medido en un Xiaomi Redmi Note 14 Pro+ 5G real, con una frase de prueba corta: generar el audio
toma alrededor de 4x la duración del audio resultante (RTF ≈4, es decir, más lento que tiempo
real). El paso de decode (un token de audio a la vez, la parte que más pesa en el tiempo total)
promedia ~140ms por paso; el prefill (una sola pasada por oración) toma menos de un segundo.

El decode en precisión mixta (sección 7) es ~25% más rápido que la versión anterior 100% fp32,
sin cambio perceptible en la calidad del audio. Sigue siendo el componente que más tiempo toma
del pipeline completo -- una optimización mayor de velocidad (más allá de precisión numérica)
requeriría un cambio estructural del modelo (por ejemplo, menos capas), lo cual exige
reentrenamiento y no está contemplado en este entregable.

---

## 9. Problemas conocidos y su solución (encontrados de verdad, instalando en el teléfono real)

**La app se cierra sola al abrir, con "Application Error" en pantalla.**
Causa (si se modificó el tema en `AndroidManifest.xml`): `MainActivity` necesita un tema
`Theme.AppCompat.*` -- un tema `Theme.Material.*` normal de Android la hace crashear al
arrancar. Ya está puesto correctamente en el manifest que trae esta carpeta; si se cambia, hay
que mantenerlo dentro de la familia AppCompat.

**La app dice "Modelos no encontrados" aunque `adb push` haya terminado sin error.**
Causa: si se empujaron los archivos ANTES de abrir la app por primera vez (con
`adb shell mkdir` en vez de dejar que la app cree su propia carpeta), el directorio queda
"propiedad" del usuario `shell` de ADB, no de la app, y la vista de almacenamiento aislada de
la app no lo puede leer (aunque `adb shell ls` sí lo muestre). Solución: seguir el orden del
paso 5.2 -- abrir la app primero, empujar los archivos después.

**No se puede tocar el botón con `adb shell input tap` para probar sin tocar el teléfono.**
En MIUI, inyectar eventos de toque por ADB da `SecurityException` aunque la app esté instalada y
depurable -- es una restricción de seguridad de MIUI, no un bug de la app. Hay que tocar la
pantalla físicamente para probar.

---

## 10. Qué falta (no implementado todavía)

- Mecanismo real de descarga de modelos al primer inicio (hoy se copian a mano con `adb push`).
- Benchmark formal de rendimiento (RTF por componente, latencia al primer audio, batería, uso de
  memoria pico) sobre una serie de frases de prueba.
- Streaming de audio totalmente solapado entre oraciones largas (hoy la oración N+1 ya empieza a
  generarse mientras suena la N, pero si la generación es más lenta que la reproducción -- como
  es el caso ahora mismo, ver punto 8 -- puede haber un corte breve entre una oración y la
  siguiente).
- Verificar la licencia del modelo (Coqui Public Model License 1.0.0, no comercial explícito)
  contra el uso final que se le vaya a dar a la app.
