package voces.tcu748

import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * App de demostración: una pantalla, sin arquitectura MVVM. El motor (tokenizador + HiFiGAN +
 * sesión de decode del GPT) se precarga automáticamente al abrir la app, para que la primera vez
 * que se toca SINTETIZAR la inferencia empiece de inmediato, sin pagar el costo de carga de
 * modelos en ese momento. Los modelos NO se empaquetan en el APK (pesan ~2GB) -- se esperan en el
 * almacenamiento externo privado de la app, empujados con `adb push`.
 *
 * Toda la interfaz se construye aca mismo, en código (no hay archivos de layout XML) -- ver los
 * bloques marcados "PERSONALIZAR" en onCreate() para los puntos mas comunes de edición: título,
 * colores, texto de ejemplo, tamaños.
 */
class MainActivity : AppCompatActivity() {

    private val mainHandler = Handler(Looper.getMainLooper())
    private var engineHandle: Long = 0
    private val isSynthesizing = AtomicBoolean(false)
    private lateinit var inputText: EditText
    private lateinit var synthButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var progressLabel: TextView

    // Registro detallado (lo que antes se veía siempre en pantalla) -- ahora solo vive en la
    // subsección "Debug" del diálogo de ajustes (ver showSettingsDialog()). Se sigue acumulando
    // en segundo plano aunque el diálogo esté cerrado; `debugLogView` es la referencia al
    // TextView del diálogo SI está abierto en este momento (null si no), para poder actualizarlo
    // en vivo mientras se sintetiza con el diálogo abierto.
    private val logBuilder = StringBuilder()
    private var debugLogView: TextView? = null

    // Parametros de muestreo del GPT -- ajustables desde el botón de ajustes (⚙), ver
    // showSettingsDialog(). Valores por defecto = los de producción (ver xtts_engine.cpp).
    // @Volatile: se escriben en el hilo principal (diálogo de ajustes) y se leen desde el hilo
    // productor de synthesizeAndPlay() -- sin esto no hay garantía de que ese hilo vea el cambio.
    @Volatile private var temperature = 0.85f
    @Volatile private var topP = 0.85f
    @Volatile private var topK = 50
    @Volatile private var repetitionPenalty = 2.0f

    private val modelDir: File
        get() = File(getExternalFilesDir(null), "xtts_models")

    private fun dp(value: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics,
    ).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // PERSONALIZAR: color de acento del botón (y podría reutilizarse para otros detalles).
        val accent = Color.parseColor("#3F51B5")

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            // PERSONALIZAR: color de fondo de toda la pantalla.
            setBackgroundColor(Color.parseColor("#FAFAFA"))
            setPadding(dp(24), dp(32), dp(24), dp(24))
        }

        val title = TextView(this).apply {
            // PERSONALIZAR: título mostrado en pantalla (independiente del nombre de la app).
            text = "VocesTCU748"
            textSize = 26f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#212121"))
            gravity = Gravity.CENTER
        }
        val settingsButton = TextView(this).apply {
            text = "⚙"
            textSize = 18f
            gravity = Gravity.CENTER
            setTextColor(Color.parseColor("#616161"))
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#ECECEC"))
                shape = GradientDrawable.OVAL
            }
            isClickable = true
            isFocusable = true
            setOnClickListener { showSettingsDialog() }
        }
        val headerFrame = FrameLayout(this)
        val settingsButtonParams = FrameLayout.LayoutParams(dp(36), dp(36)).apply {
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
        }
        headerFrame.addView(title, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER,
        ))
        headerFrame.addView(settingsButton, settingsButtonParams)

        val subtitle = TextView(this).apply {
            // PERSONALIZAR: subtítulo/descripción corta debajo del título.
            text = "Sistema de Comunicación Alternativa"
            textSize = 13f
            setTextColor(Color.parseColor("#757575"))
            gravity = Gravity.CENTER
            setPadding(0, dp(4), 0, dp(28))
        }

        inputText = EditText(this).apply {
            hint = "Escriba aquí"
            // PERSONALIZAR: texto de ejemplo que aparece precargado al abrir la app.
            minLines = 3
            gravity = Gravity.TOP or Gravity.START
            setPadding(dp(16), dp(16), dp(16), dp(16))
            setTextColor(Color.parseColor("#212121"))
            setHintTextColor(Color.parseColor("#9E9E9E"))
            background = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), Color.parseColor("#DDDDDD"))
            }
        }
        val inputParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(16) }

        synthButton = Button(this).apply {
            // PERSONALIZAR: texto del botón.
            text = "SINTETIZAR"
            setTextColor(Color.WHITE)
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = GradientDrawable().apply {
                setColor(accent)
                cornerRadius = dp(24).toFloat()
            }
            setPadding(0, dp(16), 0, dp(16))
        }
        val buttonParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(20) }

        // Barra de progreso chica -- reemplaza el log largo que antes se veía siempre en
        // pantalla (ese detalle ahora vive en Ajustes → Debug, ver showSettingsDialog()). Solo
        // busca responder "¿cuánto falta?": indeterminada mientras se carga el motor, determinada
        // por fragmento mientras genera (no hay progreso fino dentro de un fragmento sin cambiar
        // el motor nativo para reportarlo paso a paso -- granularidad por fragmento alcanza aca).
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = false
            max = 1
            progress = 0
        }
        val progressBarParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
        ).apply { topMargin = dp(4) }
        progressLabel = TextView(this).apply {
            text = ""
            textSize = 12f
            setTextColor(Color.parseColor("#757575"))
            setPadding(0, dp(4), 0, 0)
        }

        root.addView(headerFrame, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
        ))
        root.addView(subtitle)
        root.addView(inputText, inputParams)
        root.addView(synthButton, buttonParams)
        root.addView(progressBar, progressBarParams)
        root.addView(progressLabel)
        setContentView(root)

        synthButton.setOnClickListener { onSynthesizeClicked() }

        // mkdirs() aca (no solo un chequeo de existencia) importa: si el directorio ya existe
        // porque lo creo "adb shell mkdir" (el usuario shell, no la app) para pruebas, la vista
        // de almacenamiento aislada de la propia app no lo reconoce como propio y da Permission
        // denied al leerlo -- crearlo desde la app misma lo registra con el dueno correcto.
        modelDir.mkdirs()

        if (!modelsPresent()) {
            progressLabel.text = "Modelos no encontrados -- ver Ajustes → Debug"
            resetLog(
                "Modelos no encontrados en:\n${modelDir.absolutePath}\n\n" +
                    "Empujar con adb push (vocab.tsv, merges.tsv, special_tokens.tsv, cond_latent.bin, " +
                    "speaker_embedding.bin, gpt_prefill.onnx, gpt_decode_step.onnx, hifigan_decoder.onnx).\n"
            )
        } else {
            setButtonEnabled(false)  // se rehabilita cuando warmupEngine() termina de cargar
            Thread { warmupEngine() }.start()
        }
    }

    private fun modelsPresent(): Boolean {
        val required = listOf(
            "vocab.tsv", "merges.tsv", "special_tokens.tsv", "cond_latent.bin",
            "speaker_embedding.bin", "gpt_prefill.onnx", "gpt_decode_step.onnx", "hifigan_decoder.onnx",
        )
        return required.all { File(modelDir, it).exists() }
    }

    /**
     * Ventana de ajustes (botón ⚙ arriba a la derecha): permite editar los 4 parámetros de
     * muestreo del GPT que en producción son fijos (`xtts_engine.cpp`, struct `SamplingParams`) --
     * temperature, top_p, top_k, repetition_penalty. Los valores quedan en memoria (variables de
     * la Activity) y se usan en la próxima síntesis; no se guardan en disco, así que se resetean
     * a los valores por defecto al reabrir la app.
     */
    private fun showSettingsDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(8), dp(24), dp(0))
        }

        fun addRow(label: String, initialValue: String, decimal: Boolean): EditText {
            container.addView(TextView(this).apply {
                text = label
                textSize = 12f
                setTextColor(Color.parseColor("#757575"))
                setPadding(0, dp(12), 0, dp(4))
            })
            val edit = EditText(this).apply {
                setText(initialValue)
                inputType = if (decimal) {
                    InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
                } else {
                    InputType.TYPE_CLASS_NUMBER
                }
            }
            container.addView(edit)
            return edit
        }

        val temperatureEdit = addRow("Temperature (0.05 - 3.0)", temperature.toString(), decimal = true)
        val topPEdit = addRow("Top-p (0.01 - 1.0)", topP.toString(), decimal = true)
        val topKEdit = addRow("Top-k (1 - 1026)", topK.toString(), decimal = false)
        val repetitionPenaltyEdit = addRow("Repetition penalty (1.0 - 5.0)", repetitionPenalty.toString(), decimal = true)

        // Subsección Debug: el registro detallado que antes se veía siempre en la pantalla
        // principal -- se sigue acumulando en segundo plano aunque este diálogo esté cerrado
        // (ver postStatus/resetLog); acá solo se muestra. Se actualiza en vivo si el diálogo
        // sigue abierto mientras corre una síntesis (debugLogView no-null mientras tanto).
        container.addView(TextView(this).apply {
            text = "DEBUG"
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#9E9E9E"))
            setPadding(0, dp(20), 0, dp(6))
        })
        val debugText = TextView(this).apply {
            text = logBuilder.toString()
            textSize = 12f
            setTextColor(Color.parseColor("#616161"))
            setLineSpacing(dp(2).toFloat(), 1f)
        }
        debugLogView = debugText
        val debugScroll = ScrollView(this).apply { addView(debugText) }
        container.addView(
            debugScroll,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(150)),
        )

        AlertDialog.Builder(this)
            .setTitle("Ajustes de generación")
            .setView(container)
            .setPositiveButton("Guardar") { _, _ ->
                temperature = temperatureEdit.text.toString().toFloatOrNull()?.coerceIn(0.05f, 3.0f) ?: temperature
                topP = topPEdit.text.toString().toFloatOrNull()?.coerceIn(0.01f, 1.0f) ?: topP
                topK = topKEdit.text.toString().toIntOrNull()?.coerceIn(1, 1026) ?: topK
                repetitionPenalty = repetitionPenaltyEdit.text.toString().toFloatOrNull()?.coerceIn(1.0f, 5.0f)
                    ?: repetitionPenalty
                postStatus(
                    "Ajustes: temperature=$temperature top_p=$topP top_k=$topK " +
                        "repetition_penalty=$repetitionPenalty\n"
                )
            }
            .setNeutralButton("Predeterminados") { _, _ ->
                temperature = 0.85f
                topP = 0.85f
                topK = 50
                repetitionPenalty = 2.0f
                postStatus("Ajustes restablecidos a los valores por defecto.\n")
            }
            .setNegativeButton("Cancelar", null)
            .setOnDismissListener { debugLogView = null }
            .show()
    }

    /**
     * Se corre una sola vez, automáticamente, al abrir la app (ver onCreate) -- así la carga de
     * modelos (sesión de decode del GPT incluida, la más pesada) ya terminó para cuando el
     * usuario toca SINTETIZAR por primera vez, y esa primera síntesis empieza a generar audio de
     * inmediato en vez de tener que cargar el motor en ese momento. Se logra con una síntesis
     * descartable de una frase mínima: usa exactamente el mismo camino de código que una síntesis
     * real (`nativeInit` + `nativeSynthSentence`, que internamente crea la sesión de decode si
     * todavía no existe), simplemente sin reproducir el audio resultante.
     */
    private fun warmupEngine() {
        if (!isSynthesizing.compareAndSet(false, true)) return
        setButtonEnabled(false)
        mainHandler.post { resetLog("Cargando modelo...\n") }
        updateProgress("Cargando modelo…", indeterminate = true)
        try {
            if (engineHandle == 0L) {
                engineHandle = NativeBridge.nativeInit(modelDir.absolutePath, 4, 1, 4)
            }
            val outInfo = IntArray(2)
            NativeBridge.nativeSynthSentence(
                engineHandle, "Hola.", 1234, temperature, topP, topK, repetitionPenalty, outInfo,
            )
            postStatus("Modelo cargado. Listo para sintetizar.\n")
            updateProgress("Listo para sintetizar", indeterminate = false, progress = 1, max = 1)
        } catch (e: Exception) {
            postStatus("Error cargando el modelo: ${e.message}\n")
            updateProgress("Error al cargar el modelo -- ver Ajustes → Debug", indeterminate = false)
        } finally {
            isSynthesizing.set(false)
            setButtonEnabled(true)
        }
    }

    private fun onSynthesizeClicked() {
        if (!modelsPresent()) {
            resetLog("Faltan modelos en ${modelDir.absolutePath}\n")
            updateProgress("Faltan modelos -- ver Ajustes → Debug", indeterminate = false)
            return
        }
        if (!isSynthesizing.compareAndSet(false, true)) return  // ya hay una sintesis en curso

        val text = inputText.text.toString()
        setButtonEnabled(false)
        resetLog("Sintetizando...\n")
        updateProgress("Preparando…", indeterminate = true)

        // Toda la inferencia (carga del motor si hace falta, split de oraciones, GPT, HiFiGAN)
        // corre en este hilo de fondo -- nunca en el hilo principal.
        Thread {
            try {
                synthesizeAndPlay(text)
            } catch (e: Exception) {
                postStatus("Error: ${e.message}")
                updateProgress("Error -- ver Ajustes → Debug", indeterminate = false)
            } finally {
                isSynthesizing.set(false)
                setButtonEnabled(true)
            }
        }.start()
    }

    /**
     * Productor/consumidor en hilos separados (mismo patrón que `infer_repl.py`, la referencia
     * de producción): el hilo PRODUCTOR sintetiza el fragmento N+1 MIENTRAS el hilo CONSUMIDOR
     * todavía está reproduciendo el fragmento N, en vez de sintetizar-luego-reproducir en serie
     * (que dejaba un hueco de silencio entre fragmentos). `FloatArray(0)` como centinela de fin de
     * cola: `nativeSynthSentence` nunca devuelve un array vacío para un fragmento real (siempre
     * hay al menos 1 token de audio antes de poder parar, ver xtts_engine.cpp), así que es un
     * marcador seguro sin necesitar una cola de nullables.
     *
     * Buffer de anticipo (`lookahead`): la generación es más lenta que la reproducción (RTF>1,
     * ver GUIA_ANDROID.md), así que si se empieza a reproducir apenas el primer fragmento está
     * listo, tarde o temprano la reproducción alcanza a la generación y aparece un hueco de
     * silencio -- el más notorio es justo el primero, apenas termina el fragmento 1. Esperar a
     * tener `lookahead` fragmentos listos antes de arrancar la reproducción (en vez de solo 1)
     * elimina ese primer hueco a cambio de un primer sonido un poco menos inmediato -- combinado
     * con fragmentos más cortos (ver `nativeSplitSentences`, ahora corta cada ~100 caracteres en
     * vez de 239), el costo neto en tiempo hasta el primer sonido es chico. No elimina huecos más
     * adelante en textos largos (para eso haría falta generar más rápido que tiempo real, que no
     * es el objetivo aca), pero sí los hace menos frecuentes.
     */
    private fun synthesizeAndPlay(text: String) {
        if (engineHandle == 0L) {
            postStatus("Cargando motor (tokenizador + HiFiGAN)...\n")
            updateProgress("Cargando motor…", indeterminate = true)
            // prefill=4, decode=1, hifigan=4: hilos por sesion, medidos en el dispositivo real.
            engineHandle = NativeBridge.nativeInit(modelDir.absolutePath, 4, 1, 4)
        }

        val sentences = NativeBridge.nativeSplitSentences(text)
        postStatus("${sentences.size} fragmento(s).\n")
        updateProgress("Generando 0/${sentences.size}…", indeterminate = false, progress = 0, max = sentences.size)

        val queue = LinkedBlockingQueue<FloatArray>()
        var producerError: Exception? = null

        val producer = Thread {
            try {
                for ((i, sentence) in sentences.withIndex()) {
                    val t0 = System.currentTimeMillis()
                    val outInfo = IntArray(2)
                    val wav = NativeBridge.nativeSynthSentence(
                        engineHandle, sentence, 1234, temperature, topP, topK, repetitionPenalty, outInfo,
                    )
                    val elapsed = System.currentTimeMillis() - t0
                    val stoppedNaturally = outInfo[1] == 1
                    postStatus(
                        "[${i + 1}/${sentences.size}] \"$sentence\" -> ${outInfo[0]} tokens " +
                            "(${if (stoppedNaturally) "stop real" else "tope de seguridad"}), ${elapsed}ms\n"
                    )
                    updateProgress(
                        "Generando ${i + 1}/${sentences.size}…", indeterminate = false,
                        progress = i + 1, max = sentences.size,
                    )
                    queue.put(wav)
                }
            } catch (e: Exception) {
                producerError = e
            } finally {
                queue.put(FloatArray(0))  // centinela: fin de la produccion (con o sin error)
            }
        }
        producer.start()

        // Buffer de anticipo de 2 fragmentos antes de arrancar -- ver docstring de arriba.
        val lookahead = 2
        val buffered = ArrayList<FloatArray>()
        var producerDone = false
        while (buffered.size < lookahead) {
            val wav = queue.take()
            if (wav.isEmpty()) { producerDone = true; break }  // texto corto, menos de `lookahead` fragmentos
            buffered.add(wav)
        }

        val audioTrack = buildStreamingAudioTrack()
        audioTrack.play()
        var totalFramesWritten = 0L  // mono: 1 muestra float = 1 frame
        try {
            for (wav in buffered) {
                audioTrack.write(wav, 0, wav.size, AudioTrack.WRITE_BLOCKING)
                totalFramesWritten += wav.size
            }
            if (!producerDone) {
                while (true) {
                    val wav = queue.take()
                    if (wav.isEmpty()) break
                    // Escritura bloqueante: mientras esta llamada bloquea consumiendo el buffer de
                    // reproduccion, el hilo PRODUCTOR de arriba ya esta generando el siguiente
                    // fragmento en paralelo -- asi se elimina el hueco de silencio entre fragmentos
                    // (salvo que la generación ya esté más atrás que `lookahead` fragmentos).
                    audioTrack.write(wav, 0, wav.size, AudioTrack.WRITE_BLOCKING)
                    totalFramesWritten += wav.size
                }
            }
            // WRITE_BLOCKING solo garantiza que los datos ya entraron al buffer interno de
            // AudioTrack (hasta 1s, ver buildStreamingAudioTrack) -- NO que ya sonaron. Si se
            // llama stop() apenas termina el ultimo write(), se corta el final de lo que todavia
            // estaba en el buffer sin reproducirse (reportado por el usuario: "se corta el final
            // del audio, como si hiciera falta la ultima letra"). Esperar a que la posicion de
            // reproduccion alcance lo escrito antes de parar.
            while (audioTrack.playbackHeadPosition < totalFramesWritten) {
                Thread.sleep(20)
            }
        } finally {
            producer.join()
            audioTrack.stop()
            audioTrack.release()
        }
        producerError?.let { throw it }
        postStatus("Listo.\n")
        updateProgress(
            "Listo (${sentences.size} fragmento(s))", indeterminate = false,
            progress = sentences.size, max = sentences.size,
        )
    }

    private fun buildStreamingAudioTrack(): AudioTrack {
        val sampleRate = 24000
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val minBufBytes = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT,
        )
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(format)
            .setBufferSizeInBytes(minBufBytes.coerceAtLeast(sampleRate * 4))  // >= 1s de margen
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    /** Agrega una línea al registro de debug (acumulativo) -- ver `resetLog` para empezar uno nuevo. */
    private fun postStatus(line: String) {
        mainHandler.post {
            logBuilder.append(line)
            debugLogView?.text = logBuilder.toString()
        }
    }

    /** Empieza un registro de debug nuevo (lo usa cada síntesis, para no acumular todas las
     * corridas desde que se abrió la app). Debe llamarse desde el hilo principal. */
    private fun resetLog(firstLine: String) {
        logBuilder.setLength(0)
        logBuilder.append(firstLine)
        debugLogView?.text = logBuilder.toString()
    }

    /** Actualiza la barra de progreso chica de la pantalla principal y su etiqueta. */
    private fun updateProgress(label: String, indeterminate: Boolean, progress: Int = 0, max: Int = 1) {
        mainHandler.post {
            progressLabel.text = label
            progressBar.isIndeterminate = indeterminate
            if (!indeterminate) {
                progressBar.max = max
                progressBar.progress = progress
            }
        }
    }

    private fun setButtonEnabled(enabled: Boolean) {
        // El fondo del boton es un GradientDrawable propio (sin estado "disabled" nativo), asi
        // que se atenua a mano con alpha -- si no, el boton se ve igual de clickeable aunque
        // isEnabled=false, confuso para quien esta mirando una demostracion.
        mainHandler.post {
            synthButton.isEnabled = enabled
            synthButton.alpha = if (enabled) 1f else 0.5f
        }
    }

    override fun onStop() {
        super.onStop()
        // Libera la sesion de decode del GPT (la mas pesada) al pasar a segundo plano.
        // Tokenizador/latentes/HiFiGAN quedan vivos -- se recrea perezosamente en la proxima
        // sintesis (unos segundos), aceptable al volver del background, no en medio de una
        // sintesis.
        val handle = engineHandle
        if (handle != 0L) {
            Thread { NativeBridge.nativeReleaseGpt(handle) }.start()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        val handle = engineHandle
        engineHandle = 0
        if (handle != 0L) {
            Thread { NativeBridge.nativeDestroy(handle) }.start()
        }
    }
}
