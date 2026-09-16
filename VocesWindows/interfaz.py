"""
Interfaz grafica del generador de voz en CustomTkinter (esquinas redondeadas, modo claro).

    pip install customtkinter

Misma API que interfaz.py -- el trabajo pesado corre en hilos y la UI se actualiza siempre
desde el hilo principal con root.after() (tkinter no es thread-safe).

Como conectarlo al pipeline real: reemplazar main() por

    from interfaz_ctk import lanzar_interfaz
    lanzar_interfaz(cargar=mi_carga, generar=mi_sintesis)

donde
    cargar(log=None)        -> devuelve el contexto que necesites (modelo, latents...). Corre en
                                un hilo. `log`, si se pasa, es un callable(str) para reportar
                                avance -- termina en la pestana de Ajustes -> Debug.
    generar(ctx, t, temperature=.., top_p=.., top_k=.., repetition_penalty=.., log=None)
                             -> sintetiza y reproduce el texto t. Bloqueante; corre en un hilo.
                                Los 4 parametros de muestreo llegan con los valores que el
                                usuario dejo en la pestana de Ajustes (o los valores por defecto).
"""

import threading
import wave
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk

# ---------------------------------------------------------------- paleta (modo claro)
FONDO = "#F1EFEA"
PANEL = "#FFFFFF"
BARRA_TITULO = "#FAF8F4"
CAMPO = "#FBFAF7"
BORDE = "#DDD8D0"
BORDE_SUAVE = "#E9E5DE"
TEXTO = "#1D2026"
TEXTO_SUAVE = "#6B7280"
TEXTO_TENUE = "#9A968D"
ACENTO = "#2F6FB5"
ACENTO_CLARO = "#4487CE"
VERDE = "#3F9D5C"
NARANJA = "#C96A2E"
GRIS = "#A8A29A"

FUENTE = "Segoe UI"  # "Helvetica" en Linux/macOS si no existe
TITULO_APP = "TCU-748"

# Valores por defecto y rangos validos de los 4 parametros de muestreo del GPT -- mismos valores
# que trae infer.py (ver synthesize()).
PARAM_DEFECTOS = {"temperature": "0.85", "top_k": "50", "top_p": "0.85", "repetition_penalty": "2.0"}
PARAM_RANGOS = {
    "temperature": (0.05, 3.0),
    "top_k": (1, 1026),
    "top_p": (0.01, 1.0),
    "repetition_penalty": (1.0, 5.0),
}


class Interfaz:
    def __init__(self, cargar=None, generar=None, sample_rate=24000):
        self._cargar = cargar
        self._generar = generar
        self._sample_rate = sample_rate
        self._ctx = None
        self._ocupado = True  # arranca ocupada: esta cargando el modelo
        self._audio = None  # ultimo audio generado (numpy float32 mono), listo para guardar

        ctk.set_appearance_mode("light")
        self.root = ctk.CTk(fg_color=FONDO)
        self.root.title("VocesTCU748")
        self.root.geometry("600x640")
        self.root.resizable(False, False)

        self.f_titulo = ctk.CTkFont(FUENTE, 26, "bold")
        self.f_sub = ctk.CTkFont(FUENTE, 13)
        self.f_chrome = ctk.CTkFont(FUENTE, 10, "bold")
        self.f_texto = ctk.CTkFont(FUENTE, 14)
        self.f_boton = ctk.CTkFont(FUENTE, 14, "bold")
        self.f_pequeno = ctk.CTkFont(FUENTE, 11)
        self.f_debug = ctk.CTkFont("Consolas", 11)

        self._construir()
        self.root.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.root.after(120, self._iniciar_carga)

    # ------------------------------------------------------------ layout
    def _construir(self):
        marco = ctk.CTkFrame(self.root, fg_color=PANEL, corner_radius=14,
                             border_width=1, border_color=BORDE_SUAVE)
        marco.pack(fill="both", expand=True, padx=20, pady=20)

        chrome = ctk.CTkFrame(marco, fg_color=BARRA_TITULO, corner_radius=13, height=44)
        chrome.pack(fill="x", padx=1, pady=(1, 0))
        chrome.pack_propagate(False)
        punto = ctk.CTkFrame(chrome, width=9, height=9, corner_radius=5, fg_color=ACENTO)
        punto.pack(side="left", padx=(17, 9))
        ctk.CTkLabel(chrome, text=TITULO_APP, font=self.f_chrome, text_color=TEXTO_TENUE).pack(side="left")
        ctk.CTkButton(chrome, text="⚙", command=self._abrir_ajustes,
                      width=28, height=28, corner_radius=14, font=ctk.CTkFont(FUENTE, 13),
                      fg_color="#ECE9E2", hover_color="#DFDACF", text_color=TEXTO_SUAVE
                      ).pack(side="right", padx=(0, 14))

        # Un solo contenedor: las dos vistas (Generar / Ajustes) viven en el mismo lugar y se
        # muestran/ocultan con pack()/pack_forget() -- Generar es la vista por defecto. Ajustes no
        # es una ventana nueva del SO (Toplevel): es un frame mas dentro de esta misma ventana, al
        # estilo del dialogo de ajustes de VocesAndroid (integrado en la app, no una ventana aparte).
        self.contenido = ctk.CTkFrame(marco, fg_color="transparent")
        self.contenido.pack(fill="both", expand=True, padx=14, pady=(14, 18))

        self.vista_generar = ctk.CTkFrame(self.contenido, fg_color="transparent")
        self.vista_ajustes = ctk.CTkFrame(self.contenido, fg_color="transparent")

        self._construir_generar(self.vista_generar)
        self._construir_ajustes(self.vista_ajustes)

        self.vista_generar.pack(fill="both", expand=True)

    def _abrir_ajustes(self):
        self.vista_generar.pack_forget()
        self.vista_ajustes.pack(fill="both", expand=True)

    def _cerrar_ajustes(self):
        self.vista_ajustes.pack_forget()
        self.vista_generar.pack(fill="both", expand=True)

    def _construir_generar(self, tab):
        cuerpo = ctk.CTkFrame(tab, fg_color="transparent")
        cuerpo.pack(fill="both", expand=True, padx=8, pady=(10, 4))

        ctk.CTkLabel(cuerpo, text="Sistema de Comunicación Alternativa", font=self.f_titulo,
                     text_color=TEXTO, anchor="w").pack(fill="x")
        ctk.CTkLabel(cuerpo, text="Escriba el texto y presione Generar para escucharlo.",
                     font=self.f_sub, text_color=TEXTO_SUAVE, anchor="w").pack(fill="x", pady=(2, 16))

        self.entrada = ctk.CTkTextbox(cuerpo, height=150, corner_radius=10,
                                      fg_color=CAMPO, border_width=1, border_color=BORDE,
                                      text_color=TEXTO, font=self.f_texto, wrap="word")
        self.entrada.pack(fill="x")
        self.entrada.bind("<KeyRelease>", self._actualizar_conteo)
        self.entrada.bind("<Control-Return>", lambda e: (self._on_generar(), "break")[1])

        self.conteo = ctk.CTkLabel(cuerpo, text="0 caracteres", font=self.f_pequeno,
                                   text_color=TEXTO_TENUE, anchor="e")
        self.conteo.pack(fill="x", pady=(6, 16))

        botones = ctk.CTkFrame(cuerpo, fg_color="transparent")
        botones.pack(fill="x")
        self.btn_generar = ctk.CTkButton(botones, text="Generar", command=self._on_generar,
                                         height=46, corner_radius=10, font=self.f_boton,
                                         fg_color=ACENTO, hover_color=ACENTO_CLARO,
                                         text_color="#FFFFFF", text_color_disabled=TEXTO_TENUE)
        self.btn_generar.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(botones, text="Cerrar", command=self._cerrar, width=110, height=46,
                      corner_radius=10, font=self.f_boton, fg_color=PANEL,
                      hover_color="#F3F1EC", border_width=1, border_color=BORDE,
                      text_color=TEXTO_SUAVE).pack(side="left", padx=(12, 0))

        estado = ctk.CTkFrame(cuerpo, fg_color=BARRA_TITULO, corner_radius=10,
                              border_width=1, border_color=BORDE_SUAVE)
        estado.pack(fill="x", pady=(16, 0))
        fila = ctk.CTkFrame(estado, fg_color="transparent")
        fila.pack(fill="x", padx=15, pady=(13, 0))
        self.luz = ctk.CTkFrame(fila, width=8, height=8, corner_radius=4, fg_color=ACENTO)
        self.luz.pack(side="left")
        self.etiqueta_estado = ctk.CTkLabel(fila, text="Cargando modelo, un momento...",
                                            font=self.f_sub, text_color="#565C64", anchor="w")
        self.etiqueta_estado.pack(side="left", padx=(9, 0))

        self.progreso = ctk.CTkProgressBar(estado, height=6, corner_radius=3,
                                           fg_color="#E7E2D9", progress_color=ACENTO,
                                           mode="indeterminate", indeterminate_speed=1.2)
        self.progreso.pack(fill="x", padx=15, pady=(10, 14))
        self.progreso.set(0)

    def _construir_ajustes(self, tab):
        encabezado = ctk.CTkFrame(tab, fg_color="transparent")
        encabezado.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkButton(encabezado, text="← Volver", command=self._cerrar_ajustes,
                      width=92, height=32, corner_radius=8, font=self.f_pequeno,
                      fg_color=PANEL, hover_color="#F3F1EC", border_width=1, border_color=BORDE,
                      text_color=TEXTO_SUAVE).pack(side="left")
        ctk.CTkLabel(encabezado, text="Ajustes", font=self.f_titulo, text_color=TEXTO
                     ).pack(side="left", padx=(14, 0))

        contenedor = ctk.CTkScrollableFrame(tab, fg_color="transparent",
                                            scrollbar_button_color=BORDE,
                                            scrollbar_button_hover_color=TEXTO_TENUE)
        contenedor.pack(fill="both", expand=True, padx=4, pady=(10, 2))

        ctk.CTkLabel(contenedor, text="Parametros de generacion", font=self.f_sub,
                     text_color=TEXTO, anchor="w").pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(contenedor, text="Se aplican en la proxima generacion.", font=self.f_pequeno,
                     text_color=TEXTO_TENUE, anchor="w").pack(fill="x", pady=(2, 4))

        self.entry_temperature = self._fila_parametro(
            contenedor, "Temperature (0.05 - 3.0)", PARAM_DEFECTOS["temperature"])
        self.entry_top_k = self._fila_parametro(
            contenedor, "Top-k (1 - 1026)", PARAM_DEFECTOS["top_k"])
        self.entry_top_p = self._fila_parametro(
            contenedor, "Top-p (0.01 - 1.0)", PARAM_DEFECTOS["top_p"])
        self.entry_penalty = self._fila_parametro(
            contenedor, "Repetition penalty (1.0 - 5.0)", PARAM_DEFECTOS["repetition_penalty"])

        ctk.CTkButton(contenedor, text="Restablecer valores", command=self._restablecer_parametros,
                      height=38, corner_radius=10, font=self.f_pequeno, fg_color=PANEL,
                      hover_color="#F3F1EC", border_width=1, border_color=BORDE,
                      text_color=TEXTO_SUAVE).pack(fill="x", pady=(14, 0))

        self.btn_guardar = ctk.CTkButton(contenedor, text="Guardar audio", command=self._on_guardar,
                                         height=44, corner_radius=10, font=self.f_boton,
                                         fg_color=PANEL, hover_color="#F3F1EC",
                                         border_width=1, border_color=BORDE,
                                         text_color=TEXTO_SUAVE,
                                         text_color_disabled=TEXTO_TENUE, state="disabled")
        self.btn_guardar.pack(fill="x", pady=(20, 0))

        ctk.CTkLabel(contenedor, text="DEBUG", font=self.f_chrome, text_color=TEXTO_TENUE,
                     anchor="w").pack(fill="x", pady=(24, 6))
        self.debug_box = ctk.CTkTextbox(contenedor, height=220, corner_radius=10,
                                        fg_color=CAMPO, border_width=1, border_color=BORDE,
                                        text_color=TEXTO_SUAVE, font=self.f_debug,
                                        wrap="word", state="disabled")
        self.debug_box.pack(fill="both", expand=True, pady=(0, 4))

    def _fila_parametro(self, parent, etiqueta, valor_defecto):
        ctk.CTkLabel(parent, text=etiqueta, font=self.f_pequeno, text_color=TEXTO_SUAVE,
                     anchor="w").pack(fill="x", pady=(10, 4))
        entry = ctk.CTkEntry(parent, corner_radius=8, height=34, fg_color=CAMPO,
                             border_width=1, border_color=BORDE, text_color=TEXTO,
                             font=self.f_texto)
        entry.insert(0, valor_defecto)
        entry.pack(fill="x")
        return entry

    # ------------------------------------------------------------ estado
    def _estado(self, texto, color=GRIS):
        self.etiqueta_estado.configure(text=texto)
        self.luz.configure(fg_color=color)

    def _actualizar_conteo(self, _=None):
        n = len(self.entrada.get("1.0", "end-1c"))
        self.conteo.configure(text=f"{n} caracteres")

    def _bloquear(self, ocupado, etiqueta="Generar"):
        self._ocupado = ocupado
        self.btn_generar.configure(text=etiqueta, state="disabled" if ocupado else "normal")
        self.entrada.configure(state="disabled" if ocupado else "normal")

    def _audio_listo(self, listo):
        self._audio = self._audio if listo else None
        self.btn_guardar.configure(state="normal" if listo else "disabled")

    def _barra(self, activa):
        if activa:
            self.progreso.start()
        else:
            self.progreso.stop()
            self.progreso.set(0)

    # ------------------------------------------------------------ debug (pestana Ajustes)
    def _log(self, linea):
        """Agrega una linea al panel de debug. Thread-safe -- se puede llamar desde el hilo de
        carga/generacion, la escritura real siempre corre en el hilo principal via root.after()."""
        texto = linea if linea.endswith("\n") else linea + "\n"

        def _aplicar():
            self.debug_box.configure(state="normal")
            self.debug_box.insert("end", texto)
            self.debug_box.see("end")
            self.debug_box.configure(state="disabled")

        self.root.after(0, _aplicar)

    def _log_reset(self, linea):
        """Como _log(), pero primero borra el contenido anterior -- se usa al arrancar una carga
        o generacion nueva, para no acumular el historial de corridas anteriores."""
        texto = linea if linea.endswith("\n") else linea + "\n"

        def _aplicar():
            self.debug_box.configure(state="normal")
            self.debug_box.delete("1.0", "end")
            self.debug_box.insert("end", texto)
            self.debug_box.see("end")
            self.debug_box.configure(state="disabled")

        self.root.after(0, _aplicar)

    # ------------------------------------------------------------ parametros (pestana Ajustes)
    def _leer_parametros(self):
        def _clamp(entry, defecto, lo, hi):
            try:
                v = float(entry.get())
            except ValueError:
                v = defecto
            return min(max(v, lo), hi)

        lo, hi = PARAM_RANGOS["temperature"]
        temperature = _clamp(self.entry_temperature, float(PARAM_DEFECTOS["temperature"]), lo, hi)
        lo, hi = PARAM_RANGOS["top_p"]
        top_p = _clamp(self.entry_top_p, float(PARAM_DEFECTOS["top_p"]), lo, hi)
        lo, hi = PARAM_RANGOS["repetition_penalty"]
        repetition_penalty = _clamp(self.entry_penalty, float(PARAM_DEFECTOS["repetition_penalty"]), lo, hi)
        lo, hi = PARAM_RANGOS["top_k"]
        top_k = int(_clamp(self.entry_top_k, float(PARAM_DEFECTOS["top_k"]), lo, hi))

        return {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                "repetition_penalty": repetition_penalty}

    def _restablecer_parametros(self):
        for entry, clave in ((self.entry_temperature, "temperature"), (self.entry_top_k, "top_k"),
                              (self.entry_top_p, "top_p"), (self.entry_penalty, "repetition_penalty")):
            entry.delete(0, "end")
            entry.insert(0, PARAM_DEFECTOS[clave])
        self._log("Parametros restablecidos a los valores por defecto.")

    # ------------------------------------------------------------ acciones
    def _iniciar_carga(self):
        self._bloquear(True, "Cargando...")
        self._barra(True)
        self._log_reset("Cargando modelo...")

        def tarea():
            try:
                self._ctx = self._cargar(log=self._log) if self._cargar else None
                self.root.after(0, lambda: (self._barra(False), self._bloquear(False),
                                            self._estado("Listo. Modelo cargado.")))
            except Exception as e:
                self.root.after(0, lambda: (self._barra(False),
                                            self._estado(f"Error al cargar: {e}", NARANJA)))
                self._log(f"Error al cargar: {e}")

        threading.Thread(target=tarea, daemon=True).start()

    def _on_generar(self):
        if self._ocupado:
            return
        texto = self.entrada.get("1.0", "end-1c").strip()
        if not texto:
            self._estado("Escriba algo antes de generar.", NARANJA)
            return
        parametros = self._leer_parametros()
        self._bloquear(True, "Generando...")
        self._audio_listo(False)
        self._estado("Generando audio...", ACENTO)
        self._barra(True)
        self._log_reset(
            "Generando (temperature={temperature} top_k={top_k} top_p={top_p} "
            "repetition_penalty={repetition_penalty})...".format(**parametros)
        )

        def tarea():
            try:
                wav = self._generar(self._ctx, texto, log=self._log, **parametros) if self._generar else None
                self._audio = wav
                hay = wav is not None and len(wav) > 0
                self.root.after(0, lambda: (self._barra(False), self._bloquear(False),
                                            self._audio_listo(hay),
                                            self._estado("Reproducción finalizada.", VERDE)))
            except Exception as e:
                self.root.after(0, lambda: (self._barra(False), self._bloquear(False),
                                            self._estado(f"Error: {e}", NARANJA)))
                self._log(f"Error: {e}")

        threading.Thread(target=tarea, daemon=True).start()

    def _on_guardar(self):
        if self._audio is None:
            self._estado("No hay audio para guardar.", NARANJA)
            return
        sugerido = f"voz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        ruta = filedialog.asksaveasfilename(
            title="Guardar audio", defaultextension=".wav", initialfile=sugerido,
            filetypes=[("Audio WAV", "*.wav")],
        )
        if not ruta:
            return
        try:
            import numpy as np
            datos = np.clip(np.asarray(self._audio, dtype=np.float32), -1.0, 1.0)
            pcm = (datos * 32767.0).astype("<i2")  # 16 bits, little-endian
            with wave.open(ruta, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self._sample_rate)
                f.writeframes(pcm.tobytes())
            import os
            nombre = os.path.basename(ruta)
            self._estado(f"Guardado: {nombre}", VERDE)
            self._log(f"Audio guardado: {nombre}")
        except Exception as e:
            self._estado(f"Error al guardar: {e}", NARANJA)
            self._log(f"Error al guardar: {e}")

    def _cerrar(self):
        self._estado("Cerrando la aplicación...", NARANJA)
        self.root.after(150, self.root.destroy)

    def ejecutar(self):
        self.root.mainloop()


def lanzar_interfaz(cargar=None, generar=None, sample_rate=24000):
    Interfaz(cargar=cargar, generar=generar, sample_rate=sample_rate).ejecutar()


if __name__ == "__main__":
    import time
    import numpy as np

    def _demo_cargar(log=None):
        if log:
            log("Cargando modelo (demo)...")
        time.sleep(1.2)
        if log:
            log("Modelo cargado (demo).")

    def _demo_generar(ctx, t, temperature=0.85, top_p=0.85, top_k=50, repetition_penalty=2.0, log=None):
        if log:
            log(f"Sintetizando \"{t[:40]}\" (temperature={temperature}, top_k={top_k}, "
                f"top_p={top_p}, repetition_penalty={repetition_penalty})...")
        time.sleep(2.5)  # tono de prueba para probar el guardado
        n = 24000
        if log:
            log("Listo (demo).")
        return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 24000)).astype(np.float32)

    lanzar_interfaz(cargar=_demo_cargar, generar=_demo_generar)
