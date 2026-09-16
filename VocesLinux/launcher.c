/*
 * Lanzador nativo: arranca infer.py con el interprete de Python sin dejar una
 * consola/terminal abierta, para que la app se sienta como un ejecutable normal
 * de escritorio. La logica de Windows y la de POSIX (Linux/macOS) son bastante
 * distintas a nivel de API del SO, asi que estan separadas con #ifdef.
 *
 * Windows: usa el Python embebido en .\python\pythonw.exe (bundle propio).
 * POSIX:   todavia no hay un Python bundleado para Linux/macOS (pendiente),
 *          asi que usa el "python3" del sistema (PATH), o uno en ./python/bin/
 *          si en el futuro se agrega ese bundle ahi.
 *
 * En ambos casos se redirige stdout/stderr a infer.log junto al ejecutable,
 * porque esta app es una GUI (tkinter) y no tiene una consola donde mostrar
 * los print() -- en Windows pythonw.exe directamente deja sys.stdout en None
 * si no se redirige, lo cual haria fallar cualquier print().
 */

#include <stdio.h>
#include <string.h>

#ifdef _WIN32

#include <windows.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    char exePath[MAX_PATH];
    char dir[MAX_PATH];
    char pythonExe[MAX_PATH];
    char cmdLine[MAX_PATH * 2];
    char logPath[MAX_PATH];

    if (!GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        MessageBoxA(NULL, "No se pudo determinar la ruta del ejecutable.", "Error", MB_OK | MB_ICONERROR);
        return 1;
    }

    strncpy(dir, exePath, MAX_PATH);
    char *lastSlash = strrchr(dir, '\\');
    if (lastSlash != NULL) {
        *lastSlash = '\0';
    }

    snprintf(pythonExe, sizeof(pythonExe), "%s\\python\\pythonw.exe", dir);
    snprintf(cmdLine, sizeof(cmdLine), "\"%s\" -u infer.py", pythonExe);
    snprintf(logPath, sizeof(logPath), "%s\\infer.log", dir);

    /* pythonw.exe deja sys.stdout/sys.stderr en None si no se redirigen,
       lo que hace fallar cualquier print(); los mandamos a un archivo. */
    SECURITY_ATTRIBUTES sa;
    ZeroMemory(&sa, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE hLog = CreateFileA(
        logPath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL
    );
    HANDLE hNul = CreateFileA(
        "NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (hLog != INVALID_HANDLE_VALUE && hNul != INVALID_HANDLE_VALUE) {
        si.dwFlags |= STARTF_USESTDHANDLES;
        si.hStdInput = hNul;
        si.hStdOutput = hLog;
        si.hStdError = hLog;
    }

    BOOL ok = CreateProcessA(
        pythonExe,      /* aplicacion */
        cmdLine,        /* linea de comandos (mutable, requerido por la API) */
        NULL, NULL,
        TRUE,           /* hereda los handles de stdin/stdout/stderr redirigidos */
        CREATE_NO_WINDOW,
        NULL,
        dir,            /* directorio de trabajo: la carpeta del .exe */
        &si, &pi
    );

    if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
    if (hNul != INVALID_HANDLE_VALUE) CloseHandle(hNul);

    if (!ok) {
        char msg[MAX_PATH + 128];
        snprintf(msg, sizeof(msg), "No se pudo iniciar Python embebido (%s). Codigo de error: %lu", pythonExe, GetLastError());
        MessageBoxA(NULL, msg, "Error", MB_OK | MB_ICONERROR);
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return (int)exitCode;
}

#else /* POSIX: Linux y macOS */

#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

/* Escribe en dir (buffer de tamano size) la carpeta que contiene este ejecutable, sin barra final. */
static int get_exe_dir(char *dir, size_t size) {
    char exePath[PATH_MAX];

#if defined(__APPLE__)
    uint32_t bufSize = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &bufSize) != 0) {
        return -1;
    }
    char resolved[PATH_MAX];
    if (realpath(exePath, resolved) == NULL) {
        return -1;
    }
    strncpy(exePath, resolved, sizeof(exePath) - 1);
    exePath[sizeof(exePath) - 1] = '\0';
#else
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len == -1) {
        return -1;
    }
    exePath[len] = '\0';
#endif

    strncpy(dir, exePath, size - 1);
    dir[size - 1] = '\0';
    char *lastSlash = strrchr(dir, '/');
    if (lastSlash != NULL) {
        *lastSlash = '\0';
    }
    return 0;
}

int main(void) {
    char dir[PATH_MAX];
    char bundledPython[PATH_MAX];
    char logPath[PATH_MAX];

    if (get_exe_dir(dir, sizeof(dir)) != 0) {
        fprintf(stderr, "No se pudo determinar la ruta del ejecutable.\n");
        return 1;
    }

    if (chdir(dir) != 0) {
        fprintf(stderr, "No se pudo cambiar al directorio de la aplicacion (%s).\n", dir);
        return 1;
    }

    snprintf(logPath, sizeof(logPath), "%s/infer.log", dir);
    freopen(logPath, "w", stdout);
    freopen(logPath, "a", stderr);
    freopen("/dev/null", "r", stdin);

    /* Bundle propio para Linux/macOS: pendiente. Si en el futuro se agrega un
       Python portable en ./python/bin/python3, se usa; si no, cae al del sistema. */
    snprintf(bundledPython, sizeof(bundledPython), "%s/python/bin/python3", dir);
    if (access(bundledPython, X_OK) == 0) {
        execl(bundledPython, bundledPython, "-u", "infer.py", (char *)NULL);
    } else {
        execlp("python3", "python3", "-u", "infer.py", (char *)NULL);
    }

    /* Si llegamos aca, exec fallo. */
    fprintf(stderr, "No se pudo iniciar Python (ni %s ni python3 del PATH).\n", bundledPython);
    return 1;
}

#endif
