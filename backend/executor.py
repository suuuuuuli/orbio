"""Wykonywanie kodu kwanta - lokalnie, w podprocesie.

Zasada: kod, ktory sie wywalil, MUSI zostawic slad w wyniku. Pusty result
wyglada jak "jeszcze nie policzone", a nie jak "policzone i sie nie udalo" -
sedzia musi widziec roznice, bo od tego zalezy werdykt (computed vs refuted).

To nie jest piaskownica z gwarancjami bezpieczenstwa: podproces ma limit czasu
i osobny katalog roboczy, ale ma dostep do sieci i do systemu plikow. Docelowo
to samo ma jechac przez openrouter:shell w kontenerze bez sieci (llm.py) -
ta sciezka czeka na przepuszczenie narzedzi przez bramke.
"""

import subprocess
import sys
import tempfile

from config import EXEC_TIMEOUT, MAX_RESULT_CHARS


def _trim(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n[...truncated at {MAX_RESULT_CHARS} chars]"


def _compose(stdout: str, stderr: str, prefix: str = "") -> str:
    """Sklada wynik z czesci - stderr jest widoczny, nie zamiatany."""
    parts = []
    if prefix:
        parts.append(prefix)
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(f"[STDERR]\n{stderr.strip()}")
    return _trim("\n".join(parts))


def run_code(code: str, timeout: int = EXEC_TIMEOUT) -> str:
    """Uruchamia `code` przez sys.executable -c i zwraca wynik jako tekst.

    Zwraca zawsze niepusty tekst: wyjscie programu albo opis tego, co poszlo
    nie tak (limit czasu, kod wyjscia != 0, tresc stderr, brak jakiegokolwiek
    wypisania). Nigdy nie rzuca wyjatku - blad wykonania to dana dla sedziego,
    nie awaria debaty.
    """
    if not (code or "").strip():
        return "[ERROR] no code to execute"

    try:
        with tempfile.TemporaryDirectory(prefix="orbio-exec-") as workdir:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,          # nie smieci w repo
                encoding="utf-8",
                errors="replace",
            )
    except subprocess.TimeoutExpired as err:
        # Czesciowe wyjscie z ucietego procesu tez jest informacja.
        stdout = err.stdout or ""
        stderr = err.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return _compose(
            stdout,
            stderr,
            prefix=f"[ERROR] timed out after {timeout}s - the code did not finish",
        )
    except OSError as err:
        return _trim(f"[ERROR] could not start the code: {type(err).__name__}: {err}")

    if proc.returncode != 0:
        return _compose(
            proc.stdout,
            proc.stderr,
            prefix=f"[ERROR] code exited with status {proc.returncode}",
        )

    if not proc.stdout.strip() and not proc.stderr.strip():
        return "[ERROR] code ran without error but printed nothing (missing print?)"

    if proc.stderr.strip():
        # Kod zadzialal, ale zostawil ostrzezenia - sedzia ma je widziec.
        return _compose(proc.stdout, proc.stderr, prefix="[WARNING] code wrote to stderr")

    return _trim(proc.stdout.strip())


if __name__ == "__main__":
    PRZYKLADY = {
        "poprawny": "rev = 215.9\nprint(f'marza z przychodu: {rev * 0.70:.1f} mld')",
        "wyjatek": "print('zaczynam')\nraise ValueError('brak danych o marzy')",
        "timeout": "import time\nprint('licze...', flush=True)\ntime.sleep(30)",
        "nic nie wypisuje": "x = 2 + 2",
        "stderr": "import sys\nprint('wynik: 42')\nprint('uwaga: dane niepelne', file=sys.stderr)",
        "pusty": "   ",
    }
    for nazwa, kod in PRZYKLADY.items():
        print(f"\n===== {nazwa} =====")
        print(run_code(kod, timeout=3))
