"""
Benchmark de SATURACIÓN: sube la concurrencia y mide throughput Y latencia
(p50/p99) en cada nivel, para encontrar el punto donde la GPU se satura.

Correr desde ~/mlops/serving con el servidor vLLM levantado en otra terminal:
    python bench_saturation.py
"""
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Escribe un párrafo corto explicando qué es Kubernetes."

NIVELES = [32, 64, 128, 256]
PETICIONES_POR_NIVEL = 64  # total de peticiones en cada prueba

def una_peticion(_):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    lat = time.perf_counter() - t0
    return lat, data["usage"]["completion_tokens"]

def percentil(valores, p):
    o = sorted(valores)
    k = (len(o) - 1) * (p / 100)
    f = int(k); c = min(f + 1, len(o) - 1)
    if f == c:
        return o[f]
    return o[f] + (o[c] - o[f]) * (k - f)

print("Calentando...")
una_peticion(0)

print(f"\nCada prueba manda {PETICIONES_POR_NIVEL} peticiones.\n")
print(f"{'conc':>5} | {'throughput':>12} | {'latencia p50':>13} | {'latencia p99':>13}")
print("-" * 54)

filas = []
for conc in NIVELES:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(una_peticion, range(PETICIONES_POR_NIVEL)))
    dt = time.perf_counter() - t0

    latencias = [r[0] for r in res]
    tokens = sum(r[1] for r in res)
    tput = tokens / dt
    p50 = percentil(latencias, 50)
    p99 = percentil(latencias, 99)
    filas.append((conc, tput, p50, p99))
    print(f"{conc:>5} | {tput:>8.1f} t/s | {p50:>11.2f} s | {p99:>11.2f} s")

# Resumen: dónde dejó de crecer el throughput
mejor = max(filas, key=lambda r: r[1])
print("\n" + "=" * 54)
print(f"Throughput máximo: {mejor[1]:.1f} tok/s a concurrencia {mejor[0]}")
print("Si el throughput se aplana pero la latencia p99 sigue subiendo,")
print("ESE es el punto de saturación: más peticiones ya no dan más trabajo")
print("útil, solo hacen esperar más a cada una (cola por KV cache lleno).")
print("=" * 54)