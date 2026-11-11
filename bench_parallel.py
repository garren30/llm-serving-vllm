"""
Benchmark de CONCURRENCIA: demuestra el continuous batching de vLLM.
Manda las peticiones en paralelo (varios niveles de concurrencia) y compara
el throughput total contra el caso secuencial.

Correr desde ~/mlops/serving con el servidor vLLM levantado en otra terminal:
    python bench_parallel.py
"""
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Escribe un párrafo corto explicando qué es Kubernetes."

# Niveles de concurrencia a probar: 1 (secuencial), 4, 8, 16 peticiones a la vez
NIVELES = [1, 4, 8, 16]
PETICIONES_POR_NIVEL = 32  # total de peticiones en cada prueba

def una_peticion(_):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["usage"]["completion_tokens"]

print("Calentando...")
una_peticion(0)

print(f"\nCada prueba manda {PETICIONES_POR_NIVEL} peticiones, variando cuántas en paralelo.\n")
print(f"{'concurrencia':>12} | {'tiempo total':>12} | {'throughput':>14}")
print("-" * 46)

resultados = []
for conc in NIVELES:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        tokens = list(ex.map(una_peticion, range(PETICIONES_POR_NIVEL)))
    dt = time.perf_counter() - t0
    tput = sum(tokens) / dt
    resultados.append((conc, dt, tput))
    print(f"{conc:>12} | {dt:>10.2f} s | {tput:>10.1f} tok/s")

# Comparación final
base = resultados[0][2]  # throughput con concurrencia 1
mejor = max(resultados, key=lambda r: r[2])
print("\n" + "=" * 46)
print(f"Throughput secuencial (conc=1):  {base:.1f} tok/s")
print(f"Mejor throughput (conc={mejor[0]}):       {mejor[2]:.1f} tok/s")
print(f"Mejora:                          {mejor[2]/base:.1f}x")
print("=" * 46)
print("\nMisma GPU, mismo modelo. La diferencia es continuous batching:")
print("vLLM mete varias peticiones en el mismo lote de trabajo de la GPU,")
print("aprovechando la capacidad que sobraba en el caso secuencial.")