"""
Benchmark básico de tu servidor vLLM.
Manda N peticiones (una por una) y mide latencia y throughput.

Correr desde ~/mlops/serving con el servidor vLLM ya levantado en otra terminal:
    python bench_serve.py
"""
import time
import statistics
import urllib.request
import json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 20  # número de peticiones a medir

# Un prompt fijo para que todas las peticiones sean comparables
PROMPT = "Escribe un párrafo corto explicando qué es Kubernetes."

def una_peticion():
    """Manda una petición y devuelve (latencia_seg, tokens_generados)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,        # tope de tokens de respuesta (para que todas sean parecidas)
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    t1 = time.perf_counter()

    tokens_salida = data["usage"]["completion_tokens"]
    return (t1 - t0), tokens_salida

def percentil(valores, p):
    """Percentil p (0-100) de una lista, sin numpy."""
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(ordenados) - 1)
    if f == c:
        return ordenados[f]
    return ordenados[f] + (ordenados[c] - ordenados[f]) * (k - f)

print(f"Calentando (1 petición de warmup)...")
una_peticion()  # la primera suele ser más lenta; no la contamos

print(f"Midiendo {N} peticiones...\n")
latencias = []
tokens_totales = 0
t_inicio = time.perf_counter()

for i in range(N):
    lat, toks = una_peticion()
    latencias.append(lat)
    tokens_totales += toks
    print(f"  petición {i+1:>2}: {lat:6.3f} s, {toks} tokens generados")

t_total = time.perf_counter() - t_inicio

# --- Resultados ---
print("\n" + "=" * 50)
print("RESULTADOS")
print("=" * 50)
print(f"Peticiones:              {N}")
print(f"Latencia media:          {statistics.mean(latencias):.3f} s")
print(f"Latencia p50 (mediana):  {percentil(latencias, 50):.3f} s")
print(f"Latencia p95:            {percentil(latencias, 95):.3f} s")
print(f"Latencia p99:            {percentil(latencias, 99):.3f} s")
print(f"Tokens generados total:  {tokens_totales}")
print(f"Tiempo total:            {t_total:.2f} s")
print(f"Throughput:              {tokens_totales / t_total:.1f} tokens/s")
print("=" * 50)
print("\nNota: esto es SECUENCIAL (una petición a la vez).")
print("El throughput real de vLLM brilla con peticiones EN PARALELO")
print("(continuous batching) — lo mediremos después.")