"""
¿Cómo afecta el LARGO DE LA RESPUESTA al tiempo?
Mismo prompt, variando max_tokens (cuántos tokens genera la respuesta).
Muestra que el tiempo lo domina la SALIDA (fase decode), no la entrada.

Correr desde ~/mlops/serving con el servidor vLLM levantado en otra terminal:
    python bench_outlen.py
"""
import time
import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explica qué es Kubernetes."  # prompt de entrada FIJO

LARGOS = [32, 64, 128, 256, 512]  # max_tokens de salida a probar
REPS = 5  # repeticiones por largo (para promediar)

def una_peticion(max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "ignore_eos": True,  # fuerza a generar EXACTAMENTE max_tokens (no para antes)
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    lat = time.perf_counter() - t0
    u = data["usage"]
    return lat, u["prompt_tokens"], u["completion_tokens"]

print("Calentando...")
una_peticion(32)

print(f"\nPrompt de ENTRADA fijo. Variando tokens de SALIDA (max_tokens).\n")
print(f"{'max_tokens':>10} | {'tokens entrada':>14} | {'tiempo medio':>12} | {'ms por token':>13}")
print("-" * 60)

filas = []
for L in LARGOS:
    tiempos = []
    for _ in range(REPS):
        lat, p_in, c_out = una_peticion(L)
        tiempos.append(lat)
    medio = sum(tiempos) / len(tiempos)
    ms_por_token = medio / L * 1000
    filas.append((L, medio))
    print(f"{L:>10} | {p_in:>14} | {medio:>10.3f} s | {ms_por_token:>10.1f} ms")

print("\n" + "=" * 60)
base = filas[0]
grande = filas[-1]
print(f"{base[0]} tokens de salida: {base[1]:.2f} s")
print(f"{grande[0]} tokens de salida: {grande[1]:.2f} s  ({grande[1]/base[1]:.1f}x más tiempo)")
print("\nLos tokens de ENTRADA no cambiaron (mismo prompt), pero el tiempo")
print("sube casi en proporción a los tokens de SALIDA. Esa es la fase")
print("decode: cada token nuevo es un paso secuencial. Fíjate que el")
print("'ms por token' se mantiene parecido → el costo es por token generado.")
print("=" * 60)