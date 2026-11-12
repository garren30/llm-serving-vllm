# Comandos de referencia — llm-serving-vllm
---

## Entorno (WSL / venv)

```bash
# Entrar a trabajar
cd ~/mlops/serving
source .venv/bin/activate

# Verificar GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Espacio en disco
df -h /home
```

```powershell
# (PowerShell de Windows) gestionar WSL
wsl --shutdown
wsl -d Ubuntu-24.04
```

---

## vLLM nativo (venv)

```bash
# Arrancar el servidor (deja la terminal ocupada)
./serve.sh
# equivale a:
#   export VLLM_USE_FLASHINFER_SAMPLER=0
#   vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096 \
#     --gpu-memory-utilization 0.80 --enforce-eager

# Probar la API (otra terminal)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": "Hola"}]}'

# Benchmarks
python bench_serve.py        # latencia/throughput secuencial
python bench_parallel.py     # throughput vs concurrencia
python bench_saturation.py   # punto de saturación
python bench_outlen.py       # latencia vs longitud de salida
```

---

## Docker

```bash
# Verificar Docker en WSL (requiere Docker Desktop corriendo + WSL integration activada)
docker --version
docker run hello-world

# Verificar que un contenedor ve la GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Levantar vLLM con compose (lee docker-compose.yml)
docker compose up          # en primer plano (Ctrl+C para parar)
docker compose up -d       # en segundo plano (detached)
docker compose down        # detener y limpiar
docker compose logs -f     # ver logs en vivo

# Permisos del socket (una vez, si da "permission denied"):
#   sudo usermod -aG docker $USER   → luego: wsl --shutdown y reabrir
```

---

## Kubernetes (kind + kubectl)

```bash
# Crear / borrar el cluster local
kind create cluster --name mlops
kind delete cluster --name mlops

# Estado del cluster
kubectl cluster-info
kubectl get nodes

# Aplicar manifiestos (deployment, service, ...)
kubectl apply -f <archivo>.yaml

# Ver recursos
kubectl get pods
kubectl get pods -o wide          # con nodo e IP
kubectl get deployments
kubectl get services
kubectl get all                   # todo de un vistazo

# Inspeccionar / depurar
kubectl describe pod <nombre>     # detalle y eventos de un pod
kubectl logs <nombre>             # logs de un pod
kubectl logs -f <nombre>          # logs en vivo

# Escalar (o cambiar replicas en el YAML y reaplicar)
kubectl scale deployment <nombre> --replicas=4

# Probar un Service ClusterIP desde dentro del cluster
kubectl run test-curl --image=curlimages/curl --rm -it --restart=Never -- curl -s <nombre-service>

# Acceder a un Service desde tu máquina (port-forward)
kubectl port-forward service/<nombre-service> 8000:8000

# Borrar recursos
kubectl delete -f <archivo>.yaml
kubectl delete pod <nombre>       # K8s lo recrea si pertenece a un deployment
```

---

## Git (flujo del repo)

```bash
git status
git add .
git commit -m "mensaje"
git push

# Crear repo nuevo con gh (una vez)
gh repo create <nombre> --public --source=. --push
```

---
