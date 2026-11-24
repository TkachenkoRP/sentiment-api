# Sentiment API — Spring Boot + ONNX Runtime в Kubernetes (Minikube)

REST-сервис анализа тональности текста. Тональность определяет **готовая нейросеть в формате `.onnx`**
([`onnx-community/rubert-tiny-sentiment-balanced-ONNX`](https://huggingface.co/onnx-community/rubert-tiny-sentiment-balanced-ONNX),
int8, 12 МБ, русский + английский, классы `negative / neutral / positive`).
В коде нет собственного алгоритма тональности: Java выполняет токенизацию (WordPiece), вызывает ONNX Runtime и применяет softmax.

```
GET  /api/sentiment?text=...        -> {"sentiment":"positive","score":0.998,"scores":{...},"pod":"sentiment-api-xxx"}
POST /api/sentiment {"text":"..."}  -> то же
GET  /api/model                     -> информация о модели
GET  /actuator/prometheus           -> метрики Micrometer (http_server_requests_*, sentiment_*, jvm_*)
GET  /actuator/health/{liveness,readiness}
```

## Структура

| Путь | Назначение |
|---|---|
| `src/main/java/ru/edu/sentiment` | Spring Boot: контроллер, ONNX-сервис, токенизатор |
| `model/` | `model_int8.onnx`, `vocab.txt` (скрипты `scripts/download-model.*`) |
| `Dockerfile` | multi-stage, jlink-JRE, distroless/cc → ~120 МБ |
| `k8s/` | Namespace, Deployment (3 реплики), Service (LoadBalancer), Ingress (/api), HPA (CPU), ServiceMonitor, нагрузочный Job |
| `monitoring/` | `values.yaml` для kube-prometheus-stack + дашборд Grafana (ConfigMap) |

## Быстрый старт (Windows PowerShell; в Linux команды те же)

```powershell
# 0. Инструменты
winget install Kubernetes.minikube Kubernetes.kubectl Helm.Helm

# 1. Кластер из 2 узлов (если мало RAM: --memory=4096mb)
minikube start --driver=docker --cpus=4 --memory=8192mb --nodes=2
minikube addons enable ingress
minikube addons enable metrics-server
kubectl get nodes -o wide

# 2. Модель (если папки model/ нет) и образ
.\scripts\download-model.ps1
docker build -t sentiment-api:1.0.0 .
docker images sentiment-api:1.0.0
minikube image load sentiment-api:1.0.0

# 3. Развёртывание
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-deployment.yaml -f k8s/20-service.yaml -f k8s/30-ingress.yaml -f k8s/40-hpa.yaml
kubectl get all,ingress,hpa -n sentiment -o wide

# 4. Доступ (в отдельном окне PowerShell от администратора оставить работать)
minikube tunnel
curl.exe "http://127.0.0.1:8080/api/sentiment?text=hello"      # через Service LoadBalancer
curl.exe "http://127.0.0.1/api/sentiment?text=hello"           # через Ingress

# 5. Мониторинг
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace -f monitoring/values.yaml
kubectl apply -f k8s/50-servicemonitor.yaml
kubectl apply -f monitoring/grafana-dashboard.yaml
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80                       # http://localhost:3000  admin / admin
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090  # http://localhost:9090

# 6. HPA под нагрузкой
kubectl apply -f k8s/60-load-test.yaml
kubectl get hpa -n sentiment -w
```

## Локальный запуск без Docker

```bash
mvn test                 # 4 теста, в т.ч. проверка позитивного/негативного текста моделью
mvn spring-boot:run
curl.exe -G localhost:8080/api/sentiment --data-urlencode "text=Отличный сервис!"
```

## Типичные проблемы

| Симптом | Решение |
|---|---|
| `ImagePullBackOff` | образ не загружен в кластер: `minikube image load sentiment-api:1.0.0`, в манифесте `imagePullPolicy: IfNotPresent` |
| HPA `<unknown>/50%` | включить `metrics-server`, подождать ~1 мин; у контейнера должны быть `resources.requests.cpu` |
| `EXTERNAL-IP <pending>` | запустить `minikube tunnel` (окно не закрывать) |
| Порт 80/8080 занят | `minikube service sentiment-api -n sentiment --url` или `kubectl port-forward` |
| Pod `Pending` | не хватает ресурсов: уменьшить `requests` или увеличить `--memory` |
| Prometheus не видит target | проверить `kubectl get servicemonitor -n sentiment`, `serviceMonitorSelectorNilUsesHelmValues: false` в values |
| `node-exporter` CrashLoop на Windows | в `values.yaml` он отключён (`nodeExporter.enabled: false`) |
| Apple Silicon / arm64 | `docker build --build-arg ORT_ARCH=linux-aarch64 ...` |
