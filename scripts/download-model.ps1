# Скачивает готовую ONNX-модель и словарь токенизатора с Hugging Face в .\model (Windows PowerShell)
$base = "https://huggingface.co/onnx-community/rubert-tiny-sentiment-balanced-ONNX/resolve/main"
New-Item -ItemType Directory -Force -Path model | Out-Null
Invoke-WebRequest "$base/onnx/model_int8.onnx" -OutFile model\model_int8.onnx   # ~12 МБ
Invoke-WebRequest "$base/vocab.txt"            -OutFile model\vocab.txt          # словарь WordPiece
Invoke-WebRequest "$base/config.json"          -OutFile model\config.json        # id2label
Get-ChildItem model | Format-Table Name, Length
