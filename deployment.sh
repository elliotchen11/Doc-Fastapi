#!/usr/bin/env bash
#chmod +x ~/deployment.sh

REPO_DIR="/home/shanlin.chen/HabitatRestoration/DocFlowScript"
VENV_DIR="$REPO_DIR/.ocrenv"
SERVICE_NAME="tidellm"
HEALTH_URL="http://127.0.0.1:8000/health"

echo "=== Starting LLM deployment ==="

cd "$REPO_DIR"

echo "=== Git status ==="
git status --short || true

echo "=== Pull latest code ==="
git pull

echo "=== Restart systemd service ==="
sudo systemctl restart "$SERVICE_NAME"

echo "=== Check service status ==="
sudo systemctl status "$SERVICE_NAME" --no-pager

echo "=== Wait for API to become healthy ==="
for i in {1..15}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "=== Health check passed ==="
    break
  fi
  echo "Waiting for service... ($i/15)"
  sleep 2
done

echo "=== Final health response ==="
curl -fsS "$HEALTH_URL"
echo

echo "=== Validate and reload nginx ==="
sudo nginx -t
sudo systemctl reload nginx

echo "=== Deployment complete ==="