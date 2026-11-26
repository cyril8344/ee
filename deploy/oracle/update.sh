#!/usr/bin/env bash
# ============================================================================
# Met le bot à jour depuis GitHub et le relance.
#   bash deploy/oracle/update.sh          -> mise à jour forcée (rebuild)
#   bash deploy/oracle/update.sh --auto   -> cron : ne fait rien si 'main'
#                                            n'a pas bougé
# Branche suivie : XAU_BRANCH dans .env (défaut : main), comme Railway.
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
ENV_FILE="$REPO_DIR/.env"
COMPOSE=(docker compose -f deploy/oracle/docker-compose.yml --env-file "$ENV_FILE")
AUTO="${1:-}"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# Un seul build à la fois (le cron tourne toutes les 5 min, un build ARM
# peut en durer davantage).
exec 9>/tmp/xau-bot-update.lock
flock -n 9 || exit 0

# git tourne sous le propriétaire du dépôt (root via cron sinon refusé :
# "dubious ownership").
OWNER="$(stat -c %U "$REPO_DIR")"
run_git() {
  if [ "$(id -un)" = "$OWNER" ]; then git "$@"; else sudo -u "$OWNER" git "$@"; fi
}

BRANCH="$(awk 'index($0,"XAU_BRANCH=")==1 {v=substr($0,12)} END {gsub(/["'\'']/, "", v); print v}' "$ENV_FILE" 2>/dev/null || true)"
BRANCH="${BRANCH:-main}"

run_git fetch -q origin "$BRANCH"
LOCAL="$(run_git rev-parse HEAD)"
REMOTE="$(run_git rev-parse FETCH_HEAD)"

if [ "$AUTO" = "--auto" ] && [ "$LOCAL" = "$REMOTE" ]; then
  exit 0
fi

echo "[$(date -u +%FT%TZ)] mise à jour ${LOCAL:0:7} -> ${REMOTE:0:7} ($BRANCH)"
run_git checkout -q "$BRANCH" 2>/dev/null || run_git checkout -q -b "$BRANCH" FETCH_HEAD
# --ff-only : une modification locale bloque la mise à jour au lieu d'être
# écrasée en silence.
run_git merge -q --ff-only FETCH_HEAD

$SUDO "${COMPOSE[@]}" up -d --build
$SUDO docker image prune -f >/dev/null
echo "[$(date -u +%FT%TZ)] bot relancé sur ${REMOTE:0:7}"
