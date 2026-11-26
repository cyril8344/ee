#!/usr/bin/env bash
# ============================================================================
# Installation du bot sur une VM Ubuntu (Oracle Cloud Always Free).
#
#   git clone https://github.com/cyril8344/ee.git ~/ee
#   cd ~/ee && bash deploy/oracle/install.sh
#
# Relançable sans risque : chaque étape vérifie ce qui est déjà fait.
# Ne commite jamais rien : les secrets restent dans ~/ee/.env (chmod 600,
# ignoré par git).
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

COMPOSE_FILE="deploy/oracle/docker-compose.yml"
ENV_FILE="$REPO_DIR/.env"
DEFAULT_DATA_DIR="/var/lib/xau-bot"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    OK\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m    ATTENTION\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERREUR : %s\033[0m\n' "$*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
# Les questions lisent le clavier même si le script est lancé via un tube.
TTY=/dev/tty

# --- .env helpers ---------------------------------------------------------
get_var() {  # valeur d'une clé du .env, guillemets retirés ("" si absente)
  [ -f "$ENV_FILE" ] || { echo ""; return; }
  awk -v k="$1" 'index($0, k"=")==1 { v=substr($0, length(k)+2) } END {
    gsub(/^["'\'']|["'\'']$/, "", v); print v }' "$ENV_FILE"
}
set_var() {  # remplace ou ajoute KEY='VALUE'
  # Entre apostrophes, docker compose prend la valeur à la lettre : un "$"
  # dans un mot de passe n'est pas interprété comme une variable.
  local tmp; tmp="$(mktemp)"
  [ -f "$ENV_FILE" ] && awk -v k="$1" 'index($0, k"=")!=1' "$ENV_FILE" > "$tmp"
  case "$2" in
    *"'"*) printf '%s=%s\n' "$1" "$2" >> "$tmp" ;;
    *)     printf "%s='%s'\n" "$1" "$2" >> "$tmp" ;;
  esac
  mv "$tmp" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

# ---------------------------------------------------------------------------
say "1/7 Paquets système et Docker"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl ca-certificates openssl cron >/dev/null
if ! $SUDO docker compose version >/dev/null 2>&1; then
  $SUDO apt-get install -y -qq docker.io docker-compose-v2 >/dev/null 2>&1 || true
fi
if ! $SUDO docker compose version >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | $SUDO sh
fi
$SUDO systemctl enable --now docker >/dev/null 2>&1 || true
$SUDO docker compose version >/dev/null 2>&1 || die "Docker Compose introuvable après installation."
ok "$($SUDO docker --version)"

# ---------------------------------------------------------------------------
say "2/7 Mémoire d'appoint (swap)"
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 3000 ] && [ -z "$($SUDO swapon --show --noheadings)" ]; then
  # La petite VM AMD gratuite n'a que 1 Go : sans swap le build Docker échoue.
  $SUDO fallocate -l 2G /swapfile || $SUDO dd if=/dev/zero of=/swapfile bs=1M count=2048
  $SUDO chmod 600 /swapfile
  $SUDO mkswap /swapfile >/dev/null
  $SUDO swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null
  ok "swap de 2 Go ajouté (RAM : ${MEM_MB} Mo)"
else
  ok "rien à faire (RAM : ${MEM_MB} Mo)"
fi

# ---------------------------------------------------------------------------
say "3/7 Pare-feu de la VM (ports 80 et 443)"
# Les images Ubuntu d'Oracle bloquent tout sauf SSH dans iptables, EN PLUS
# de la "liste de sécurité" du réseau Oracle (à ouvrir dans la console).
for p in 80 443; do
  $SUDO iptables -C INPUT -p tcp --dport "$p" -j ACCEPT 2>/dev/null \
    || $SUDO iptables -I INPUT 1 -p tcp --dport "$p" -j ACCEPT
done
if command -v netfilter-persistent >/dev/null 2>&1; then
  $SUDO netfilter-persistent save >/dev/null 2>&1 || true
else
  $SUDO apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
  $SUDO netfilter-persistent save >/dev/null 2>&1 || true
fi
if command -v ufw >/dev/null 2>&1 && $SUDO ufw status | grep -q "Status: active"; then
  $SUDO ufw allow 80/tcp >/dev/null; $SUDO ufw allow 443/tcp >/dev/null
fi
ok "ports 80 et 443 ouverts sur la VM"

# ---------------------------------------------------------------------------
say "4/7 Secrets (.env)"
if [ ! -f "$ENV_FILE" ]; then
  cat <<'TXT'

    Dans Railway : ton service -> onglet "Variables" -> bouton "Raw Editor".
    Copie tout le texte affiché, colle-le ici (clic droit ou Ctrl+Maj+V),
    puis tape FIN sur une ligne seule et appuie sur Entrée.
    (Si tu n'as pas accès à Railway, tape directement FIN.)

TXT
  RAW="$(mktemp)"
  while IFS= read -r line < "$TTY"; do
    [ "$(printf '%s' "$line" | tr -d '[:space:]')" = "FIN" ] && break
    printf '%s\n' "$line" >> "$RAW"
  done
  # On garde les lignes KEY=VALUE, sans les variables propres à Railway,
  # et on réécrit chaque valeur entre apostrophes (cf. set_var).
  : > "$ENV_FILE"; chmod 600 "$ENV_FILE"
  grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$RAW" \
    | grep -vE '^(RAILWAY_[A-Z0-9_]*|PORT|NIXPACKS_[A-Z0-9_]*)=' \
    | while IFS= read -r l; do
        k="${l%%=*}"; v="${l#*=}"; v="${v%$'\r'}"
        case "$v" in \"*\") v="${v#\"}"; v="${v%\"}" ;; \'*\') v="${v#\'}"; v="${v%\'}" ;; esac
        set_var "$k" "$v"
      done
  rm -f "$RAW"
  ok "$(grep -c '=' "$ENV_FILE" || true) variable(s) récupérée(s)"
else
  ok ".env existe déjà, conservé tel quel"
fi

[ -n "$(get_var ADMIN_USERNAME)" ] || set_var ADMIN_USERNAME admin

PW="$(get_var ADMIN_PASSWORD)"
if [ -z "$PW" ] || [ "$PW" = "changeme" ]; then
  # auth.py retombe sur "changeme" sans mot de passe : refusé sur Internet.
  while true; do
    printf '    Choisis le mot de passe du dashboard (identifiant : %s) : ' "$(get_var ADMIN_USERNAME)"
    read -rs PW1 < "$TTY"; echo
    printf '    Retape-le : '; read -rs PW2 < "$TTY"; echo
    if [ -n "$PW1" ] && [ "$PW1" = "$PW2" ] && [ "$PW1" != "changeme" ]; then
      set_var ADMIN_PASSWORD "$PW1"; break
    fi
    warn "vide, différent, ou 'changeme' : recommence."
  done
fi

# Sans AUTH_SECRET, auth.py en tire un au hasard à chaque démarrage et
# déconnecte tout le monde à chaque redémarrage.
[ -n "$(get_var AUTH_SECRET)" ] || set_var AUTH_SECRET "$(openssl rand -hex 32)"

if [ -z "$(get_var TWELVEDATA_API_KEY)" ]; then
  printf '    Clé Twelve Data (TWELVEDATA_API_KEY), Entrée pour passer : '
  read -r TD < "$TTY"
  if [ -n "$TD" ]; then set_var TWELVEDATA_API_KEY "$TD"
  else warn "sans clé Twelve Data, le bot se rabat sur yfinance (données moins fiables)."; fi
fi

[ -n "$(get_var XAU_DATA_DIR)" ] || set_var XAU_DATA_DIR "$DEFAULT_DATA_DIR"
DATA_DIR="$(get_var XAU_DATA_DIR)"

if [ -z "$(get_var DOMAIN)" ]; then
  IP="$(curl -fsS --max-time 10 https://api.ipify.org || curl -fsS --max-time 10 https://ifconfig.me || true)"
  printf '%s' "$IP" | grep -Eq '^[0-9]+(\.[0-9]+){3}$' || die "adresse IP publique introuvable."
  # sslip.io : nom gratuit qui pointe sur l'IP, pour obtenir un certificat HTTPS.
  set_var DOMAIN "$(printf '%s' "$IP" | tr '.' '-').sslip.io"
fi
DOMAIN="$(get_var DOMAIN)"
ok "adresse du dashboard : https://$DOMAIN"

# ---------------------------------------------------------------------------
say "5/7 Dossier des données ($DATA_DIR)"
$SUDO mkdir -p "$DATA_DIR"
if [ -f "$HOME/xau_bot.db" ] && [ ! -f "$DATA_DIR/xau_bot.db" ]; then
  $SUDO cp "$HOME/xau_bot.db" "$DATA_DIR/xau_bot.db"
  ok "base existante importée depuis ~/xau_bot.db"
elif [ -f "$DATA_DIR/xau_bot.db" ]; then
  ok "base existante conservée"
else
  ok "nouvelle base (créée au premier démarrage, mode paper par défaut)"
fi

# ---------------------------------------------------------------------------
say "6/7 Construction et démarrage (5 à 15 minutes la première fois)"
$SUDO docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build

printf '    Attente du démarrage du bot'
STATUS=""
for _ in $(seq 1 60); do
  STATUS="$($SUDO docker inspect -f '{{.State.Health.Status}}' xau-bot 2>/dev/null || echo absent)"
  [ "$STATUS" = "healthy" ] && break
  printf '.'; sleep 5
done
echo
[ "$STATUS" = "healthy" ] || die "le bot ne répond pas (état : $STATUS). Voir : sudo docker logs --tail 100 xau-bot"
ok "bot démarré"

# ---------------------------------------------------------------------------
say "7/7 Mises à jour automatiques depuis GitHub (comme Railway)"
CRON_FILE=/etc/cron.d/xau-bot-autoupdate
printf '%s\n' \
  "# Toutes les 5 min : si 'main' a changé sur GitHub, reconstruit et relance le bot." \
  "*/5 * * * * root /bin/bash $REPO_DIR/deploy/oracle/update.sh --auto >> /var/log/xau-bot-update.log 2>&1" \
  | $SUDO tee "$CRON_FILE" >/dev/null
$SUDO chmod 644 "$CRON_FILE"
ok "vérification toutes les 5 minutes (journal : /var/log/xau-bot-update.log)"

# ---------------------------------------------------------------------------
say "Vérification HTTPS"
HTTPS_OK=""
for _ in $(seq 1 24); do
  if curl -fsS -o /dev/null --max-time 10 "https://$DOMAIN/"; then HTTPS_OK=1; break; fi
  sleep 5
done

echo
if [ -n "$HTTPS_OK" ]; then
  printf '\033[1;32m  C'\''est en ligne : https://%s\033[0m\n' "$DOMAIN"
else
  warn "le bot tourne mais https://$DOMAIN ne répond pas encore depuis la VM."
  warn "Vérifie que les ports 80 et 443 sont ouverts dans la liste de sécurité Oracle,"
  warn "puis réessaie dans 2 minutes (le certificat HTTPS peut tarder)."
fi
printf '  Identifiant : %s  (mot de passe : celui que tu as choisi / celui de Railway)\n\n' "$(get_var ADMIN_USERNAME)"
