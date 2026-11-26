# Déploiement sur une VM Oracle Cloud « Always Free »

Alternative gratuite à Railway. Sur une VM Ubuntu fraîche (ARM Ampere A1
conseillé, ou la petite AMD E2.1.Micro), avec les ports 80 et 443 ouverts
dans la liste de sécurité du réseau Oracle :

```bash
git clone https://github.com/cyril8344/ee.git ~/ee
cd ~/ee && bash deploy/oracle/install.sh
```

Le script demande de coller les variables Railway (Variables → Raw Editor,
puis `FIN`), le mot de passe du dashboard s'il manque, puis construit et
démarre le bot. Le dashboard est servi en HTTPS sur `https://<ip-avec-tirets>.sslip.io`.

| Fichier | Rôle |
|---|---|
| `Dockerfile` | même build que Railway (nixpacks) : Python 3.12, `npm run build`, uvicorn sert API + dashboard |
| `docker-compose.yml` | `bot` + `caddy` (HTTPS Let's Encrypt), base SQLite dans `XAU_DATA_DIR` (défaut `/var/lib/xau-bot`) |
| `Caddyfile` | reverse proxy vers `bot:8000`, WebSocket compris |
| `install.sh` | Docker, swap si < 3 Go de RAM, iptables 80/443, `.env`, démarrage, cron de mise à jour |
| `update.sh` | `git fetch` de `main` (ou `XAU_BRANCH`), rebuild si changé ; lancé par cron toutes les 5 min |

Commandes utiles sur la VM :

```bash
sudo docker logs --tail 100 -f xau-bot          # journaux du bot
bash ~/ee/deploy/oracle/update.sh               # forcer une mise à jour
tail -f /var/log/xau-bot-update.log             # journal des mises à jour auto
sudo cp /var/lib/xau-bot/xau_bot.db ~/sauvegarde-$(date +%F).db   # copie de la base
```

Le mode reste celui enregistré en base (`paper` pour une base neuve).
Les secrets ne vivent que dans `~/ee/.env` (ignoré par git, chmod 600).
