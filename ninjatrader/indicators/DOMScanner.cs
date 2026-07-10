// DOMScanner.cs — NinjaTrader 8 Indicator
// Détecte les gros ordres DOM, l'absorption réelle et le spoofing sur ES/GC
//
// Installation :
//   1. Copier ce fichier dans Documents\NinjaTrader 8\bin\Custom\Indicators\
//   2. NinjaTrader → Tools → Edit NinjaScript → Compile
//   3. Ajouter l'indicateur sur ton chart ES/GC (clic droit → Indicators → DOMScanner)
//
// Paramètres configurables dans l'interface NinjaTrader :
//   - MinLots        : seuil de détection (défaut 50)
//   - ApproachTicks  : distance d'approche en ticks (défaut 3)
//   - AbsorptionPct  : % du volume absorbé pour confirmer (défaut 15%)

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class DOMScanner : Indicator
    {
        // --- État du DOM ---
        private Dictionary<double, int> prevAsk  = new Dictionary<double, int>();
        private Dictionary<double, int> prevBid  = new Dictionary<double, int>();

        // Anti-spam : évite de répéter la même alerte toutes les millisecondes
        private Dictionary<string, DateTime> lastAlertTime = new Dictionary<string, DateTime>();
        private const int ALERT_COOLDOWN_SEC = 10;

        // ------------------------------------------------------------------ //
        //  INITIALISATION
        // ------------------------------------------------------------------ //
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description  = "Scanne le DOM — détecte gros ordres, absorption et spoofing";
                Name         = "DOMScanner";
                Calculate    = Calculate.OnEachTick;
                IsOverlay    = true;
                DisplayInDataBox = false;

                MinLots       = 50;
                ApproachTicks = 3;
                AbsorptionPct = 15;
            }
        }

        // ------------------------------------------------------------------ //
        //  MISE À JOUR DOM (appelé à chaque changement du carnet d'ordres)
        // ------------------------------------------------------------------ //
        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            // On ne scanne qu'en temps réel (pas sur historique)
            if (State != State.Realtime) return;

            double price        = e.Price;
            int    size         = e.Volume;
            double currentPrice = Close[0];
            double tickSize     = Instrument.MasterInstrument.TickSize;
            double distTicks    = Math.Abs(price - currentPrice) / tickSize;

            bool isAsk = e.MarketDataType == MarketDataType.Ask;
            var  side  = isAsk ? prevAsk : prevBid;
            string label = isAsk ? "VENTE" : "ACHAT";

            AnalyseLevel(price, size, label, distTicks, side);
        }

        // ------------------------------------------------------------------ //
        //  LOGIQUE DE DÉTECTION
        // ------------------------------------------------------------------ //
        private void AnalyseLevel(double price, int size, string label,
                                  double distTicks, Dictionary<double, int> side)
        {
            bool proche = distTicks <= ApproachTicks;

            if (size >= MinLots)
            {
                if (!side.ContainsKey(price))
                {
                    // Nouvel ordre gros détecté
                    side[price] = size;
                    if (proche)
                        Alerter(price, size, label, "NOUVEAU", distTicks);
                }
                else
                {
                    int    prev    = side[price];
                    double absorbé = prev > 0 ? (double)(prev - size) / prev * 100.0 : 0;

                    // Absorption progressive (le volume baisse doucement) + prix proche
                    // → c'est un ordre réel qui absorbe la pression adverse
                    if (proche && absorbé >= AbsorptionPct && absorbé < 95)
                        Alerter(price, size, label, "ABSORPTION", distTicks);

                    side[price] = size;
                }
            }
            else if (side.ContainsKey(price))
            {
                int prev = side[price];

                // Gros ordre qui disparaît d'un coup quand le prix s'approche → SPOOF
                if (prev >= MinLots && size < 5 && proche)
                    Alerter(price, prev, label, "SPOOF", distTicks);

                side.Remove(price);
            }
        }

        // ------------------------------------------------------------------ //
        //  AFFICHAGE ET ALERTE
        // ------------------------------------------------------------------ //
        private void Alerter(double price, int size, string label, string type, double distTicks)
        {
            // Anti-spam
            string key = $"{type}_{label}_{price:F2}";
            if (lastAlertTime.TryGetValue(key, out DateTime last) &&
                (DateTime.Now - last).TotalSeconds < ALERT_COOLDOWN_SEC)
                return;
            lastAlertTime[key] = DateTime.Now;

            // Message lisible
            string msg = type switch
            {
                "ABSORPTION" => $"▶ ABSORPTION {label}  —  {size} lots à {price:F2}  ({distTicks:F0} ticks)",
                "SPOOF"      => $"⚠ SPOOF  —  {size} lots disparus à {price:F2}  ({distTicks:F0} ticks)",
                _            => $"◆ GROS ORDRE {label}  —  {size} lots à {price:F2}  ({distTicks:F0} ticks)"
            };

            // Couleur selon le type
            Brush couleur = type == "SPOOF"          ? Brushes.OrangeRed
                          : label == "ACHAT"          ? Brushes.LimeGreen
                                                      : Brushes.Crimson;

            // Texte overlay en haut à gauche du chart
            Draw.TextFixed(this, "dom_" + type,
                msg,
                TextPosition.TopLeft,
                couleur,
                new SimpleFont("Consolas", 12) { Bold = true },
                Brushes.Transparent,
                Brushes.Black,
                80);

            // Alerte sonore uniquement sur absorption réelle (pas sur chaque nouveau gros ordre)
            if (type == "ABSORPTION")
                Alert("DOMAlert", Priority.High, msg,
                    NinjaTrader.Core.Globals.InstallDir + @"\sounds\Alert2.wav",
                    10, couleur, Brushes.Black);

            // Log dans la fenêtre Output de NinjaTrader (View → Output)
            Print($"[DOMScanner] {DateTime.Now:HH:mm:ss}  {msg}");
        }

        // ------------------------------------------------------------------ //
        //  PARAMÈTRES EXPOSÉS DANS L'INTERFACE NINJATRADER
        // ------------------------------------------------------------------ //

        [Range(10, 500), NinjaScriptProperty]
        [Display(Name        = "Seuil minimum (lots)",
                 Description = "Taille minimale d'un ordre pour déclencher la surveillance",
                 Order = 1, GroupName = "DOMScanner")]
        public int MinLots { get; set; }

        [Range(1, 15), NinjaScriptProperty]
        [Display(Name        = "Distance d'approche (ticks)",
                 Description = "Nombre de ticks entre le prix et l'ordre pour considérer qu'il est 'proche'",
                 Order = 2, GroupName = "DOMScanner")]
        public int ApproachTicks { get; set; }

        [Range(5, 50), NinjaScriptProperty]
        [Display(Name        = "% absorption déclencheur",
                 Description = "Pourcentage du volume absorbé pour confirmer un ordre réel (vs spoof)",
                 Order = 3, GroupName = "DOMScanner")]
        public int AbsorptionPct { get; set; }
    }
}
