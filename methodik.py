import streamlit as st
def render_methodik() -> None:

    # --------------------------------------------------
    # Motivation & Ziel
    # --------------------------------------------------
    st.markdown("### Motivation & Ziel")
    st.markdown(
        """
Diese Arbeit untersucht, inwiefern die Bewertung von Turns im Moguls-Skiing
durch datenbasierte Bewegungsmetriken nachvollzogen werden kann.

Grundlage bilden Videoaufnahmen aus den **Final-Läufen der Olympischen Spiele 2026**.
Der Fokus liegt auf der technischen Bewertung der Turns, die rund **60 %** der Gesamtwertung ausmachen.

Ziel ist es, interpretierbare Bewegungsmetriken aus Videodaten abzuleiten und deren Zusammenhang
mit den offiziellen Wettkampfscores zu analysieren.
"""
    )

    st.divider()

    # --------------------------------------------------
    # Daten
    # --------------------------------------------------
    st.markdown("### Daten & Pose Estimation")
    st.markdown(
        """
- Videoaufnahmen einzelner Athletinnen und Athleten (ca. 3–4 Sekunden, ~25 fps, Full HD)  
- Pose Estimation mittels **YOLOv8n** (Ultralytics)  
- Pro Frame extrahiert:
  - 17 Keypoints (2D)
  - Bounding Box
  - Confidence Scores  

Relevante Gelenkpunkte:
- Hüfte (links/rechts)  
- Knie (links/rechts)  
- Schulter (links/rechts)
"""
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    st.markdown("### Preprocessing")
    st.markdown(
        """
- **Interpolation:** zeitliche Interpolation fehlender Keypoints  
- **Glättung:** Savitzky-Golay-Filter  
- **Referenzsignal:** Hüftmittelpunkt  

Einschränkungen:
- ausschliesslich 2D-Daten  
- mögliche Tracking-Artefakte  
- keine Tiefeninformation
"""
    )

    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------
    st.markdown("### Pipeline")
    st.markdown(
        "**Video → Pose Estimation → Preprocessing → Metriken → MES → Score Comparison**"
    )

    st.divider()

    # --------------------------------------------------
    # MES Formel
    # --------------------------------------------------
    st.markdown("### Mogul Elegance Score (MES)")
    st.latex(r"MES = 10 \cdot (R + S + C + Y + M + L)")

    st.markdown(
        """
Alle Metriken sind auf **[0,1] normiert** und gleich gewichtet.

- **R** = Rhythmus  
- **S** = Stabilität  
- **C** = Kompaktheit  
- **Y** = Symmetrie  
- **M** = Smoothness  
- **L** = Line Integrity  

Ein hoher MES-Wert entspricht einer rhythmischen, stabilen, kompakten und kontrollierten Fahrweise.
"""
    )

    st.divider()

    # --------------------------------------------------
    # Intro Metriken
    # --------------------------------------------------
    st.markdown("### Bewegungsmetriken")
    st.markdown(
        """
Die Metriken basieren auf der zeitlichen Entwicklung des Hüftmittelpunkts sowie weiterer relevanter Keypoints (Knie und Schultern) und daraus abgeleiteter Grössen wie Gelenkwinkel und Bewegungsvektoren.

Wichtig: Die Liniencharts zeigen nicht direkt den Score, sondern das zugrunde liegende Bewegungssignal.
Form, Peaks und Stabilität der Kurven lassen sich direkt als Bewegungsqualität interpretieren.
"""
    )

    # --------------------------------------------------
    # Rhythmus
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Rhythmus (R)")
    st.markdown(
        """
**Definition:**  
Regelmässigkeit der Turnabfolge.

**Technik:**  
- laterale Hüftbewegung  
- Detektion von Maxima/Minima  
- Zeitabstände zwischen Turns  
- Varianz der Abstände  

**Interpretation (Line Chart):**  
- periodische Wellen → konstanter Rhythmus  
- unregelmässige Peaks → inkonsistente Turns  
"""
    )

    # --------------------------------------------------
    # Stabilität
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Stabilität (S)")
    st.markdown(
        """
**Definition:**  
Stabilität des Oberkörpers.

**Technik:**  
- Schulterwinkel  
- zeitliche Varianz  

**Interpretation (Line Chart):**  
- flache Linie → stabil  
- Ausschläge → Instabilität  
"""
    )

    # --------------------------------------------------
    # Kompaktheit
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Kompaktheit (C)")
    st.markdown(
        """
**Definition:**  
Knieabstand relativ zur Hüftbreite.

**Technik:**  
- Kniedistanz  
- Normalisierung  

**Interpretation (Line Chart):**  
- tiefe Werte → kompakt  
- hohe Werte → ausladend  
"""
    )

    # --------------------------------------------------
    # Symmetrie
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Symmetrie (Y)")
    st.markdown(
        """
**Definition:**  
Balance zwischen Links- und Rechtsturns.

**Technik:**  
- laterale Hüftbewegung  
- Rolling Window  

**Interpretation (Line Chart):**  
- hohe Werte → lokal symmetrisch  
- tiefe Werte → eine Seite dominiert  

**Wichtig:**  
Zeigt lokale Symmetrie, nicht die gesamte Sequenz.
"""
    )

    # --------------------------------------------------
    # Smoothness
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Smoothness (M)")
    st.markdown(
        """
**Definition:**  
Jerk (Änderung der Beschleunigung).

**Technik:**  
- Ableitungen der Hüftbewegung  
- Berechnung des Jerk  

**Interpretation (Line Chart):**  
- ruhiger Verlauf → flüssig  
- Peaks → ruckartig  

**Wichtig:**  
Misst nicht Geschwindigkeit, sondern Abruptheit.
"""
    )

    # --------------------------------------------------
    # Line Integrity
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("#### Line Integrity (L)")
    st.markdown(
        """
**Definition:**  
Abweichung von der Falllinie.

**Technik:**  
- Bewegungsvektor  
- Winkelabweichung  

**Interpretation (Line Chart):**  
- ruhiger Verlauf → saubere Linie  
- schwankend → Abdriften  
"""
    )

    st.divider()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    st.markdown("### Evaluation")
    st.markdown(
        """
- Synchronisierte Darstellung von Video, Pose Estimation und Zeitreihen  
- Visuelle Analyse mit Zeitreihen, Radar-Chart und Scatterplots 
- Vergleich mit offiziellen Turns-Scores  
- Regressionsanalyse (Metriken → Score)  
"""
    )

    # --------------------------------------------------
    # Erkenntnisse
    # --------------------------------------------------
    st.markdown("### Zentrale Erkenntnisse")
    st.markdown(
        """
Teilweise erfolgreiche Approximation der Bewertung. Am besten interpretierbar und korrelierend mit dem offiziellen Score:
- Rhythmus 
- Stabilität  
- Kompaktheit  
 
Schwieriger:
- Symmetrie  
- Smoothness  
- Line Integrity
"""
    )

    # --------------------------------------------------
    # Limitationen
    # --------------------------------------------------
    st.markdown("### Limitationen")
    st.markdown(
        """
- Fokus ausschliesslich auf Turns
- Kurze Videosequenzen  
- Begrenzte Framerate (~25 fps)
- Geringe Auflösung der Person im Bild
- Anfälligkeit für Messfehler bei einzelnen Metriken
- 2D-Daten ohne Tiefeninformation
"""
    )

    # --------------------------------------------------
    # Danksagung
    # --------------------------------------------------
    st.markdown("### Danksagung")
    st.markdown(
        """
Wir danken **Dr. Manuel Stein**, **Dr. Daniel Seebacher** sowie **Philipp Zimmermann**
für die inspirierende Vorlesung, die wertvollen Impulse und die fachliche Unterstützung im Rahmen dieses Projekts.

Aaron Gitz  
Simon Kim  
Raphael Weiss  

28. März 2026
"""
    )