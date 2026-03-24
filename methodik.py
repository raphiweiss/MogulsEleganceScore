import streamlit as st

def render_methodik() -> None:
    st.markdown("## Methodik & Hintergrund")

    st.markdown(
        """
## Quantitative Approximation der FIS-Turnbewertung im Moguls-Skiing mittels Pose Tracking

Eine Untersuchung, ob die durch menschliche Bewertungen entstandene Rangliste eines Mogul-Wettbewerbs
(Winter Olympics 2026) mittels datenbasierter Analyse von Skelett-Tracking repliziert werden kann.
"""
    )

    st.markdown("## Ziel")
    st.markdown(
        """
Von den drei Aspekten der Gesamtbewertung – **Sprünge, Geschwindigkeit und Technik** – konzentriert sich
dieses Projekt auf den Bereich **Technik**, konkret die sogenannten **Turns**, die 60 % der Endbewertung ausmachen.

Die Athleten werden entlang der Piste von mehreren Wettkampfrichtern bewertet. Der Durchschnitt ihrer Bewertungen
bestimmt den Rang des Athleten.

Ziel ist es, auf Basis von aus Videoaufnahmen gewonnenen Bewegungsdaten Metriken zu berechnen, die das von Menschen
erstellte Ranking möglichst gut nachvollziehen können.
"""
    )

    st.markdown("## Forschungsfrage")
    st.markdown(
        """
Lassen sich subjektive, ästhetische Sportbewegungen durch objektive, datenbasierte Bewertungskriterien approximieren?
"""
    )

    st.markdown("## Verwendete Tools")
    st.markdown(
        """
Die Extraktion der Bewegungsdaten aus den Videos erfolgt mittels **YOLO** (Pose Estimation).  
Die Datenaufbereitung sowie die Berechnung der Metriken werden in **Python** durchgeführt.  
Die visuelle Darstellung erfolgt in Form eines interaktiven Dashboards mit **Streamlit**.
"""
    )

    st.markdown("## Python-Version und Bibliotheken")
    st.markdown(
        """
Für die Durchführung des Projekts wird **Python 3.12 oder höher** benötigt.

Verwendete Bibliotheken:
- **numpy** – Berechnung der Metriken  
- **pandas** – Datenverarbeitung  
- **matplotlib** – visuelle Datenanalyse  
- **ultralytics** – Pose Estimation mittels YOLOv8  
- **streamlit** – Erstellung des Dashboards
"""
    )

    st.markdown("## Datenquellen")
    st.markdown(
        """
Die Bewegungsdaten stammen aus Videoaufnahmen der einzelnen Athleten.  
Die Videos wurden so zugeschnitten, dass ausschließlich die **Turns** analysiert werden.

Die offiziellen Bewertungen und Ranglisten wurden von der Website der Olympischen Spiele übernommen.  
Die Bewertungskriterien basieren auf dem offiziellen **FIS Freestyle Skiing Judging Handbook**.
"""
    )

    st.markdown("## Daten & Preprocessing")
    st.markdown(
        """
Die Bewegungsdaten werden mittels **YOLO Pose Estimation** aus den Videos extrahiert.

Für jedes Video wird eine CSV-Datei erzeugt, die frameweise folgende Informationen enthält:
- Frame-Index  
- Track-ID des Athleten  
- Bounding-Box-Koordinaten  
- 2D-Pixelkoordinaten der Keypoints  
- Confidence-Werte der Detektion  

Das verwendete Modell erkennt insgesamt **17 Gelenkpunkte**.

Für die Analyse der Turns werden insbesondere folgende Keypoints verwendet:
- linke und rechte Hüfte  
- linkes und rechtes Knie  
- linke und rechte Schulter
"""
    )

    st.markdown("### Umgang mit fehlenden Werten")
    st.markdown(
        """
Beim Pose Tracking können temporär einzelne Keypoints fehlen, zum Beispiel durch:
- Okklusion durch Moguls  
- Bewegungsunschärfe  
- ungünstige Kameraperspektiven  

Fehlende Werte werden durch **lineare Interpolation** über die Zeit ersetzt, um kontinuierliche Bewegungssignale
zu gewährleisten. Frames mit dauerhaft geringer Detektionssicherheit können optional ausgeschlossen werden.
"""
    )

    st.markdown("### Signalglättung")
    st.markdown(
        """
Da numerische Ableitungen – etwa für die Smoothness-Metrik – empfindlich auf Rauschen reagieren, werden die
Positionssignale mittels **Savitzky–Golay-Filter** geglättet.

Dieser ermöglicht:
- Reduktion von Messrauschen  
- Erhalt der Bewegungsstruktur  
- stabile Berechnung von Geschwindigkeit, Beschleunigung und Jerk
"""
    )

    st.markdown("### Einschränkungen der Daten")
    st.markdown(
        """
- ausschließlich 2D-Bewegungsinformationen verfügbar  
- keine direkte Information über Tiefenbewegung oder Skikantenwinkel  
- kamerabedingte Verzerrungen der Bewegungsamplituden  
- mögliche Artefakte durch Tracking-Fehler  

Trotz dieser Einschränkungen ermöglichen die Daten eine robuste Approximation der Bewegungsqualität.
"""
    )

    st.markdown("## Datenpipeline")
    st.code("Video → YOLO Pose → Signal Processing → Metriken → Ranking → Vergleich")

    st.markdown("## Berechnung der Bewertungsmetriken")
    st.markdown(
        """
Zur Approximation der Turn-Bewertung werden mehrere Bewegungsmetriken aus den Pose-Daten berechnet.

Zentrale Grundlage ist die zeitliche Entwicklung des **Hüftmittelpunkts**, berechnet aus linker und rechter Hüfte.

Diese Trajektorie beschreibt die laterale Bewegung entlang der Mogul-Linie und bildet das zentrale Signal für die Analyse.
"""
    )

    st.markdown("### Rhythmus-Metrik (R)")
    st.markdown(
        """
Bewertet die Regelmäßigkeit der Turnabfolge.

Basierend auf:
- Detektion von Extrempunkten  
- zeitlichen Abständen zwischen Turns  
- Varianz dieser Abstände  

Ein hoher Score entspricht einer gleichmäßigen Turnfrequenz.
"""
    )

    st.markdown("### Stabilitäts-Metrik (S)")
    st.markdown(
        """
Bewertet die Stabilität der Oberkörperhaltung.

Basierend auf:
- Schulterwinkel relativ zur Horizontalen  
- Streuung dieses Winkels über die Zeit  

Hohe Werte zeigen eine ruhige und stabile Oberkörperführung.
"""
    )

    st.markdown("### Smoothness-Metrik (M)")
    st.markdown(
        """
Misst die Bewegungsglätte anhand der dritten Ableitung der Hüftbewegung (**Jerk**).

Hohe Jerk-Werte entstehen insbesondere durch:
- abrupte Richtungswechsel  
- Skidding  
- instabile Bewegungen  

Ein hoher Score entspricht flüssigen und technisch sauberen Turns.
"""
    )

    st.markdown("### Symmetrie-Metrik (Y)")
    st.markdown(
        """
Bewertet die Balance zwischen Links- und Rechtsturns.

Verglichen werden:
- mittlere Amplituden der lateralen Auslenkung  

Ein hoher Wert bedeutet eine symmetrische Bewegungsausführung.
"""
    )

    st.markdown("### Kompaktheits-Metrik (C)")
    st.markdown(
        """
Beschreibt die Stabilität der unteren Körperhaltung.

Basierend auf:
- Abstand der Knie relativ zur Hüftbreite  

Eine geringe Knieöffnung entspricht einer kompakten und kontrollierten Fahrtechnik.
"""
    )

    st.markdown("### Linienintegrität / Ausrichtung (L)")
    st.markdown(
        """
Bewertet die Stabilität der Fahrtrichtung entlang der Falllinie.

Basierend auf:
- Bewegungsvektor des Hüftmittelpunkts  
- Winkelabweichung zur idealen Linie  

Ein hoher Score bedeutet eine stabile Linienführung mit geringer seitlicher Abweichung.
"""
    )

    st.markdown("## Dashboard (aktueller Stand)")
    st.markdown(
        """
Im Dashboard kann ein Athlet ausgewählt werden.  
Das Video wird abgespielt und die berechneten Metriken werden synchron visualisiert.
"""
    )

    st.markdown("## Limitationen")
    st.markdown(
        """
- Der Vergleich basiert nur auf dem Turn-Anteil der Bewertung  
- Die offiziellen Ranglisten berücksichtigen zusätzlich Geschwindigkeit und Sprünge  
- Kurze Sequenzen von ca. 3–4 Sekunden bei rund 25 fps  
- Begrenzte Auflösung der Person im Bild  
- Sensitivität einzelner Metriken gegenüber Messfehlern
"""
    )

    st.markdown("## Entwicklungsmöglichkeiten")
    st.markdown(
        """
- Analyse kompletter Runs  
- Integration der Bewertungsbereiche Geschwindigkeit und Sprünge  
- Erstellung einer vollständig datenbasierten Rangliste
"""
    )

    st.markdown("## Danksagung")
    st.markdown(
        """
Der Dank gilt Herrn **Dr. Manuel Stein** sowie **Dr. Daniel Seebacher** und **Herrn Philipp Zimmermann**
für die Betreuung und die Einführung in dieses Thema.
"""
    )