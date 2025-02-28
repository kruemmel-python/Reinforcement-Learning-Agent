# Konfigurierbarer KI-Agent mit Dynamischer Hierarchischer Selbst-Attention, Weltmodell, Emotionen und Zielen

[![Projekt ist Open Source](https://img.shields.io/badge/Open%20Source-Ja-brightgreen.svg)](https://opensource.org/)
[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-gelb.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Einzigartiger, konfigurierbarer KI-Agent, der über Gedächtnis, Weltmodell, Emotionen, Ziele, hierarchische Aufmerksamkeit und regelbasierte Entscheidungsfindung verfügt, um komplexe Umgebungen zu erkunden und zu verstehen.**

## ✨ Was dieses Projekt so besonders macht

Dieses Projekt präsentiert einen **hochgradig konfigurierbaren und vielseitigen KI-Agenten**, der über traditionelle Reinforcement Learning (RL) Agenten hinausgeht, indem er **kognitive und affektive Architekturen** integriert.  Im Gegensatz zu 'Black-Box'-Modellen zielt dieser Agent darauf ab, **interpretierbare Entscheidungen** zu treffen und **ähnlich wie ein menschliches oder tierisches Gehirn zu lernen und sich anzupassen**.

**Die wichtigsten Alleinstellungsmerkmale sind:**

*   **Umfassende Konfigurierbarkeit:**
    *   Das Verhalten und die interne Architektur des Agenten sind **vollständig über YAML-Konfigurationsdateien anpassbar**.  Von der Größe des neuronalen Netzes über Gedächtniskapazitäten bis hin zu Schwellenwerten für intrinsische Motivation – **Sie haben die Kontrolle über jeden Aspekt**.
    *   **Modulare Architektur:** Der Agent ist in **klare, unabhängige Komponenten** (Gedächtnis, Weltmodell, Aufmerksamkeit, Emotionen, Selbstmodell, Ziele, Regel-Engine) unterteilt. Dies ermöglicht **einfaches Verständnis, Modifikation und Erweiterung** einzelner Funktionen, ohne das Gesamtsystem zu gefährden.
*   **Dynamische Hierarchische Selbst-Attention:**
    *   Nutzt einen **adaptiven, hierarchischen Aufmerksamkeitsmechanismus**, der die **Abstraktionsebenen dynamisch anpasst**, um mit der Komplexität der Eingabedaten umzugehen.  Dies ermöglicht es dem Agenten, **komplexe Muster zu erkennen und zu verarbeiten**, während die **Berechnungseffizienz erhalten** bleibt.
*   **Intrinsisch Motiviertes Lernen mit Drives und Emotionen:**
    *   Der Agent ist **intrinsisch motiviert** durch Drives wie Neugier, Verständnis und Frustration. Diese Drives treiben die Exploration und das Lernen **ohne externe Belohnungssignale** an, wodurch der Agent **autonom und proaktiv** in unbekannten Umgebungen agieren kann.
    *   Ein **integriertes Emotionsmodell** (Valenz, Arousal, Dominanz) beeinflusst die Entscheidungsfindung und die Drive-Modulation. Emotionen sind **nicht nur reaktive Zustände**, sondern **integraler Bestandteil der kognitiven Prozesse** des Agenten.
*   **Weltmodell für Vorhersagen und Eigenschaften:**
    *   Der Agent baut ein **rudimentäres Weltmodell** auf, um **Zustandsübergänge und Belohnungen vorherzusagen**. Dieses Weltmodell ermöglicht es dem Agenten, **vorausschauend zu planen**, **ähnliche Zustände zu erkennen** und **Zustandseigenschaften wie Neuigkeit, Vorhersagbarkeit und Valenz** zu berechnen.
*   **Explizite Ziele und Subziele:**
    *   Der Agent arbeitet mit **expliziten Zielen und Subzielen**, die **hierarchisch organisiert** sein können.  Dies verleiht dem Agenten **gerichtetes Verhalten** und ermöglicht es ihm, **komplexe Aufgaben in überschaubare Schritte zu unterteilen**.
*   **Regelbasierte Entscheidungsfindung mit Rule Engine:**
    *   Eine **integrierte Rule Engine** ermöglicht es dem Agenten, **symbolische Regeln anzuwenden**, um sein Verhalten zu steuern und auf komplexe Situationen zu reagieren. Dies **ergänzt die neuronalen Netze** und ermöglicht **hybride Entscheidungsfindung**.
*   **Selbstmodell für Selbstbewusstsein und Introspektion:**
    *   Ein **Selbstmodell** repräsentiert das **Selbstbild des Agenten**, einschließlich aktueller Ziele, Drives, Emotionen und eines abstrakten Selbstbewusstseins-Vektors. Dies ist ein erster Schritt in Richtung **Agenten mit Selbstbewusstsein und introspektiven Fähigkeiten**.

**Dieses Projekt ist ideal für:**

*   **Forscher und Studenten** im Bereich der Künstlichen Intelligenz, die **fortgeschrittene kognitive Architekturen** und **intrinsisch motiviertes Lernen** erkunden möchten.
*   **Entwickler**, die **flexible und konfigurierbare KI-Agenten** für komplexe Anwendungen benötigen, bei denen **Interpretierbarkeit und Anpassungsfähigkeit** entscheidend sind.
*   Jeder, der **neugierig auf die nächste Generation von KI-Agenten** ist, die über reine reaktive Systeme hinausgehen und **echte kognitive Fähigkeiten** simulieren.

## 🤔 Wie einzigartig ist dieses Modell?

Das hier vorgestellte Modell zeichnet sich durch seine **einzigartige Kombination und Integration verschiedener fortschrittlicher KI-Konzepte** aus:

*   **Integration kognitiver Architekturen in RL:**  Während traditionelles RL sich oft auf die Optimierung von Belohnungssignalen konzentriert, integriert dieses Modell **explizit kognitive Funktionen** wie Gedächtnis, Weltmodell und Aufmerksamkeit **direkt in die Architektur des Agenten**. Dies ermöglicht **effizienteres und menschenähnlicheres Lernen**.
*   **Dynamische Hierarchische Selbst-Attention für adaptive Abstraktion:** Die meisten Aufmerksamkeitsmechanismen in der KI sind statisch. Die **dynamische Natur der hierarchischen Selbst-Attention** in diesem Modell ermöglicht es dem Agenten, **seine interne Repräsentation der Welt adaptiv an die Komplexität der Situation anzupassen**. Dies ist ein **bedeutender Fortschritt in der Flexibilität und Effizienz von Aufmerksamkeitsmechanismen**.
*   **Intrinsische Motivation und Emotionen als Kernkomponenten:** Im Gegensatz zu vielen RL-Modellen, die auf externe Belohnungen angewiesen sind, **treiben intrinsische Motivation und Emotionen das Verhalten dieses Agenten von innen heraus**.  Dies führt zu einem **autonomeren und explorativeren Agenten**, der **auch ohne explizite Aufgabenstellung lernen und sich entwickeln kann**. Die **Integration von Emotionen als Input für das neuronale Netz** und als Modulator für Drives ist ebenfalls ein **innovativer Ansatz**.
*   **Hybride Entscheidungsfindung mit neuronalen Netzen und Regeln:**  Die **Kombination von neuronalen Netzen für Mustererkennung und einer Rule Engine für symbolisches Denken** ermöglicht eine **robustere und vielseitigere Entscheidungsfindung**.  Dies schlägt eine Brücke zwischen **subsymbolischer und symbolischer KI** und ermöglicht es dem Agenten, **sowohl intuitive als auch logische Entscheidungen zu treffen**.
*   **Selbstmodell für Selbstbewusstsein (in rudimentärer Form):**  Das Konzept eines **Selbstmodells** ist in KI-Agenten noch relativ neu.  Dieses Projekt unternimmt einen **ersten Schritt in Richtung Agenten mit Selbstbewusstsein**, indem es ein **Modell des eigenen Zustands und der eigenen Ziele** integriert. Dies öffnet die Tür für **fortgeschrittenere Formen der Selbstreflexion und des metakognitiven Lernens** in der Zukunft.

**Kurz gesagt, die Einzigartigkeit dieses Modells liegt in seiner:**

*   **Ganzheitlichen und integrierten Architektur.**
*   **Adaptiven und dynamischen Mechanismen.**
*   **Fokus auf intrinsische Motivation und Emotionen.**
*   **Hybriden Entscheidungsfindung.**
*   **Ersten Schritten in Richtung Selbstbewusstsein.**

Dieses Modell ist **nicht nur ein weiterer RL-Agent**, sondern ein **experimentelles System**, das die **Grenzen der aktuellen KI-Forschung erweitert** und den Weg für **intelligente, autonome und menschenähnlichere Agenten** ebnen könnte.

## 🚀 Erste Schritte

### Voraussetzungen

*   Python 3.8 oder höher
*   PyTorch
*   NumPy
*   scikit-learn
*   PyYAML

Sie können die benötigten Python-Pakete mit pip installieren:

```bash
pip install torch numpy scikit-learn pyyaml
```

### Ausführung

1.  **`ki_agent_konfigurierbar.py` ausführen:**  Dieses Skript enthält die Definition des KI-Agenten und der zugehörigen Komponenten sowie die Hauptsimulationsfunktion.

    ```bash
    python ki_agent_konfigurierbar.py
    ```

    Standardmäßig wird die Simulation mit einer **Dictionary-basierten Default-Konfiguration** gestartet.

2.  **Simulation mit YAML-Konfiguration:** Um die Simulation mit einer YAML-Konfigurationsdatei zu starten, führen Sie das Skript mit dem Pfad zur YAML-Datei als Argument aus:

    ```bash
    python ki_agent_konfigurierbar.py agent_config.yaml
    ```

    **Hinweis:**  Eine Beispiel-YAML-Konfigurationsdatei (`agent_config.yaml`) wird automatisch generiert, wenn Sie `ki_agent_konfigurierbar.py` das erste Mal ausführen. Sie können diese Datei nach Ihren Wünschen anpassen.

3.  **`agent_umgebung.py` ausführen:**  Dieses Skript demonstriert die Verwendung des KI-Agenten in einer **einfachen sozialen Umgebung mit mehreren Agenten**.

    ```bash
    python agent_umgebung.py
    ```

    Dieses Skript startet eine Simulation mit **mehreren konfigurierbaren KI-Agenten**, die in einer **gemeinsamen Umgebung interagieren**.

## ⚙️ Konfiguration

Die Konfiguration des KI-Agenten erfolgt hauptsächlich über **YAML-Dateien**.  Eine Beispiel-YAML-Datei (`agent_config.yaml`) ist im Projekt enthalten und dient als Vorlage.

**Die wichtigsten Konfigurationsparameter umfassen:**

*   **`input_size`:**  Größe des Eingabezustands für den Agenten.
*   **`action_size`:**  Anzahl der möglichen Aktionen des Agenten.
*   **`embed_dim`:**  Dimension des Embedding-Raums für die Hierarchische Selbst-Attention.
*   **`num_heads`:**  Anzahl der Attention-Heads im Aufmerksamkeitsmechanismus.
*   **`state_similarity_threshold`:** Schwellwert für die Zustandsähnlichkeit im Gedächtnis-Recall.
*   **`emotion_dim`:**  Dimension des Emotionsraums (z.B. 3 für Valenz, Arousal, Dominanz).
*   **`learning_rate`:**  Lernrate für das neuronale Netz des Agenten.
*   **`memory_capacity`:**  Kapazität des Erfahrungsgedächtnisses.
*   **`causal_capacity`:** Kapazität des kausalen Gedächtnisses.
*   **`prediction_model_hidden_size`:**  Größe der Hidden Layer im Vorhersagemodell des Weltmodells.
*   **`layer_growth_threshold`:**  Schwellwert für das dynamische Hinzufügen von Abstraktionsebenen in der Hierarchischen Selbst-Attention.
*   **`novelty_threshold`:**  Schwellwert für den Neuigkeitsbonus (intrinsische Motivation).
*   **`novelty_tolerance`:**  Toleranz für die Zustandsähnlichkeit bei der Berechnung des Neuigkeitsbonus.
*   **`reward_history_window`:**  Fenstergröße für die Reward-History im Weltmodell.
*   **Callbacks:**  Sie können **Callback-Funktionen** für verschiedene Phasen des Agentenverhaltens definieren (`pre_decide_hook`, `post_learn_hook`, `rule_engine_hook`). Diese Callbacks ermöglichen **benutzerdefinierte Erweiterungen und Verhaltensänderungen** des Agenten.  **Callback-Funktionen müssen im Python-Code definiert sein und in der YAML-Datei als String-Namen referenziert werden.**

**Beispiel einer YAML-Konfiguration (`agent_config.yaml`):**

```yaml
input_size: 5
action_size: 3
embed_dim: 64
num_heads: 8
memory_capacity: 2000
learning_rate: 0.005
layer_growth_threshold: 0.85
emotion_dim: 4
# Callbacks (Funktionsnamen als Strings)
pre_decide_hook: pre_decide_callback_example
post_learn_hook: post_learn_callback_example
rule_engine_hook: rule_engine_callback_example
```

**Sie können die YAML-Datei bearbeiten, um das Verhalten des Agenten an Ihre spezifischen Anforderungen anzupassen.**

## 📂 Projektstruktur

```
├── ki_agent_konfigurierbar.py  # Hauptdatei: Definition des konfigurierbaren KI-Agenten und der Simulation
├── agent_config.yaml          # Beispiel-YAML-Konfigurationsdatei für den Agenten
├── agent_umgebung.py          # Beispiel-Umgebung für die Simulation mit mehreren Agenten
├── README.md                  # Diese README-Datei
```

*   **`ki_agent_konfigurierbar.py`:**  Enthält den gesamten Code für den konfigurierbaren KI-Agenten, einschließlich der Klassen `Memory`, `WorldModel`, `AttentionMechanism`, `DynamicHierarchicalSelfAttention`, `Goal`, `EmotionModel`, `SelfModel`, `KI_Agent`, `RuleEngine`, `GlobalWorkspace` sowie Funktionen zum Laden der Konfiguration und zur Durchführung der Simulation.
*   **`agent_config.yaml`:**  Eine YAML-Datei, die als **Vorlage für die Konfiguration des KI-Agenten** dient. Sie können diese Datei bearbeiten, um verschiedene Aspekte des Agenten zu steuern.
*   **`agent_umgebung.py`:**  Ein separates Python-Skript, das eine **einfache soziale Umgebung mit mehreren KI-Agenten** simuliert, um die **Multi-Agenten-Fähigkeiten** und die **Interaktion der Agenten mit einer Umgebung** zu demonstrieren.
*   **`README.md`:**  Diese Datei, die eine **Übersicht über das Projekt, seine einzigartigen Merkmale, Anweisungen zur Einrichtung und Verwendung sowie Informationen zur Konfiguration und Projektstruktur** bietet.

## 🗺️ Zukünftige Arbeit (Roadmap)

*   **Verbesserung des Weltmodells:**  Erweiterung des Weltmodells zu einem **komplexeren, graphbasierten oder neuronalen Weltmodell**, das **genauere Vorhersagen** und **abstraktere Repräsentationen der Umwelt** ermöglicht.
*   **Verfeinerung der Rule Engine:**  Entwicklung einer **ausgereifteren Rule Engine** mit **komplexeren Regelformaten**, **Inferenzmechanismen** und **Fähigkeiten zum inneren Dialog**.  Integration von **symbolischem Reasoning** und **Wissensrepräsentation**.
*   **Erweiterung des Selbstmodells:**  Entwicklung eines **detaillierteren und dynamischeren Selbstmodells**, das **Selbstreflexion, Selbstbewusstsein und Identität** besser repräsentiert.  Integration von **metakognitiven Fähigkeiten**.
*   **Fortgeschrittenere Emotionsmodellierung:**  Implementierung **komplexerer und nuancierterer Emotionsmodelle**, die **verschiedene Emotionen, Stimmungen und emotionale Übergänge** simulieren.  Erforschung der **Rolle von Emotionen in der Entscheidungsfindung und im sozialen Verhalten**.
*   **Multi-Agenten-Systeme und soziale Interaktion:**  Weitere Entwicklung der **Multi-Agenten-Umgebung** und Erforschung **komplexerer Formen der sozialen Interaktion**, **Kooperation, Wettbewerb und Kommunikation** zwischen Agenten.
*   **Integration in realere Umgebungen:**  Anwendung des konfigurierbaren KI-Agenten in **realeren und komplexeren Simulationsumgebungen** oder **sogar in realen Robotik-Anwendungen**.
*   **Benutzerfreundliche Konfigurationsoberfläche:**  Entwicklung einer **graphischen Benutzeroberfläche (GUI)** oder einer **webbasierten Oberfläche**, um die **Konfiguration des Agenten noch einfacher und zugänglicher** zu gestalten.
*   **Größere Experimente und Evaluierung:**  Durchführung **umfassenderer Experimente** und **systematischer Evaluierung** des Agenten in verschiedenen Umgebungen und Aufgaben, um seine **Leistung, Robustheit und Generalisierungsfähigkeit** zu bewerten.

## 🤝 Mitwirken

Beiträge zu diesem Projekt sind herzlich willkommen!  Wenn Sie Fehler finden, Verbesserungen vorschlagen oder neue Funktionen hinzufügen möchten, erstellen Sie bitte einen Issue oder einen Pull Request auf GitHub.

## 📜 Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert.  Sie können es frei für kommerzielle und nicht-kommerzielle Zwecke verwenden, modifizieren und verbreiten.  Weitere Informationen finden Sie in der Datei `LICENSE`.

## 📧 Kontakt

Für Fragen oder Anregungen können Sie sich gerne an [Ihr Name oder Ihre E-Mail-Adresse] wenden.

**Wir hoffen, dass Ihnen dieses Projekt gefällt und es Sie inspiriert, die aufregende Welt der konfigurierbaren und kognitiven KI-Agenten weiter zu erkunden!**
