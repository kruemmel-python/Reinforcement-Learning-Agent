# ki_agent_konfigurierbar.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from sklearn.metrics.pairwise import cosine_similarity # Import für Kosinus-Ähnlichkeit
import yaml # Import für YAML

# 🧠 Gedächtnis: Speichert Erlebnisse & Erkenntnisse
class Memory:
    def __init__(self, capacity=1000, causal_capacity=100): # Kapazität für kausales Gedächtnis hinzugefügt
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
        self.causal_memory = collections.deque(maxlen=causal_capacity) # Separates kausales Gedächtnis mit Capacity

    def store(self, experience):
        """Speichert eine Erinnerung (Erlebnis, Emotion, Reaktion)."""
        self.memory.append(experience)

    def recall(self):
        """Ruft zufällige Erinnerung ab."""
        if len(self.memory) > 0:
            return random.choice(self.memory)
        return None

    def store_causal(self, state, action, next_state, reward): # Reward hinzugefügt
        """Speichert eine kausale Erinnerung (Zustand, Aktion, nächster Zustand, Belohnung)."""
        self.causal_memory.append((state, action, next_state, reward)) # Reward gespeichert

    def recall_causal(self):
        """Ruft zufällige kausale Erinnerung ab."""
        if len(self.causal_memory) > 0:
            return random.choice(self.causal_memory)
        return None

    def recall_similar(self, state, tolerance=0.5):
        """Ruft ähnliche Erinnerung ab, basierend auf Zustandsähnlichkeit."""
        similar_memories = []
        for memory in self.memory:
            if memory and torch.allclose(torch.tensor(memory[0]), state.squeeze(0), atol=tolerance):
                similar_memories.append(memory)
        if similar_memories:
            return random.choice(similar_memories)
        return None


# 🌐 Weltmodell (Graph-basiert - rudimentär)
class WorldModel:
    def __init__(self, state_size=3, action_size=1, prediction_model_hidden_size=16, learning_rate=0.001): # Prediction Model Parameter, action_size korrigiert zu 1
        self.graph = {}  # Graph als Dictionary: Zustand -> {Aktion: [(nächster_Zustand, Belohnung, Zustandseigenschaften)]}
        self.state_visit_counts = collections.defaultdict(int) # Zähler für Zustandbesuche (vorerst beibehalten)
        self.state_reward_history = collections.defaultdict(collections.deque) # Reward-History pro Zustand (für Valence - vorerst beibehalten)
        self.reward_history_window = 10 # Fenstergröße für Reward-History

        # **Neu: Vorhersagemodell (Prediction Model)**
        self.prediction_model = nn.Sequential(
            nn.Linear(state_size + action_size, prediction_model_hidden_size), # Zustand + Aktion als Input
            nn.ReLU(),
            nn.Linear(prediction_model_hidden_size, state_size + 1) # Output: nächster Zustand + Belohnung (1 Wert für Belohnung)
        ).float() # Stelle sicher, dass das Modell Float verwendet
        self.prediction_model_optimizer = optim.Adam(self.prediction_model.parameters(), lr=learning_rate)
        self.prediction_model_loss_fn = nn.MSELoss() # MSE Loss für Zustand und Belohnung


    def add_transition(self, state, action, next_state, reward):
        """Fügt eine Transition zum Weltmodell-Graphen hinzu, trainiert Vorhersagemodell & speichert Zustandseigenschaften."""
        state_tuple = tuple(state.tolist()) # Zustand als Tuple für Dictionary-Key
        next_state_tuple = tuple(next_state.tolist())

        # Zustandbesuchs-Zähler aktualisieren (vorerst beibehalten)
        self.state_visit_counts[state_tuple] += 1

        # Reward-History aktualisieren (vorerst beibehalten)
        self.state_reward_history[state_tuple].append(reward)
        if len(self.state_reward_history[state_tuple]) > self.reward_history_window:
            self.state_reward_history[state_tuple].popleft() # Älteste Belohnung entfernen

        # **Neu: Vorhersagemodell trainieren**
        self.train_prediction_model(state, action, next_state, reward)

        # **Neu: Zustandseigenschaften (ML-basiert) - werden DYNAMISCH in calculate_state_properties berechnet**
        state_properties = self.calculate_state_properties(state, action) # Berechne Eigenschaften (jetzt dynamisch)
        next_state_properties = self.calculate_state_properties(next_state, action) # Eigenschaften für nächsten Zustand (auch dynamisch)


        if state_tuple not in self.graph:
            self.graph[state_tuple] = {}
        if action not in self.graph[state_tuple]:
            self.graph[state_tuple][action] = []
        self.graph[state_tuple][action].append((next_state_tuple, reward, next_state_properties)) # Speichere Eigenschaften


    def train_prediction_model(self, state, action, next_state, reward):
        """Trainiert das Vorhersagemodell mit einer einzelnen Transition."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Batch-Dimension
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0) # Batch-Dimension
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) # Batch-Dimension
        reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0) # Batch-Dimension

        # Kombiniere Zustand und Aktion zum Input des Modells
        model_input = torch.cat([state_tensor, action_tensor], dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren

        # Zielwerte: Nächster Zustand und Belohnung
        target_output = torch.cat([next_state_tensor, reward_tensor], dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren

        # Vorhersage und Loss berechnen
        predicted_output = self.prediction_model(model_input)
        loss = self.prediction_model_loss_fn(predicted_output, target_output)

        # Backpropagation und Optimierung
        self.prediction_model_optimizer.zero_grad()
        loss.backward()
        self.prediction_model_optimizer.step()


    def get_transitions(self, state):
        """Ruft Transitionen für einen gegebenen Zustand ab."""
        state_tuple = tuple(state.tolist())
        return self.graph.get(state_tuple, {})

    def recall_similar_state(self, current_state, tolerance=0.5):
        """
        Ruft einen ähnlichen Zustand aus dem Weltmodell ab, basierend auf Zustandsähnlichkeit.
        Gibt den ähnlichen Zustand (als Tensor) und seine Eigenschaften oder None zurück.
        """
        current_state_tensor = torch.tensor(current_state)
        for state_tuple in self.graph:
            state_tensor = torch.tensor(state_tuple)
            if torch.allclose(state_tensor, current_state_tensor, atol=tolerance):
                # **Neu: Zustandseigenschaften zurückgeben**
                transitions = self.graph.get(state_tuple, {})
                if transitions: # Transitionen vorhanden
                    _, _, properties = transitions[list(transitions.keys())[0]][0] # Erste Transition, erste Aktion, Zustandseigenschaften
                    return state_tensor.unsqueeze(0), properties # Rückgabe als Tensor mit Batch-Dimension und Eigenschaften
                return state_tensor.unsqueeze(0), {} # Zustand gefunden, aber keine Eigenschaften (sollte nicht passieren, aber sicherheitshalber)
        return None, {} # Kein ähnlicher Zustand gefunden, leere Eigenschaften


    def calculate_state_properties(self, state, action):
        """
        Berechnet Zustandseigenschaften (ML-basiert).
        Aktuell: novelty_score, frustration_level, predictability (ML-basiert), valence (ML-basiert).
        """
        state_tuple = tuple(state.tolist())
        properties = {}
        properties['novelty_score'] = 1.0 # Neuigkeitswert (vereinfacht: immer 1.0 für neue Zustände - vorerst beibehalten)
        properties['frustration_level'] = 0.0 # Frustration (vorerst statisch/heuristisch)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Batch-Dimension
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0) # Batch-Dimension
        model_input = torch.cat([state_tensor, action_tensor], dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren

        # **Neu: Predictability (ML-basiert - Vorhersagefehler)**
        with torch.no_grad(): # Keine Gradientenberechnung für Zustandseigenschaften-Berechnung
            predicted_output = self.prediction_model(model_input)
            predicted_next_state = predicted_output[:, :state_tensor.shape[1]] # Vorhergesagter Zustand
            # Annahme: Tatsächlicher nächster Zustand ist unbekannt zum Zeitpunkt der Berechnung der Eigenschaften des *aktuellen* Zustands.
            # Daher Approximation:  Wir nutzen den *vorhergesagten* nächsten Zustand als Referenz für den aktuellen Zustand.
            # **WICHTIG:** Dies ist eine Vereinfachung und könnte in Zukunft verfeinert werden.
            state_prediction_error = self.prediction_model_loss_fn(predicted_next_state, state_tensor) # MSE-Loss zwischen vorhergesagtem Zustand und aktuellem Zustand (als Approximation des Fehlers)
            properties['predictability'] = 1.0 / (state_prediction_error.item() + 1e-6) # Inverser Fehler (kleiner Fehler -> hohe Predictability) + kleine Konstante für Stabilität


        # **Neu: Valence (ML-basiert - Vorhergesagte Belohnung)**
        with torch.no_grad(): # Keine Gradientenberechnung
            predicted_output = self.prediction_model(model_input)
            predicted_reward = predicted_output[:, state_tensor.shape[1]:] # Vorhergesagte Belohnung
            properties['valence'] = predicted_reward.item() # Vorhergesagte Belohnung als Valenz


        return properties



# 🤖 Aufmerksamkeitsmechanismus (Self-Attention)
class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).float() # batch_first=True hinzugefügt, Float-Typ erzwingen

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query.float() # Stelle sicher, dass key Float ist
        if value is None:
            value = query.float() # Stelle sicher, dass value Float ist
        query = query.float() # Stelle sicher, dass query Float ist
        attention_output, _ = self.attention(query, key, value)
        return attention_output


# 🧠 Dynamische Hierarchische Selbst-Attention
class DynamicHierarchicalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_layers=1, layer_growth_threshold=0.9): # Start mit einer Ebene, Schwellwert für Ebenenwachstum
        super().__init__()
        self.attention_layers = nn.ModuleList([AttentionMechanism(embed_dim, num_heads) for _ in range(initial_layers)])
        self.level_projections = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim).float() for _ in range(initial_layers-1)]) # Projektionen zwischen den Ebenen, Float-Typ
        self.inter_level_attention = nn.ModuleList([AttentionMechanism(embed_dim, num_heads) for _ in range(initial_layers-1)]) # Attention-Mechanismen ZWISCHEN den Ebenen
        self.layer_growth_threshold = layer_growth_threshold # Schwellwert für das Hinzufügen neuer Ebenen
        self.num_layers = initial_layers
        self.relu = nn.ReLU() # ReLU Aktivierungsfunktion als Klassenattribut
        self.embed_dim = embed_dim # Speichere embed_dim

    def forward(self, abstract_state):
        """Dynamische Hierarchische Selbst-Attention mit adaptiver Ebenenanzahl."""
        level_outputs = []
        current_level_input = abstract_state.float() # Stelle sicher, dass Input Float ist

        for i in range(self.num_layers): # Dynamische Anzahl Ebenen
            attention_layer = self.attention_layers[i] # Ebene dynamisch auswählen
            attention_output = attention_layer(current_level_input)
            level_outputs.append(attention_output)

            if i < self.num_layers - 1: # Für Ebenen UNTERHALB der obersten Ebene:
                # Gerichtete Attention zwischen den Ebenen (wie zuvor)
                query_next_level = attention_output
                key_value_current_level = level_outputs[0] # Beispiel: Output der ERSTEN Ebene als Key/Value
                inter_level_output = self.inter_level_attention[i](query_next_level, key_value_current_level, key_value_current_level)

                combined_output = torch.cat([attention_output, inter_level_output], dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren
                current_level_input = self.level_projections[i](combined_output)
                current_level_input = self.relu(current_level_input) # ReLU Aktivierungsfunktion nutzen

            # **Neu: Dynamisches Ebenenwachstum basierend auf "Komplexität" oder "Unsicherheit"**
            if i == self.num_layers - 1: # Nach der obersten Ebene prüfen, ob weitere Ebene benötigt
                complexity_metric = self.calculate_complexity(attention_output) # Funktion zur Messung der Komplexität/Unsicherheit (muss implementiert werden)
                if complexity_metric > self.layer_growth_threshold:
                    self.add_layer() # Füge dynamisch eine neue Ebene hinzu

        return level_outputs[-1] # Output der obersten Ebene

    def calculate_complexity(self, level_output):
        """
        Misst die "Komplexität" oder "Unsicherheit" des Outputs einer Ebene.
        Aktuelle Implementierung: Varianz des Outputs.
        """
        # Beispiel: Varianz des Outputs als Komplexitätsmaß
        return torch.var(level_output).item() # Varianz als Beispiel

    def add_layer(self):
        """Fügt dynamisch eine neue Abstraktionsebene hinzu."""
        embed_dim = self.attention_layers[-1].attention.embed_dim
        num_heads = self.attention_layers[-1].attention.num_heads
        new_attention_layer = AttentionMechanism(embed_dim, num_heads) # Parameter der letzten Ebene übernehmen
        self.attention_layers.append(new_attention_layer)

        # Projektionen und Inter-Level-Attention nur hinzufügen, wenn es MEHR als eine Ebene gibt
        if self.num_layers > 0:
            self.level_projections.append(nn.Linear(embed_dim * 2, embed_dim).float()) # Float-Typ
            self.inter_level_attention.append(AttentionMechanism(embed_dim, num_heads))

        self.num_layers += 1
        print(f"INFO: Dynamisch neue Abstraktionsebene hinzugefügt. Aktuelle Ebenenanzahl: {self.num_layers}")


# 🎯 Ziel-Klasse
class Goal:
    def __init__(self, name, priority="mittel", subgoals=None): # Subziele hinzugefügt
        self.name = name
        self.priority = priority # "hoch", "mittel", "niedrig"
        self.status = "aktiv" # "aktiv", "inaktiv", "erreicht"
        self.subgoals = subgoals if subgoals is not None else [] # Liste von Subzielen (Strings)

    def __str__(self):
        return f"Ziel: {self.name} (Priorität: {self.priority}, Status: {self.status}, Subziele: {self.subgoals})"

# 🎭 **Neu: Emotionsmodell - ERWEITERT um Dominanz**
class EmotionModel(nn.Module):
    def __init__(self, emotion_dim=3): # Beispiel: 3 dimensionale Emotionsraum (Valenz, Arousal, Dominanz)
        super().__init__()
        self.emotion_dim = emotion_dim
        self.emotions = nn.Parameter(torch.zeros(emotion_dim).float()) # Emotionen als Parameter, initialisiert mit Null

    def update_emotions(self, reward, predictability): # Predictability als Input
        """Aktualisiert Emotionen basierend auf Belohnung UND Predictability."""
        # **Erweiterte Emotionsaktualisierung:**
        # Valenz: Reaktion auf Belohnung (wie zuvor)
        # Fehler behoben: In-place Operation durch out-of-place ersetzt
        self.emotions = nn.Parameter(torch.tensor([self.emotions[0] + reward * 0.1, self.emotions[1], self.emotions[2]])) # Valenz-Dimension (angenommen Dimension 0 ist Valenz)
        self.emotions.data[0] = torch.clamp(self.emotions.data[0], -1.0, 1.0) # Valenz begrenzen (-1 bis 1)

        # Arousal (Erregung/Intensität) basierend auf Predictability (wie zuvor)
        arousal_intensity = 1.0 - predictability # Je niedriger Predictability, desto höher Arousal
        self.emotions.data[1] += arousal_intensity * 0.05 # Arousal-Dimension (angenommen Dimension 1 ist Arousal)
        self.emotions.data[1] = torch.clamp(self.emotions.data[1], 0.0, 1.0) # Arousal begrenzen (0 bis 1)
        self.emotions.data[1] *= 0.99 # Langsamer, natürlicher "Decay" von Arousal über die Zeit (Vergessen von Aufregung)

        # **Neu: Dominanz (Kontrolle/Macht) basierend auf Belohnung und Predictability (Beispielhaft)**
        # Hohe Belohnung UND hohe Predictability -> Gefühl von Dominanz/Kontrolle verstärken
        dominance_intensity = (reward + predictability) / 2.0 # Kombinierter Faktor (vereinfacht)
        self.emotions.data[2] += dominance_intensity * 0.05 # Dominanz-Dimension (angenommen Dimension 2 ist Dominanz)
        self.emotions.data[2] = torch.clamp(self.emotions.data[2], -1.0, 1.0) # Dominanz begrenzen (-1 bis 1)
        self.emotions.data[2] *= 0.99 # Langsamer Decay für Dominanz


    def get_emotions(self):
        """Gibt den aktuellen Emotionszustand zurück."""
        return self.emotions

# **NEU: Selbstmodell - Repräsentation des Agenten SELBST**
class SelfModel:
    def __init__(self):
        """Initialisiert das Selbstmodell."""
        self.current_goal_key = "explore" # Startziel
        self.current_subgoal = None # Aktuelles Subziel (wird später gesetzt)
        self.drives = {
            "curiosity": random.uniform(0.1, 0.5),
            "understanding": random.uniform(0.1, 0.3),
            "frustration": random.uniform(0.0, 0.2)
        }
        self.emotions = torch.zeros(3).float() # Initialer Emotionszustand (Valenz, Arousal, Dominanz)
        self.self_awareness_vector = torch.randn(32).float() # Beispiel: Abstrakter Selbstbewusstseins-Vektor

    def update_goal(self, goal_key, subgoal=None):
        """Aktualisiert das aktuelle Ziel und Subziel im Selbstmodell."""
        self.current_goal_key = goal_key
        self.current_subgoal = subgoal

    def update_drives(self, curiosity=None, understanding=None, frustration=None):
        """Aktualisiert die Drives im Selbstmodell."""
        if curiosity is not None: self.drives["curiosity"] = curiosity
        if understanding is not None: self.drives["understanding"] = understanding
        if frustration is not None: self.drives["frustration"] = frustration

    def update_emotions(self, emotions):
        """Aktualisiert den Emotionszustand im Selbstmodell."""
        self.emotions = emotions

    def update_self_awareness(self, awareness_signal):
        """Aktualisiert den Selbstbewusstseins-Vektor (Beispielhaft)."""
        # Hier könnte komplexere Logik stehen, z.B. basierend auf Reflexion, Kohärenz, etc.
        self.self_awareness_vector += awareness_signal * 0.1
        self.self_awareness_vector = torch.clamp(self.self_awareness_vector, -1.0, 1.0) # Begrenzen

    def get_self_state(self):
        """Gibt den aktuellen Zustand des Selbstmodells als Dictionary zurück."""
        return {
            "current_goal_key": self.current_goal_key,
            "current_subgoal": self.current_subgoal,
            "drives": self.drives,
            "emotions": self.emotions,
            "self_awareness_vector": self.self_awareness_vector
        }


# 🤖 KI-Agent mit RL, Feedback-Loop, Hierarchischer Selbst-Attention, Drives, Weltmodell & Zielen
class KI_Agent(nn.Module): # Annahme: KI_Agent Klasse aus vorherigem Code ist vorhanden
    def __init__(self, config, workspace): # **Neu: Workspace als Parameter hinzugefügt**
        super().__init__()

        # **Konfiguration aus Dictionary extrahieren (oder Defaultwerte nutzen)**
        input_size = config.get('input_size', 3) # Default: 3
        action_size = config.get('action_size', 2) # Default: 2
        embed_dim = config.get('embed_dim', 32) # Default: 32
        num_heads = config.get('num_heads', 4) # Default: 4
        state_similarity_threshold = config.get('state_similarity_threshold', 0.5) # Default: 0.5
        emotion_dim = config.get('emotion_dim', 3) # Default: 3
        learning_rate = config.get('learning_rate', 0.01) # Default: 0.01
        memory_capacity = config.get('memory_capacity', 1000) # Default: 1000
        causal_capacity = config.get('causal_capacity', 100) # Default: 100
        prediction_model_hidden_size = config.get('prediction_model_hidden_size', 16) # Default: 16
        layer_growth_threshold = config.get('layer_growth_threshold', 0.9) # Default: 0.9
        novelty_threshold = config.get('novelty_threshold', 0.5) # Default: 0.5
        novelty_tolerance = config.get('novelty_tolerance', 0.5) # Default: 0.5
        reward_history_window = config.get('reward_history_window', 10) # Default: 10


        # **Callbacks aus Konfiguration (falls vorhanden)**
        self.pre_decide_hook = config.get('pre_decide_hook') # Callback vor decide()
        self.post_learn_hook = config.get('post_learn_hook') # Callback nach learn()
        self.rule_engine_hook = config.get('rule_engine_hook') # Callback für Rule Engine


        # **Modell-Definition (bleibt weitgehend gleich, nutzt konfigurierte Parameter)**
        self.model = nn.Sequential(
            nn.Linear(input_size + emotion_dim, 32), # **Emotion Dim HINZUGEFÜGT als Input-Dimension**
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        ).float() # Stelle sicher, dass das Modell Float verwendet
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # **Komponenten initialisieren (nutzt konfigurierte Parameter)**
        self.memory = Memory(capacity=memory_capacity, causal_capacity=causal_capacity) # Konfigurierbare Memory-Kapazität
        self.state_similarity_threshold = state_similarity_threshold # Schwellwert speichern

        self.hierarchical_attention = DynamicHierarchicalSelfAttention(embed_dim, num_heads, layer_growth_threshold=layer_growth_threshold).float() # Nutze DynamicHierarchicalSelfAttention, Float-Typ, konfigurierbarer Schwellwert
        self.input_size = input_size # Speichere input_size für create_abstract_state
        self.novelty_threshold = novelty_threshold # Schwellwert für Neuigkeitsbonus
        self.novelty_tolerance = novelty_tolerance # Toleranz für Ähnlichkeit im Neuigkeitsbonus
        self.embed_dim = embed_dim # Wichtig: Speichere embed_dim hier

        # **Neu: Weltmodell (angepasst für state_size & action_size & konfigurierbare Parameter)**
        self.world_model = WorldModel(state_size=input_size, action_size=1, prediction_model_hidden_size=prediction_model_hidden_size, learning_rate=learning_rate) # state_size und action_size übergeben, action_size korrigiert zu 1, konfigurierbare Parameter

        # **Neu: Ziele (Objekte) mit Subzielen**
        self.goals = {
            "explore": Goal("Erkunde die Welt", priority="hoch", subgoals=["Finde einen neuen Zustand", "Besuche unvorhersehbaren Zustand", "Besuche Zustand mit positiver Valenz"]), # Ziel mit Subzielen
            "reduce_frustration": Goal("Reduziere Frustration", priority="mittel") # Ziel ohne explizite Subziele (fürs Erste)
        }
        self.current_subgoal_index = 0 # Index des aktuellen Subziels (für "Erkunde die Welt")

        # **Neu: Emotionsmodell instanziieren**
        self.emotion_model = EmotionModel(emotion_dim=emotion_dim) # Emotionsmodell initialisieren
        self.emotion_dim = emotion_dim # Speichere emotion_dim

        # **NEU: Selbstmodell instanziieren**
        self.self_model = SelfModel()
        self.current_goal_key = self.self_model.current_goal_key # Ziel aus Selbstmodell
        self.current_goal = self.goals[self.current_goal_key] # Ziel-Objekt setzen
        self.current_subgoal = self.get_current_subgoal() # Subziel setzen
        self.last_action = 0 # Letzte Aktion initialisieren
        self.last_reward = 0 # Letzte Belohnung initialisieren

        # **Direkter Zugriff auf Komponenten ermöglichen (wichtig für Erweiterbarkeit!)**
        self.config = config # Konfiguration speichern für evtl. späteren Zugriff
        # Komponenten als öffentliche Attribute zugänglich machen
        self.memory_component = self.memory
        self.world_model_component = self.world_model
        self.rule_engine_component = workspace.get_rule_engine() # Zugriff auf Rule Engine über Workspace (wie zuvor)
        self.hierarchical_attention_component = self.hierarchical_attention
        self.emotion_model_component = self.emotion_model
        self.self_model_component = self.self_model


    def get_current_subgoal(self):
        """Ruft das aktuelle Subziel basierend auf dem Index ab."""
        if self.current_goal_key == "explore" and self.current_goal.subgoals: # Nur für "Erkunde die Welt" mit Subzielen
            return self.current_goal.subgoals[self.current_subgoal_index]
        return None # Kein Subziel für andere Ziele oder wenn keine Subziele definiert


    def decide(self, state):
        """Agent entscheidet basierend auf aktuellem Zustand, Emotionen & Belohnung in kausaler Erinnerung."""
        # **Callback vor der Entscheidung (pre_decide_hook)**
        if self.pre_decide_hook:
            self.pre_decide_hook(self, state) # Agent-Instanz und Zustand übergeben

        emotion_state_tensor = self.emotion_model.get_emotions() # Aktuellen Emotionszustand abrufen (als Tensor)
        valence = emotion_state_tensor[0].item() # Valenz extrahieren (Dimension 0)
        arousal = emotion_state_tensor[1].item() # Arousal extrahieren (Dimension 1)
        dominance = emotion_state_tensor[2].item() # Dominanz extrahieren (Dimension 2)
        print(f"😊 Emotionen (Valenz: {valence:.2f}, Arousal: {arousal:.2f}, Dominanz: {dominance:.2f})") # Ausgabe der Emotionen


        causal_memories = [self.memory.recall_causal() for _ in range(5)] # Mehrere Erinnerungen abrufen
        relevant_memories = [
            mem for mem in causal_memories
            if mem and self.state_similarity(mem[0], state) < self.state_similarity_threshold
        ]
        action_rewards = {} # Dictionary für summierte Belohnungen
        if relevant_memories:
            for mem in relevant_memories:
                action, reward = mem[1], mem[3] # Belohnung aus Erinnerung holen
                action_rewards[action] = action_rewards.get(action, 0) + reward # Belohnung summieren

            if action_rewards:
                best_action = max(action_rewards, key=action_rewards.get) # Aktion mit höchster Belohnung
                print(f"💡 Kausale Erinnerung (belohnungsbasiert) beeinflusst Entscheidung! Bevorzugte Aktion: {best_action}")
                # **Epsilon-Greedy: Zufällige Aktion mit Wahrscheinlichkeit epsilon**
                epsilon = 0.1 # Exploration Rate
                if random.random() < epsilon:
                    random_action = random.choice(range(self.model[2].out_features)) # Zufällige Aktion aus Aktionsraum
                    print(f"ε-greedy Exploration: Zufällige Aktion gewählt: {random_action}")
                    return random_action
                return best_action

        # **Erweiterung: Emotionen als INPUT für das neuronale Netz**
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # **Neu: Emotionszustand zum Input-Zustand hinzufügen**
        emotion_state_tensor = self.emotion_model.get_emotions() # Aktuelle Emotionen abrufen
        model_input = torch.cat([state_tensor, emotion_state_tensor], dim=-1) # Emotionen konkatenieren
        action_probs = self.model(model_input) # **Input enthält nun auch Emotionen**
        nn_action = torch.argmax(action_probs).item()
        print(f"🧠 Neuronales Netz entscheidet (mit Emotionen als Input): Aktion: {nn_action}")
        # **Epsilon-Greedy: Zufällige Aktion mit Wahrscheinlichkeit epsilon**
        epsilon = 0.1 # Exploration Rate
        if random.random() < epsilon:
            random_action = random.choice(range(self.model[2].out_features)) # Zufällige Aktion aus Aktionsraum
            print(f"ε-greedy Exploration: Zufällige Aktion gewählt: {random_action}")
            return random_action
        return nn_action


    def reflect(self, abstract_state): # Beispiel für Reflektionsfunktion, die abstrakten Zustand verarbeitet
        """Reflektiert über den abstrakten Zustand mit hierarchischer Aufmerksamkeit."""
        abstract_state = abstract_state.float() # Stelle sicher, dass abstract_state Float ist
        processed_state = self.hierarchical_attention(abstract_state)
        return processed_state

    def learn(self, state, action, reward, next_state):
        #super().learn(state, action, reward) # Basis-Lernprozess beibehalten

        target = torch.zeros(2)
        target[action] = reward
        state_tensor = torch.tensor(state, dtype=torch.float32)

        self.optimizer.zero_grad()
        # **Lernprozess angepasst an neuen Input des Modells (mit Emotionen)**
        emotion_state_tensor = self.emotion_model.get_emotions() # Aktuelle Emotionen abrufen
        model_input = torch.cat([state_tensor, emotion_state_tensor], dim=-1) # Input für Modell erstellen (Zustand + Emotionen)
        output = self.model(model_input) # **Modell-Input enthält nun Emotionen**
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()

        # Kausale Erinnerung speichern (mit next_state und reward)
        self.memory.store_causal(state, action, next_state, reward)

        # Last Action/Reward speichern
        self.last_action = action
        self.last_reward = reward

        # **Neu: Weltmodell aktualisieren (inkl. Training des Prediction Models)**
        next_state_wm = np.random.rand(self.input_size) # Vereinfachung: Nächster Zustand = zufällige Wahrnehmung (für Demo)
        self.world_model.add_transition(state, action, next_state_wm, reward) # Weltmodell aktualisieren, inkl. Training

        # Zustandseigenschaften berechnen (für Emotionsmodell-Update)
        state_properties = self.world_model.calculate_state_properties(state, action) # Zustandseigenschaften berechnen (ACTION hinzugefügt!)
        predictability = state_properties.get('predictability', 0.0) # Predictability extrahieren

        # Reflektion nach dem Lernen, um intrinsische Belohnung zu generieren
        abstract_state = self.create_abstract_state() # Funktion zur Erzeugung eines abstrakten Zustands
        reflected_state = self.reflect(abstract_state) # Reflektiere über den abstrakten Zustand

        intrinsic_reward = self.calculate_intrinsic_reward(abstract_state, reflected_state) # Berechne intrinsische Belohnung
        total_reward = reward + intrinsic_reward # Kombiniere extrinsische und intrinsische Belohnung

        # Erweitere Lernfunktion, um intrinsische Belohnung zu nutzen (Beispielhaft)
        target_intrinsic = torch.zeros(2)
        target_intrinsic[action] = total_reward # Nutze Gesamtbelohnung
        state_tensor_intrinsic = torch.tensor(state, dtype=torch.float32)
        # **Lernprozess für intrinsische Belohnung angepasst (mit Emotionen im Input)**
        emotion_state_tensor_intrinsic = self.emotion_model.get_emotions() # Aktuelle Emotionen abrufen
        model_input_intrinsic = torch.cat([state_tensor_intrinsic, emotion_state_tensor_intrinsic], dim=-1) # Input erstellen
        output_intrinsic = self.model(model_input_intrinsic) # **Modell-Input enthält Emotionen**
        loss_intrinsic = self.loss_fn(output_intrinsic, target_intrinsic)
        loss_intrinsic.backward()
        self.optimizer.step()

        # Drives aktualisieren (rudimentär)
        valence_emotion = self.emotion_model.get_emotions()[0].item() # Valenz für Drive-Modulation extrahieren
        self.update_drives(intrinsic_reward, valence_emotion) # Valenz übergeben für Drive-Modulation

        # **Neu: Emotionsmodell aktualisieren (mit Predictability)**
        self.emotion_model.update_emotions(reward, predictability) # Emotionen basierend auf extrinsischer Belohnung & Predictability aktualisieren

        # **NEU: Selbstmodell aktualisieren (nach allen anderen Updates)**
        self.update_self_model()

        # **Callback nach dem Lernen (post_learn_hook)**
        if self.post_learn_hook:
            self.post_learn_hook(self, state, action, reward, next_state) # Agent-Instanz und Lernparameter übergeben


    def update_self_model(self):
        """Aktualisiert das Selbstmodell mit aktuellen Agenten-Zuständen."""
        # Drives direkt aus dem Selbstmodell holen
        self.self_model.update_drives(
            curiosity=self.self_model.drives["curiosity"],
            understanding=self.self_model.drives["understanding"],
            frustration=self.self_model.drives["frustration"]
        )
        self.self_model.update_emotions(self.emotion_model.get_emotions())
        self.self_model.update_goal(self.current_goal_key, self.current_subgoal) # Ziel & Subziel auch updaten
        # **Zukunft:** Hier könnte man komplexere Selbst-Reflexions-Prozesse einbauen, um das Selbstmodell zu aktualisieren.
        # Z.B. basierend auf Erfolg/Misserfolg bei Zielen, Veränderungen in Drives/Emotionen, etc.
        # Fürs Erste: Einfache Übernahme der aktuellen Werte.


    def create_abstract_state(self):
        """
        Erzeugt einen abstrakten Zustand aus dem internen Gedächtnis, aktuellen Wahrnehmungen, Drives, Weltmodell (rudimentär) & Ziel & Subziel & **Emotionen & Selbstmodell**.
        """
        # Beispiel: Kombiniere zufällige Erinnerungen, aktuellen Zustand und Drives zu einem abstrakten Zustand
        recalled_memory = self.memory.recall()
        current_perception = torch.tensor(np.random.rand(self.input_size), dtype=torch.float32).unsqueeze(0) # Aktuelle Wahrnehmung simulieren, Batch-Dimension hinzufügen
        #drive_state = torch.tensor([self.curiosity_drive, self.understanding_drive, self.frustration_drive]).unsqueeze(0) # Drive-Zustände als Tensor, Batch-Dimension hinzufügen
        world_model_state = torch.tensor([self.last_action, self.last_reward]).unsqueeze(0) # Letzte Aktion und Belohnung (rudimentäres Weltmodell)
        goal_state = torch.tensor([1.0 if self.current_goal_key == "explore" else 0.0, 1.0 if self.current_goal_key == "reduce_frustration" else 0.0]).unsqueeze(0) # Ziel als One-Hot über Key
        subgoal_state = torch.tensor([1.0 if self.current_subgoal == "Finde einen neuen Zustand" else 0.0, 1.0 if self.current_subgoal == "Besuche unvorhersehbaren Zustand" else 0.0, 1.0 if self.current_subgoal == "Besuche Zustand mit positiver Valenz" else 0.0]).unsqueeze(0) # Subziel als One-Hot (rudimentär)

        # **Neu: Ähnlichen Zustand aus Weltmodell abrufen (mit Eigenschaften)**
        similar_world_state, state_properties = self.world_model.recall_similar_state(current_perception.squeeze(0).numpy(), tolerance=0.8) # Zustandsteil der Erinnerung nutzen (vereinfacht), Batch-Dimension hinzufügen
        if similar_world_state is None:
            similar_world_state = torch.zeros_like(current_perception) # Wenn kein ähnlicher Zustand, Null-Tensor
            state_properties = {} # Leere Eigenschaften, wenn kein Zustand gefunden

        # **Neu: Zustandseigenschaften in abstrakten Zustand integrieren (rudimentär)**
        novelty_prop = torch.tensor([state_properties.get('novelty_score', 0.0)]).unsqueeze(0) # Neuigkeitswert extrahieren, default 0.0
        frustration_prop = torch.tensor([state_properties.get('frustration_level', 0.0)]).unsqueeze(0) # Frustrationslevel extrahieren, default 0.0
        predictability_prop = torch.tensor([state_properties.get('predictability', 0.0)]).unsqueeze(0) # Predictability extrahieren, default 0.0
        valence_prop = torch.tensor([state_properties.get('valence', 0.0)]).unsqueeze(0) # Valence extrahieren, default 0.0

        # **Neu: Emotionszustand hinzufügen (jetzt Valenz & Arousal & Dominanz)**
        emotion_state = self.emotion_model.get_emotions().unsqueeze(0) # Emotionszustand abrufen und Batch-Dimension hinzufügen

        # **NEU: Drives aus dem Selbstmodell holen**
        drive_state = torch.tensor(list(self.self_model.drives.values())).unsqueeze(0) # Drives aus Selbstmodell, Batch-Dimension

        # **NEU: Selbstbewusstseins-Vektor aus Selbstmodell holen**
        self_awareness_vector = self.self_model.self_awareness_vector.unsqueeze(0) # Selbstbewusstseins-Vektor, Batch-Dimension


        if recalled_memory:
            memory_tensor = torch.tensor(recalled_memory[0], dtype=torch.float32).unsqueeze(0) # Nur den Zustandsteil der Erinnerung nutzen (vereinfacht), Batch-Dimension hinzufügen
            abstract_state_list = [current_perception, memory_tensor, drive_state, world_model_state, goal_state, subgoal_state, similar_world_state, novelty_prop, frustration_prop, predictability_prop, valence_prop, emotion_state, self_awareness_vector] # Emotionszustand + Selbstmodell hinzugefügt
        else:
            abstract_state_list = [current_perception, drive_state, world_model_state, goal_state, subgoal_state, similar_world_state, novelty_prop, frustration_prop, predictability_prop, valence_prop, emotion_state, self_awareness_vector] # Emotionszustand + Selbstmodell hinzugefügt

        abstract_state = torch.cat(abstract_state_list, dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren
        abstract_state = abstract_state.float() # Stelle sicher, dass abstract_state Float ist

        # Padding, um sicherzustellen, dass die Dimension immer embed_dim ist
        current_dim = abstract_state.shape[1]
        if current_dim < self.embed_dim:
            padding_size = self.embed_dim - current_dim
            padding = torch.zeros(abstract_state.shape[0], padding_size) # Batch-Dimension beibehalten
            abstract_state = torch.cat([abstract_state, padding], dim=-1) # Korrektur: dim=-1 für horizontales Konkatenieren
        elif current_dim > self.embed_dim:
            # Fallback: Dimension reduzieren, falls sie größer ist (sollte nicht passieren, aber sicherheitshalber)
            abstract_state = abstract_state[:, :self.embed_dim]

        return abstract_state


    def calculate_intrinsic_reward(self, abstract_state, reflected_state):
        """
        Berechnet die intrinsische Belohnung basierend auf Informationsgewinn, Neuigkeit und Kohärenz, beeinflusst von Drives.
        """
        information_gain = self.calculate_information_gain(abstract_state, reflected_state)
        novelty_bonus = self.calculate_novelty_bonus(abstract_state)
        coherence_reward = self.calculate_coherence_reward(reflected_state)

        # Drives beeinflussen Belohnungsgewichtungen (dynamischer)
        # **NEU: Drives aus Selbstmodell holen**
        curiosity_drive = self.self_model.drives["curiosity"]
        understanding_drive = self.self_model.drives["understanding"]

        novelty_weight = 0.2 + curiosity_drive * 0.8 # Neugier erhöht Gewichtung von Neuigkeit (stärker)
        coherence_weight = 0.1 + understanding_drive * 0.7 # Verständnis-Drive erhöht Gewichtung von Kohärenz (stärker)
        information_gain_weight = 1.0 - novelty_weight - coherence_weight # Rest für Informationsgewinn

        # Kombiniere die Belohnungskomponenten (dynamische Gewichtungen basierend auf Drives)
        intrinsic_reward = information_gain_weight * information_gain + novelty_weight * novelty_bonus + coherence_weight * coherence_reward
        return intrinsic_reward

    def calculate_information_gain(self, state_before, state_after):
        """Misst den Informationsgewinn (Beispiel: Distanzänderung)."""
        return torch.norm(state_after - state_before).item() # Beispiel: Euklidische Distanz

    def calculate_novelty_bonus(self, state):
        """Berechnet den Neuigkeits-Bonus (Beispiel: Seltenheit im Gedächtnis)."""
        # Vereinfachtes Beispiel: Zähle Vorkommen ähnlicher Zustände im Gedächtnis
        novelty_count = 0
        for memory in self.memory.memory:
            if memory and torch.allclose(torch.tensor(memory[0]), state.squeeze(0), atol=self.novelty_tolerance): # Zustandsteil der Erinnerung vergleichen (Toleranz), Batch-Dimension entfernen
                novelty_count += 1

        # Neu: Bonus nur, wenn Zustand "neu genug" ist
        if novelty_count < self.novelty_threshold * self.memory.capacity: # Beispiel: Bonus, wenn Zustand seltener als Schwellwert * Kapazität
            return 1.0 / (novelty_count + 1) # Je seltener, desto höher der Bonus
        else:
            return 0.0 # Kein Bonus für "bekannte" Zustände


    def calculate_coherence_reward(self, reflected_state):
        """Berechnet die Kohärenz-Belohnung (Beispiel: Innere Konsistenz)."""
        # Sehr vereinfachtes Beispiel:  Annahme: Reflektierter Zustand sollte "einfacher" oder "kompakter" sein
        # -> Messung der Norm des reflektierten Zustands als Indikator für "Einfachheit/Kohärenz"
        return -torch.norm(reflected_state).item() # Negative Norm -> Minimierung der Norm wird belohnt (vereinfacht!)

    def update_drives(self, intrinsic_reward, valence_emotion): # Valenz als Parameter
        """Aktualisiert die Drives basierend auf intrinsischer Belohnung, Emotionen, Zufall und Frustration."""
        # Rudimentäre Drive-Aktualisierung (Beispielhaft, inkl. Frustration)
        # **NEU: Drives werden DIREKT im Selbstmodell aktualisiert (und von dort in create_abstract_state geholt)**
        current_curiosity = self.self_model.drives["curiosity"]
        current_understanding = self.self_model.drives["understanding"]
        current_frustration = self.self_model.drives["frustration"]

        updated_curiosity = current_curiosity + intrinsic_reward * 0.02 + random.uniform(-0.01, 0.02) # Durch Belohnung verstärken, zufällige Fluktuation
        updated_understanding = current_understanding + intrinsic_reward * 0.01 + random.uniform(-0.005, 0.005) # Durch Belohnung verstärken, zufällige Fluktuation
        updated_frustration = current_frustration

        # **Neu: Emotionale Modulation der Drives (rudimentär)**
        # Negative Valenz -> Frustration verstärken, Neugier & Verständnis dämpfen (Beispielhaft)
        if valence_emotion < -0.2:
            updated_frustration += 0.03 # Frustration stärker erhöhen bei negativer Valenz
            updated_curiosity -= 0.02    # Neugier leicht dämpfen
            updated_understanding -= 0.01 # Verständnis leicht dämpfen
            print(f"   -> Emotionale Drive-Modulation (negative Valenz): Frustration++, Neugier--, Verständnis--")
        elif valence_emotion > 0.3: # Positive Valenz -> Neugier & Verständnis verstärken (Beispielhaft)
            updated_curiosity += 0.03     # Neugier leicht verstärken
            updated_understanding += 0.02  # Verständnis leicht verstärken
            updated_frustration -= 0.01    # Frustration leicht dämpfen
            print(f"   -> Emotionale Drive-Modulation (positive Valenz): Neugier++, Verständnis++, Frustration--")


        # Frustration anpassen: Steigt bei niedriger Neuigkeit/Kohärenz, sinkt leicht zufällig (wie zuvor)
        novelty_bonus = self.calculate_novelty_bonus(self.create_abstract_state()) # Neuigkeitsbonus aktuell berechnen (etwas ineffizient, aber für Demo ok)
        coherence_reward = self.calculate_coherence_reward(self.reflect(self.create_abstract_state())) # Kohärenz auch
        if novelty_bonus < 0.1 or coherence_reward < -0.8: # Beispielschwellwerte für Frustration
            updated_frustration += 0.05 # Frustration steigt
        else:
            updated_frustration -= 0.01 # Frustration sinkt leicht (Baseline-Abbau)

        updated_frustration = np.clip(updated_frustration, 0.0, 1.0) # Frustration begrenzen
        updated_curiosity = np.clip(updated_curiosity, 0.0, 1.0) # Drives begrenzen
        updated_understanding = np.clip(updated_understanding, 0.0, 1.0)

        # **NEU: Drives im Selbstmodell aktualisieren**
        self.self_model.update_drives(curiosity=updated_curiosity, understanding=updated_understanding, frustration=updated_frustration)

        # Beispiel: Ausgabe der Drive-Zustände (für Beobachtung)
        print(f"INFO: Drives aktualisiert - Neugier: {updated_curiosity:.3f}, Verständnis: {updated_understanding:.3f}, Frustration: {updated_frustration:.3f}")
        print(f"      Intrinsische Belohnung: {intrinsic_reward:.3f}, Valenz: {valence_emotion:.3f}")

        # **Neu: Ziel- & Subziel-Überprüfung & -Anpassung (rudimentär)**
        self.check_and_update_goal()


    def check_and_update_goal(self):
        """Überprüft das aktuelle Ziel und ändert es ggf. basierend auf Agenten-Zustand (rudimentär), inkl. Subziel-Management."""
        if self.current_goal_key == "reduce_frustration": # Ziel-Key nutzen
            if self.self_model.drives["frustration"] < 0.3: # Beispielschwellwert für Frustration reduziert
                print(f"✅ Ziel erreicht: Frustration reduziert ({self.goals['reduce_frustration'].name}). Neues Ziel: Erkunde die Welt.") # Zielnamen aus Objekt
                self.current_goal_key = "explore" # Ziel-Key wechseln
                self.current_goal = self.goals[self.current_goal_key] # Ziel-Objekt aktualisieren
                self.goals["reduce_frustration"].status = "erreicht" # Status des alten Ziels ändern
                self.goals["explore"].status = "aktiv" # Status des neuen Ziels ändern
                self.current_subgoal_index = 0 # Subziel-Index zurücksetzen für neues Hauptziel
                self.current_subgoal = self.get_current_subgoal() # Subziel aktualisieren
                # **NEU: Selbstmodell aktualisieren mit neuem Ziel**
                self.self_model.update_goal(self.current_goal_key, self.current_subgoal)


        elif self.current_goal_key == "explore": # Ziel-Key nutzen
            if self.self_model.drives["frustration"] > 0.7: # Wenn Frustration beim Erkunden zu hoch wird...
                print(f"⚠️ Frustration beim Erkunden zu hoch ({self.goals['explore'].name}). Zielwechsel: Reduziere Frustration.") # Zielnamen aus Objekt
                self.current_goal_key = "reduce_frustration" # Ziel-Key wechseln
                self.current_goal = self.goals[self.current_goal_key] # Ziel-Objekt aktualisieren
                self.goals["explore"].status = "inaktiv" # Status des alten Ziels ändern
                self.goals["reduce_frustration"].status = "aktiv" # Status des neuen Ziels ändern
                self.current_subgoal_index = 0 # Subziel-Index zurücksetzen für neues Hauptziel
                self.current_subgoal = self.get_current_subgoal() # Subziel aktualisieren
                # **NEU: Selbstmodell aktualisieren mit neuem Ziel**
                self.self_model.update_goal(self.current_goal_key, self.current_subgoal)

            else: # **Neu: Subziel-Management innerhalb von "Erkunde die Welt"**
                if self.current_subgoal == "Finde einen neuen Zustand":
                    novelty_bonus = self.calculate_novelty_bonus(self.create_abstract_state())
                    if novelty_bonus < 0.1: # Wenn wenig Neues gefunden... (Beispielschwellwert)
                        print(f"Subziel '{self.current_subgoal}' (Ziel: {self.current_goal.name}) scheint schwierig. Wechsle Subziel.")
                        self.current_subgoal_index = (self.current_subgoal_index + 1) % len(self.current_goal.subgoals) # Zum nächsten Subziel wechseln (zyklisch)
                        self.current_subgoal = self.get_current_subgoal() # Subziel aktualisieren
                        print(f"Neues Subziel: '{self.current_subgoal}' (Ziel: {self.current_goal.name})")
                        # **NEU: Selbstmodell aktualisieren mit neuem Subziel**
                        self.self_model.update_goal(self.current_goal_key, self.current_subgoal)


    def state_similarity(self, state1, state2):
        """Berechnet die euklidische Distanz zwischen zwei Zuständen."""
        return np.linalg.norm(np.array(state1) - np.array(state2))


# 🌐 Globale Regel-Engine (zukünftig für inneren Dialog)
class RuleEngine:
    def __init__(self):
        # Hier werden später Regeln für den inneren Dialog gespeichert
        pass

    def apply_rules(self, agent_state, global_workspace, agent): # Zugriff auf Agent benötigt
        """
        Wendet Regeln an, um den Informationsfluss und das Verhalten des Agenten zu steuern.
        Erweitert um zielbezogene Regeln, Weltmodell-Nutzung & **Selbst-Reflexions-Regeln** & **Emotions-basierte Regeln & **Selbstmodell-basierte Regeln**.
        """
        novelty_bonus = agent_state.get('novelty_bonus', 0.0)
        coherence_reward = agent_state.get('coherence_reward', 0.0)
        frustration_drive = agent_state.get('frustration_drive', 0.0)
        complexity_metric = agent_state.get('complexity_metric', 0.0)
        predictability = agent_state.get('predictability', 0.0) # Neu: Predictability
        valence = agent_state.get('valence', 0.0) # Neu: Valence
        current_goal_key = agent_state.get('current_goal_key', "Kein Ziel") # Aktueller Ziel-Key abrufen
        current_subgoal = agent_state.get('current_subgoal', "Kein Subziel") # Aktuelles Subziel abrufen
        emotion_state = agent_state.get('emotion_state', torch.zeros(agent.emotion_dim).tolist()) # Neu: Emotionszustand abrufen
        valence_emotion = emotion_state[0] # Annahme: Erste Dimension ist Valenz
        arousal_emotion = emotion_state[1] # Annahme: Zweite Dimension ist Arousal
        dominance_emotion = emotion_state[2] # Annahme: Dritte Dimension ist Dominanz
        self_awareness_vector = agent_state.get('self_awareness_vector', torch.zeros(32).tolist()) # **NEU**: Selbstbewusstseins-Vektor

        # **Callback für Rule Engine (rule_engine_hook) - VOR Regeln**
        if agent.rule_engine_hook:
            agent.rule_engine_hook(agent, agent_state, "pre_rules") # Agent-Instanz, Agent-Zustand, Phase übergeben


        # **Ziel-bezogene Regeln (rudimentär) - Ziel-KEY nutzen**
        if current_goal_key == "explore": # Ziel-Key nutzen
            if novelty_bonus > 0.8: # Bei hohem Neuigkeitsbonus...
                print(f"💬 Regel-Engine (Ziel: Erkunden, Subziel: {current_subgoal}): Neuigkeitsbonus hoch ({novelty_bonus:.2f}) - 'Frage' Gedächtnis nach ähnlichen Erfahrungen...")
                # Neu: Aktion: Gezielter Memory Recall (ähnliche Erfahrungen)
                similar_memory = agent.memory.recall_similar(agent.create_abstract_state()) # Ähnliche Erinnerung abrufen
                if similar_memory:
                    global_workspace.share(f"Regel-Engine-Hinweis (Ziel: Erkunden, Subziel: {current_subgoal}): Ähnliche Erinnerung abgerufen: {similar_memory}")

            if complexity_metric > 0.95: # Bei hoher Komplexität... (immer noch relevant)
                print(f"💬 Regel-Engine (Ziel: Erkunden, Subziel: {current_subgoal}): Hohe Komplexität ({complexity_metric:.2f}) - Dynamische Ebene hinzufügen (Beispiel)")
                agent.hierarchical_attention.add_layer() # Dynamisch Ebene hinzufügen (als Reaktion auf Komplexität)

        elif current_goal_key == "reduce_frustration": # Ziel-Key nutzen
            if frustration_drive > 0.6: # Bei hoher Frustration...
                print(f"💬 Regel-Engine (Ziel: Frustration reduzieren): Frustration hoch ({frustration_drive:.2f}) - 'Suche' nach einfacheren Mustern, erhöhe Toleranz...")
                # Neu: Aktion: Erhöhe Neuigkeitstoleranz (für explorativeres Verhalten -> einfachere Muster "akzeptieren")
                agent.novelty_tolerance += 0.2 # Erhöhe Toleranz stärker, um "bekanntere" Zustände als neu zu betrachten
                agent.novelty_tolerance = np.clip(agent.novelty_tolerance, 0.0, 1.0) # Begrenzen
                print(f"   -> Neuigkeitstoleranz erhöht auf {agent.novelty_tolerance:.2f} (Fokus auf Bekanntes)")

            if coherence_reward < -0.7: # Bei SEHR niedriger Kohärenz (evtl. in frustrierender Situation)...
                print(f"💬 Regel-Engine (Ziel: Frustration reduzieren): Kohärenz SEHR niedrig ({coherence_reward:.2f}) - Setze Neuigkeitstoleranz zurück (Fokus auf Bekanntes)...")
                agent.novelty_tolerance = 0.1 # Setze Toleranz niedriger, um Fokus stärker auf Bekanntes zu lenken (Verankerung in bekanntem Terrain)
                print(f"   -> Neuigkeitstoleranz reduziert auf {agent.novelty_tolerance:.2f} (Fokus auf Bekanntes)")

        # Weltmodell-bezogene Regeln (Beispielhaft)
        if novelty_bonus < 0.2 and current_goal_key == "explore": # Wenn wenig Neues entdeckt wird, aber Ziel "Erkunden" ist... (Ziel-Key nutzen)
            print(f"💬 Regel-Engine (Ziel: Erkunden, Subziel: {current_subgoal}): Wenig Neues ({novelty_bonus:.2f}) -  'Überprüfe' Weltmodell nach unerkundeten Pfaden...")
            # **Neu: Aktion: Nutze Weltmodell, um explorativer zu werden (rudimentär)**
            similar_state_from_wm = agent.world_model.recall_similar_state(agent.create_abstract_state().squeeze(0).numpy(), tolerance=0.8)[0] # Ähnlichen Zustand abrufen, nur Zustandsteil [0]
            if similar_state_from_wm is not None:
                transitions = agent.world_model.get_transitions(similar_state_from_wm.squeeze(0).numpy()) # Transitionen für ähnlichen Zustand abrufen
                if transitions:
                    print(f"   -> Weltmodell-Hinweis: Transitionen gefunden für ähnlichen Zustand: {transitions.keys()}")
                    # **Visionär:** Hier könnte Agent Aktionen wählen, die zu neuen/unbekannten Zuständen führen (basierend auf Weltmodell)
                    # **Aktuell (rudimentär):**  Erhöhe Neugier-Drive leicht, um Exploration zu fördern
                    agent.self_model.update_drives(curiosity=agent.self_model.drives["curiosity"] + 0.05) # Neugier im Selbstmodell aktualisieren
                    print(f"      -> Neugier-Drive leicht erhöht auf {agent.self_model.drives['curiosity']:.2f}")

        # **Neu: Selbst-Reflexions-Regeln (rudimentär)**
        if frustration_drive > 0.8: # Beispiel-Trigger für Selbst-Reflexion: Hohe Frustration
            print(f"🤔 Selbst-Reflexion (Frustration hoch): 'Warum bin ich so frustriert?  Was kann ich ändern?' (Ziel: {current_goal_key}, Subziel: {current_subgoal})")
            global_workspace.share(f"Selbst-Reflexion: Agent frustriert. Ziel: {current_goal_key}, Subziel: {current_subgoal}, Frustration: {frustration_drive:.2f}") # Workspace-Mitteilung

            if predictability < 0.2: # Wenn Zustand auch noch unvorhersehbar ist...
                print(f"   -> Zustand unvorhersehbar ({predictability:.2f}). 'Vielleicht sollte ich mich auf Vorhersagbareres konzentrieren?'")
                global_workspace.share(f"Selbst-Reflexion: Zustand unvorhersehbar. Predictability: {predictability:.2f}")

        elif coherence_reward < -0.9: # Beispiel-Trigger: Sehr niedrige Kohärenz
            print(f"🤔 Selbst-Reflexion (Kohärenz niedrig): 'Meine Gedanken sind unzusammenhängend. Muss Fokus ändern?' (Ziel: {current_goal_key}, Subziel: {current_subgoal}, Kohärenz: {coherence_reward:.2f})")
            global_workspace.share(f"Selbst-Reflexion: Kohärenz niedrig. Ziel: {current_goal_key}, Subziel: {current_subgoal}, Kohärenz: {coherence_reward:.2f}")
            # **Rudimentäre Anpassung: Subziel wechseln (Beispiel)**
            if current_goal_key == "explore": # Nur für Explore-Ziel (Beispiel)
                print(f"   -> Subziel-Wechsel initiiert (Kohärenz niedrig).")
                agent.current_subgoal_index = (agent.current_subgoal_index + 1) % len(agent.current_goal.subgoals) # Zum nächsten Subziel wechseln (zyklisch)
                agent.current_subgoal = agent.get_current_subgoal() # Subziel aktualisieren
                print(f"   -> Neues Subziel: '{current_subgoal}' (Ziel: {agent.goals['explore'].name})")
                # **NEU: Selbstmodell aktualisieren mit neuem Subziel**
                agent.self_model.update_goal(agent.current_goal_key, agent.current_subgoal)


        # **Neu: Emotions-basierte Regeln (rudimentär) - ERWEITERT mit AROUSAL & DOMINANZ**
        if valence_emotion < -0.5 and current_goal_key == "explore": # Negative Valenz beim Erkunden...
            print(f"😞 Regel-Engine (Ziel: Erkunden, Emotion: Negative Valenz ({valence_emotion:.2f}), Arousal: {arousal_emotion:.2f}, Dominanz: {dominance_emotion:.2f}): 'Erkundung ist unangenehm. Fokus ändern?'")
            global_workspace.share(f"Regel-Engine-Hinweis: Negative Valenz beim Erkunden. Valenz: {valence_emotion:.2f}, Arousal: {arousal_emotion:.2f}, Dominanz: {dominance_emotion:.2f}")
            # **Rudimentäre Reaktion: Zielwechsel zu Frustrationsreduktion (Beispiel)**
            print(f"   -> Zielwechsel zu 'Reduziere Frustration' vorgeschlagen (basierend auf negativer Valenz).")
            agent.current_goal_key = "reduce_frustration"
            agent.current_goal = agent.goals[agent.current_goal_key]
            agent.goals["explore"].status = "inaktiv"
            agent.goals["reduce_frustration"].status = "aktiv"
            agent.current_subgoal_index = 0
            agent.current_subgoal = agent.get_current_subgoal()
            # **NEU: Selbstmodell aktualisieren mit neuem Ziel**
            agent.self_model.update_goal(agent.current_goal_key, agent.current_subgoal)


        if arousal_emotion > 0.8 and current_goal_key == "explore": # Hohes Arousal beim Erkunden... (z.B. unerwarteter Zustand)
            print(f"😮 Regel-Engine (Ziel: Erkunden, Emotion: Hohes Arousal ({arousal_emotion:.2f}), Valenz: {valence_emotion:.2f}, Dominanz: {dominance_emotion:.2f}): 'Unerwartetes/Aufregendes gefunden! Fokus verstärken?'")
            global_workspace.share(f"Regel-Engine-Hinweis: Hohes Arousal beim Erkunden. Arousal: {arousal_emotion:.2f}, Valenz: {valence_emotion:.2f}, Dominanz: {dominance_emotion:.2f}")
            # **Rudimentäre Reaktion: Neugier-Drive verstärken, Subziel anpassen (Beispiel)**
            print(f"   -> Neugier-Drive leicht erhöht, Subziel ggf. anpassen (Fokus auf Unerwartetes?).")
            agent.self_model.update_drives(curiosity=agent.self_model.drives["curiosity"] + 0.1) # Neugier im Selbstmodell aktualisieren
            # **Zukünftig:** Subziel könnte auf "Besuche unvorhersehbaren Zustand" wechseln oder verstärkt werden.

        # **Neu: Dominanz-basierte Regel (Beispiel)**
        if dominance_emotion > 0.6 and current_goal_key == "reduce_frustration": # Gefühl von Dominanz, während Frustration reduziert werden soll...
            print(f"😎 Regel-Engine (Ziel: Frustration reduzieren, Emotion: Hohe Dominanz ({dominance_emotion:.2f}), Valenz: {valence_emotion:.2f}, Arousal: {arousal_emotion:.2f}): 'Ich habe Kontrolle über die Frustration! Ziel überdenken?'")
            global_workspace.share(f"Regel-Engine-Hinweis: Dominanzgefühl bei Frustrationsreduktion. Dominanz: {dominance_emotion:.2f}")
            # **Visionär/Provokativ:**  Hier könnte der Agent *in Frage stellen*, ob das Ziel "Frustration reduzieren" noch sinnvoll ist, wenn er sich dominant/kontrollierend fühlt.
            # **Rudimentäre Reaktion (Beispiel):**  Priorität des Ziels "Frustration reduzieren" leicht senken, um Exploration wieder zu ermöglichen.
            if agent.goals["reduce_frustration"].priority == "mittel":
                agent.goals["reduce_frustration"].priority = "niedrig"
                agent.goals["explore"].priority = "mittel" # Priorität von "Erkunden" leicht erhöhen
                print(f"   -> Priorität von 'Frustration reduzieren' auf 'niedrig' gesenkt, 'Erkunden' auf 'mittel' erhöht (basierend auf Dominanzgefühl).")


        # **NEU: Selbstmodell-basierte Regeln (Beispiel)**
        self_awareness_norm = torch.norm(torch.tensor(self_awareness_vector)).item() # Norm des Selbstbewusstseins-Vektors als rudimentäres Maß
        if self_awareness_norm > 5.0 and current_goal_key == "explore": # Beispielschwellwert für "hohes Selbstbewusstsein"
            print(f"🌟 Regel-Engine (Selbstmodell-basiert, Ziel: Erkunden, Selbstbewusstsein hoch ({self_awareness_norm:.2f})): 'Ich bin mir meiner selbst stärker bewusst.  Vielleicht komplexere Exploration?'")
            global_workspace.share(f"Regel-Engine-Hinweis: Selbstbewusstsein hoch beim Erkunden. Selbstbewusstseins-Norm: {self_awareness_norm:.2f}")
            # **Visionär:** Hier könnte der Agent beginnen, komplexere Explorationsstrategien zu entwickeln, basierend auf seinem erhöhten Selbstverständnis.
            # **Rudimentäre Reaktion (Beispiel):**  Schwellwert für Neuigkeitsbonus leicht erhöhen, um anspruchsvollere neue Zustände zu suchen.
            agent.novelty_threshold += 0.1
            agent.novelty_threshold = np.clip(agent.novelty_threshold, 0.0, 1.0)
            print(f"   -> Neuigkeits-Schwellwert leicht erhöht auf {agent.novelty_threshold:.2f} (anspruchsvollere Exploration)")


        else:
            print("💬 Regel-Engine: Keine aktiven Regeln (Standard).")



# 🌐 Globaler Workspace zur Koordination
class GlobalWorkspace:
    def __init__(self):
        self.data = []
        self.rule_engine = RuleEngine() # Regel-Engine hinzufügen

    def share(self, info):
        """Daten mit anderen Agenten teilen."""
        self.data.append(info)

    def access(self):
        """Zufällige Information abrufen."""
        if self.data:
            return random.choice(self.data)
        return None

    def get_rule_engine(self): # Zugriff auf die Rule Engine ermöglichen
        return self.rule_engine


# **Neu: Funktion zum Laden der YAML-Konfiguration**
def lade_config_aus_yaml(dateipfad):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    with open(dateipfad, 'r') as file:
        config = yaml.safe_load(file)
    return config

# **Neu: Funktion zum Erstellen eines konfigurierten Agenten (nimmt jetzt Konfiguration entgegen)**
def erstelle_konfigurierten_agenten(config, workspace): # **Workspace als Parameter hinzugefügt**
    """Erstellt einen KI-Agenten mit der gegebenen Konfiguration."""
    return KI_Agent(config, workspace) # Workspace weitergeben


# 🚀 Hauptprogramm – Simulation des "Bewusstseins"
def simulation(config_path=None): # **Neu: config_path Parameter hinzugefügt**
    workspace = GlobalWorkspace()

    # **Konfiguration laden**
    if config_path:
        config = lade_config_aus_yaml(config_path) # Konfiguration aus YAML laden
        print(f"⚙️ Konfiguration geladen aus YAML-Datei: {config_path}")
    else:
        # **Default-Konfiguration (Dictionary-basiert), falls keine YAML-Datei angegeben**
        config = {
            'input_size': 3,
            'action_size': 2,
            'embed_dim': 32,
            'num_heads': 4,
            'state_similarity_threshold': 0.5,
            'emotion_dim': 3,
            'learning_rate': 0.01,
            'memory_capacity': 1000,
            'causal_capacity': 100,
            'prediction_model_hidden_size': 16,
            'layer_growth_threshold': 0.9,
            'novelty_threshold': 0.5,
            'novelty_tolerance': 0.5,
            'reward_history_window': 10,
            # **Beispiel-Callbacks (können in YAML oder Dictionary definiert werden, oder weggelassen werden)**
            'pre_decide_hook': pre_decide_callback_example, # Funktion als Callback
            'post_learn_hook': post_learn_callback_example, # Funktion als Callback
            'rule_engine_hook': rule_engine_callback_example # Funktion als Callback
        }
        print("⚙️ Verwende Default-Konfiguration (Dictionary-basiert).")


    agent = erstelle_konfigurierten_agenten(config, workspace) # Agent mit Konfiguration erstellen, Workspace übergeben

    # Stelle sicher, dass das gesamte Agentenmodell auf Float-Typ umgestellt ist.
    agent.float()

    rule_engine = workspace.get_rule_engine() # Rule Engine abrufen

    # **Demonstration: Direkter Zugriff auf Komponenten**
    print("\nℹ️ Demonstration: Direkter Zugriff auf Agenten-Komponenten:")
    print(f"  - Gedächtnis-Kapazität: {agent.memory_component.capacity}") # Zugriff auf Memory-Komponente
    print(f"  - Anzahl Attention-Ebenen: {agent.hierarchical_attention_component.num_layers}") # Zugriff auf Attention-Komponente
    print(f"  - Aktuelles Ziel-Key: {agent.self_model_component.current_goal_key}") # Zugriff auf Selbstmodell-Komponente

    for step in range(50):  # 500 Denkzyklen
        print(f"\n🔄 Schritt {step}: {agent.current_goal}, Subziel: {agent.current_subgoal}") # Ziel- & Subziel-Objekt wird ausgegeben
        # 1. 🌍 Agent nimmt Welt wahr
        state = np.random.rand(agent.input_size)  # Sensorische Wahrnehmung (nutze konfigurierte input_size)

        # 2. 🧠 Agent denkt & entscheidet
        action = agent.decide(state)

        # 3. 🔄 Feedback erhalten
        reward = np.random.choice([1, -1])  # Positive/Negative Erfahrung
        next_state = state + np.random.normal(0, 0.1, agent.input_size) # Einfache Simulation des nächsten Zustands (nutze konfigurierte input_size)
        agent.learn(state, action, reward, next_state)

        # 4. 📝 Agent speichert Gedanken
        memory = agent.memory.recall()
        if memory:
            workspace.share(f"Erinnerung: {memory}")

        # 5. 🌐 Workspace reflektiert über Wissen
        reflection = workspace.access()
        if reflection:
            print(f"🤔 Der Agent reflektiert: {reflection}")

        # 6. 🤖 Regelbasierter Innerer Dialog (rudimentär implementiert, zielbezogen)
        abstract_state_for_rules = agent.create_abstract_state() # Zustand einmal erstellen, um Ineffizienz zu reduzieren
        agent_state_for_rules = { # Beispiel für Agent-State, der an Rule-Engine übergeben wird
            'novelty_bonus': agent.calculate_novelty_bonus(abstract_state_for_rules),
            'coherence_reward': agent.calculate_coherence_reward(agent.reflect(abstract_state_for_rules)),
            'frustration_drive': agent.self_model.drives["frustration"], # Drives aus Selbstmodell
            'complexity_metric': agent.hierarchical_attention.calculate_complexity(agent.hierarchical_attention(abstract_state_for_rules)), # Komplexität berechnen
            'predictability': agent.world_model.calculate_state_properties(state, action).get('predictability', 0.0), # Predictability übergeben (ACTION hinzugefügt!)
            'valence': agent.world_model.calculate_state_properties(state, action).get('valence', 0.0), # Valence übergeben (ACTION hinzugefügt!)
            'current_goal_key': agent.current_goal_key, # Aktuellen Ziel-Key übergeben (für robustere Regel-Engine)
            'current_subgoal': agent.current_subgoal, # Aktuelles Subziel übergeben
            'emotion_state': agent.emotion_model.get_emotions().tolist(), # **Neu:** Emotionszustand übergeben (als Liste für Rule-Engine)
            'self_awareness_vector': agent.self_model.self_awareness_vector.tolist() # **NEU**: Selbstbewusstseins-Vektor übergeben
        }
        rule_engine.apply_rules(agent_state_for_rules, workspace, agent) # Rule-Engine anwenden, Agenten-Instanz übergeben


# **Beispiel-Callback-Funktionen (können in der Konfiguration angegeben werden)**
def pre_decide_callback_example(agent_instance, state):
    """Beispiel für einen pre_decide_hook Callback."""
    print("\n[Callback - PRE-DECIDE]: Agent steht kurz vor der Entscheidung. Zustand:", state)
    # Hier könnte man z.B. den Zustand manipulieren oder zusätzliche Analysen durchführen, BEVOR die Entscheidung fällt.

def post_learn_callback_example(agent_instance, state, action, reward, next_state):
    """Beispiel für einen post_learn_hook Callback."""
    print("\n[Callback - POST-LEARN]: Agent hat gerade gelernt. Gelernte Erfahrung:", (state, action, reward, next_state))
    # Hier könnte man z.B. Daten loggen, Visualisierungen erstellen oder externe Systeme informieren, NACHDEM der Agent gelernt hat.

def rule_engine_callback_example(agent_instance, agent_state, phase):
    """Beispiel für einen rule_engine_hook Callback."""
    print(f"\n[Callback - RULE-ENGINE - {phase.upper()}]: Regel-Engine wird angewendet (Phase: {phase}). Agent-Zustand (Auszug):")
    print(f"  - Ziel: {agent_state.get('current_goal_key')}, Subziel: {agent_state.get('current_subgoal')}")
    print(f"  - Neuigkeitsbonus: {agent_state.get('novelty_bonus'):.2f}, Frustration: {agent_state.get('frustration_drive'):.2f}")
    # Hier könnte man z.B. eigene Regeln hinzufügen, die Rule-Engine erweitern oder die Ausgabe der Rule-Engine analysieren.


if __name__ == "__main__":
    # **1. Simulation mit Default-Konfiguration (Dictionary)**
    print("\n======================== Simulation 1: Default-Konfiguration (Dictionary) ========================")
    simulation()