#agent_umgebung.py
import numpy as np
import time
import random
import torch
from ki_agent_konfigurierbar import KI_Agent, GlobalWorkspace, RuleEngine  # Annahme: Ihr Agentenmodell ist in agent_modell.py gespeichert


# 🌍 Soziale Umgebung mit mehreren Agenten
class SozialeUmgebung:
    def __init__(self, umgebungsgroesse=10, anzahl_agenten=2, zustands_groesse_agent=2): # Anzahl Agenten hinzugefügt, zustands_groesse_agent hinzugefügt
        self.umgebungsgroesse = umgebungsgroesse
        self.zustands_groesse_agent = zustands_groesse_agent # Grösse des Zustands PRO Agent speichern
        self.ziel_positionen = [(random.uniform(0, umgebungsgroesse), random.uniform(0, umgebungsgroesse)) for _ in range(anzahl_agenten)] # Ziele für jeden Agenten
        self.agenten_positionen = [(random.uniform(0, umgebungsgroesse), random.uniform(0, umgebungsgroesse)) for _ in range(anzahl_agenten)] # Startpositionen für jeden Agenten
        self.anzahl_agenten = anzahl_agenten # Anzahl Agenten speichern

    def reset(self):
        """Setzt die Umgebung und die Agentenpositionen zurück."""
        self.agenten_positionen = [(random.uniform(0, self.umgebungsgroesse), random.uniform(0, self.umgebungsgroesse)) for _ in range(self.anzahl_agenten)]
        return self.get_zustände() # Gibt Zustände für ALLE Agenten zurück

    def schritt(self, aktionen): # Nimmt Aktionen für ALLE Agenten entgegen
        """Führt einen Schritt in der Umgebung für alle Agenten aus."""
        naechste_zustände = []
        belohnungen = []
        fertig_werte = []

        for agent_index in range(self.anzahl_agenten): # Iteriere über Agenten
            aktion = aktionen[agent_index] # Aktion für aktuellen Agenten
            aktuelle_position = list(self.agenten_positionen[agent_index]) # Aktuelle Position als Liste (änderbar)
            ziel_position = self.ziel_positionen[agent_index] # Zielposition für aktuellen Agenten

            # Einfache Bewegungslogik (ähnlich wie zuvor)
            if aktion == 0: # Bewege in Richtung Ziel
                richtung_x = ziel_position[0] - aktuelle_position[0]
                richtung_y = ziel_position[1] - aktuelle_position[1]
                schrittgroesse = 0.5 # Schrittgröße beibehalten
                aktuelle_position[0] += richtung_x * schrittgroesse
                aktuelle_position[1] += richtung_y * schrittgroesse
            elif aktion == 1: # Zufällige Bewegung
                aktuelle_position[0] += random.uniform(-0.5, 0.5)
                aktuelle_position[1] += random.uniform(-0.5, 0.5)

            # Position innerhalb der Umgebung begrenzen
            aktuelle_position[0] = np.clip(aktuelle_position[0], 0, self.umgebungsgroesse)
            aktuelle_position[1] = np.clip(aktuelle_position[1], 0, self.umgebungsgroesse)
            self.agenten_positionen[agent_index] = tuple(aktuelle_position) # Zurück als Tuple speichern

            # Belohnung basierend auf Distanz zum eigenen Ziel (individuell für jeden Agenten)
            distanz_zum_ziel = np.sqrt((aktuelle_position[0] - ziel_position[0])**2 + (aktuelle_position[1] - ziel_position[1])**2)
            belohnung = -distanz_zum_ziel # Negative Distanz als Belohnung (je näher, desto besser)

            naechste_zustände.append(self.get_agenten_zustand(agent_index)) # Zustand für aktuellen Agenten
            belohnungen.append(belohnung)
            fertig_werte.append(False) # "Fertig" in dieser Umgebung nicht relevant

        return naechste_zustände, belohnungen, fertig_werte # Gibt Listen für alle Agenten zurück

    def get_zustände(self):
        """Gibt die Zustände für alle Agenten zurück (individuell)."""
        zustände = []
        for agent_index in range(self.anzahl_agenten):
            zustände.append(self.get_agenten_zustand(agent_index))
        return zustände

    def get_agenten_zustand(self, agent_index):
            """Gibt den Zustand für einen bestimmten Agenten zurück, inkl. Positionen anderer Agenten."""
            zustand = [self.agenten_positionen[agent_index][0], self.agenten_positionen[agent_index][1]] # Eigene Position
            # **Kreative Fehlerbehebung:** Zustandsgröße des Agenten dynamisch anpassen!
            if self.zustands_groesse_agent > 2: # Nur Positionsdaten anderer Agenten hinzufügen, wenn Zustand grösser als 2 sein soll.
                for i in range(self.anzahl_agenten):
                    if i != agent_index: # Positionen der ANDEREN Agenten hinzufügen
                        zustand.append(self.agenten_positionen[i][0])
                        zustand.append(self.agenten_positionen[i][1])
            return np.array(zustand)

    def render(self):
        """Rudimentäre Text-basierte Visualisierung der Umgebung."""
        print("-" * 30)
        for agent_index in range(self.anzahl_agenten):
            agent_pos = self.agenten_positionen[agent_index]
            ziel_pos = self.ziel_positionen[agent_index]
            print(f"Agent {agent_index}: Position = ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}), Ziel = ({ziel_pos[0]:.2f}, {ziel_pos[1]:.2f})")
        print("-" * 30)


if __name__ == "__main__":
    anzahl_agenten_simulation = 2 # Anzahl Agenten in der Simulation
    zustands_groesse_pro_agent = 2 # Zustand besteht nur aus eigener Position, keine Positionen anderer Agenten
    umgebung = SozialeUmgebung(anzahl_agenten=anzahl_agenten_simulation, zustands_groesse_agent=zustands_groesse_pro_agent) # zustands_groesse_agent Parameter hinzugefügt
    agenten_configs = [ # Konfigurationen für jeden Agenten
        {'input_size': zustands_groesse_pro_agent, 'action_size': 2, 'emotion_dim': 3}, # Konfiguration für Agent 0, input_size dynamisch
        {'input_size': zustands_groesse_pro_agent, 'action_size': 2, 'emotion_dim': 3}  # Konfiguration für Agent 1 (oder unterschiedliche Konfigs), input_size dynamisch
    ]
    agenten = [KI_Agent(config=agenten_configs[i], workspace=GlobalWorkspace()) for i in range(anzahl_agenten_simulation)] # Liste von Agenteninstanzen mit Konfigs & Workspace
    workspaces = [GlobalWorkspace() for _ in range(anzahl_agenten_simulation)] # Workspace für jeden Agenten -  bereits in Agenten Instanziierung oben integriert, nicht mehr separat benötigt
    rule_engines = [RuleEngine() for _ in range(anzahl_agenten_simulation)] # Rule Engine für jeden Agenten -  nicht direkt genutzt in diesem Beispiel, aber vorhanden

    # Agenten mit Workspaces und Rule Engines verbinden (falls nötig - Architekturfrage)
    for i in range(anzahl_agenten_simulation):
        # Beispiel: Workspace und Rule Engine dem Agenten zuweisen (Architektur abhängig)
        # agenten[i].global_workspace = workspaces[i] # Wenn Agent direkten Workspace-Zugriff hat
        # agenten[i].rule_engine = rule_engines[i] # Wenn Agent direkten Rule-Engine Zugriff hat
        pass # Oder übergeben Sie Workspace/RuleEngine Funktionen an Agenten, falls gewünscht

    for agent in agenten: # Stelle sicher, dass alle Agenten-Modelle auf Float-Typ umgestellt sind.
        agent.float()

    episodes = 100
    schritte_pro_episode = 100

    for episode in range(episodes):
        zustände = umgebung.reset() # Reset der Umgebung liefert Zustände für ALLE Agenten
        print(f"\nEpisode {episode + 1}/{episodes} gestartet:")

        for schritt in range(schritte_pro_episode):
            aktionen = []
            for agent_index in range(anzahl_agenten_simulation): # Entscheidungen für jeden Agenten
                agent = agenten[agent_index]
                zustand = zustände[agent_index]
                aktion = agent.decide(zustand) # Agent entscheidet basierend auf SEINEM Zustand
                aktionen.append(aktion)
                print(f"  Agent {agent_index}: Zustand = {zustand}, Aktion = {aktion}", end=" ")

            naechste_zustände, belohnungen, fertig_werte = umgebung.schritt(aktionen) # Umgebungsschritt mit Aktionen ALLER Agenten
            umgebung.render() # Umgebung visualisieren

            for agent_index in range(anzahl_agenten_simulation): # Lernen für jeden Agenten
                agent = agenten[agent_index]
                zustand = zustände[agent_index]
                aktion = aktionen[agent_index]
                belohnung = belohnungen[agent_index]
                naechster_zustand = naechste_zustände[agent_index]
                agent.learn(zustand, aktion, belohnung, naechster_zustand) # Agent lernt INDIVIDUELL

                print(f"| Belohnung = {belohnung:.3f}, Emotionen (Agent {agent_index}): Valenz: {agent.emotion_model.get_emotions()[0]:.2f}, Arousal: {agent.emotion_model.get_emotions()[1]:.2f}, Dominanz: {agent.emotion_model.get_emotions()[2]:.2f}, Drives: Neugier: {agent.self_model.drives['curiosity']:.2f}, Verständnis: {agent.self_model.drives['understanding']:.2f}, Frustration: {agent.self_model.drives['frustration']:.2f}, Ziel: {agent.current_goal_key}, Subziel: {agent.current_subgoal}")


            zustände = naechste_zustände # Nächste Zustände für den nächsten Schritt

            time.sleep(0.1) # Langsamere Simulation für bessere Beobachtung

    print("\nTraining abgeschlossen.")