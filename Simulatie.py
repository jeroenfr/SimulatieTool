import numpy as np
import pandas as pd
import seaborn as sns

class Simulatie:
    """
    Klasse voor het simuleren van gegevens en het berekenen van empirische waarschijnlijkheid.
    """
    def __init__(self):
        """
        Initialiseert de Simulatie klasse.

        Attributen:
        - drempel (float): De drempelwaarde voor succes.
        - y (DataFrame): De gegenereerde gegevens.
        """
        self.drempel = None
        self.y = None
    
    def genereer_data(self, p_0, n, p_a, aantal_samples):
        """
        Genereert gegevens op basis van de opgegeven parameters en het aantal samples.

        Parameters:
        - p_0 (float): De kans op succes voor elke poging.
        - n (int): Het aantal pogingen.
        - p_a (float): De drempelwaarde voor succes.
        - aantal_samples (int): Het aantal samples om te genereren.
        """
        self.drempel = p_a * n
        x = np.random.binomial(n, p_0, aantal_samples)
        self.y = pd.DataFrame(x, columns=['waarden'])
        self.y['vlag'] = np.where(self.y['waarden'] >= self.drempel, 1, 0)
    
    def bereken_empirische_p(self):
        """
        Berekent de empirische waarschijnlijkheid.

        Returns:
        - float: De empirische waarschijnlijkheid.
        """
        if self.y is None:
            raise ValueError("Data is nog niet gegenereerd. Roep eerst genereer_data() aan.")
        return (np.where(self.y['waarden'] >= self.drempel, 1, 0).sum()) / self.y.shape[0]
    
    def plot_verdeling(self):
        """
        Plot de verdeling van de gegenereerde gegevens.
        """
        if self.y is None:
            raise ValueError("Data is nog niet gegenereerd. Roep eerst genereer_data() aan.")
        sns.displot(self.y['waarden'], kde=True, palette='flare')