# ------------------------------------- -*- coding : utf-8 -*-
# perceptron simple
# input : x1, x2 + biais  --> 3 poids � apprendre
# avec plot des droites initiale, interm�diaires et finale
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        # poids tir�s au hasard entre -0.5 et 0.5
        self.weights = np.random.rand(3) - 0.5

    def heaviside(self, x):
        # Fonction d'activation la plus simple: step
        return 1 if x >= 0 else 0

    def plot(self, color='b'):
        # Dessin de la courbe correspondant aux poids courants, couleur en param�tre
        a1, a2, a0 = self.weights
        intercept = -a0/a2
        slope = -a1/a2
        plt.axis('equal')
        plt.axline((0, intercept), slope=slope, linestyle='--', color=color)

    def prediction(self, inputs):
        # Les deux entrees forment un vecteur avec le biais (=1)
        vect_in = np.append(inputs, 1)
        # Calcul de la somme ponderee: produit scalaire
        # et application de la fonction d'activation
        return self.heaviside(np.dot(vect_in, self.weights))

    def train(self, data_ref, learning_rate=0.1, epochs=10):
        for epoch in range(epochs):
            for x1, x2, label in data_ref:
                # R�cup�ration de la valeur pr�dite
                prediction = self.prediction([x1, x2])
                # Les deux entr�es forment un vecteur avec le biais (=1)
                vect_in = np.array([x1, x2, 1])
                # Mise � jour des poids 
                self.weights += learning_rate * (label - prediction) * vect_in
                # On plotte la courbe pour voir
                self.plot('y')


# Donn�es de r�f�rence (entr�e_1, entr�e_2, label)
data_ref = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])  # AND
#data_ref = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])  # OR

def plotte_config_AND():
    plt.plot(0,0,'bo')
    plt.plot(0,1,'bo')
    plt.plot(1,0,'bo')
    plt.plot(1,1,'rx')

def plotte_config_OR():
    plt.plot(0,0,'bo')
    plt.plot(0,1,'rx')
    plt.plot(1,0,'rx')
    plt.plot(1,1,'rx')
    
    
# Initialisation du perceptron (donc des poids)
perceptron = Perceptron()
# Plot de la droite correspondant au d�part
perceptron.plot('m')
# Entrainement du perceptron
# Hyperparam�tres
nbepochs = 5
nu = .1
perceptron.train(data_ref,learning_rate = nu, epochs=nbepochs)
# Plot en rouge de la derni�re configuration de poids
perceptron.plot('r')


# Mesure du taux d'erreurs
er = 0
for x1, x2, label in data_ref:
    # R�cup�ration de la valeur pr�dite
    prediction = perceptron.prediction([x1, x2])
    er += prediction != label

titre_figure = ("APPROXIMATION R�USSIE" if er == 0 else "PAS DE CONVERGENCE") + " (%d �poques, lr = %.2f)" % (nbepochs, nu)

plt.title(titre_figure)
plotte_config_AND()
plt.show()

