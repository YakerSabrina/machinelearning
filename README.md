Projet classification d'image de la base de donnée MNIST

La base de donnée MNIST(Mixed National Institute of Standards and Technology) est constituée d’un échantillon d’apprentissage : 60 000 images 
de 28x28 pixels à niveau de gris et d’un échantillon test de 10 000 images. 
L’objectif général de ce projet est la construction d’un modèle de reconnaissance de ces chiffres manuscrits de la base MNIST du site de Yann LeCun, 
source des données étudiées.

Dans un premier temps j’ai créé un modèle séquentiel entièrement connecté composé de 6 couches (une couche pour les entrées, une couche pour les sorties, 
et quatre couches intermédiaires) pour la reconnaissance de chiffres avec Keras et Tensorﬂow (bibliothèques open source de machine learning), 
puis je l’ai entraîné sur les 60000 images de la base de données MNIST, après les avoir chargées et redimensionnées sous forme de vecteurs de taille 784 pixels.
J’ai enﬁn sauvegardé ce modèle.

Pour vériﬁer que le modèle est bien entraîné, je l’ai testé sur des images test de la base de données MNIST qu’il n’a jamais vues. 
Ensuite, j’ai afﬁché le résultat sous forme de vidéo qui prend en entrée dix images test et nous donne en sortie les prédictions faites par le modèle à 1 fps,
avec la librairie OpenCV.

