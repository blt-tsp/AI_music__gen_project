-Il est nécessaire d'installer les bibliothèques music21 et TensorFlow pour exécuter les différents programmes.


-Le dossier musiques contient les chansons en format MIDI sur lesquelles le réseau de neurones s'entraîne


-Le fichier Training.py contient le code python qui a permis de créer les tables de mappages qui associent à chaque note un nombre et inversement: model_rnn_int_to_vocab et model_rnn_vocab_to_int. De plus il il entraîne un modèle puis l'enregistre :model_rnn.h5

ATTENTION exécuter Training.py peut durer un long moment


-Le fichier CreateSong.py permet de créer une chanson à partir du modèle entraîner et avec test_son.mid qui contient une chanson que lequel le modèle n'est pas entraîné.



- L'exécution de CreateSong.py crée test_output.mid un fichier au format midi contenant la musique créée par notre réseau de neurones. Il suffit d'exécuter CreateSong.py pour avoir le résultat de notre travail.