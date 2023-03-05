    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:10:19 2021

@author: yannvastel
"""
import tensorflow as tf
import numpy as np
import json
from music21 import converter, instrument, note, chord, stream

#OnehotEncoding Layer
class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)


    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['depth'] = self.depth
        
        return config



model=tf.keras.models.load_model("model_rnn.h5", custom_objects={'OneHot': OneHot}, compile=False)


@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions

#On prend les tables de mappages créées auparavant
with open("model_rnn_int_to_vocab", "r") as f:
    int_to_vocab=json.loads(f.readline())

with open("model_rnn_vocab_to_int", "r") as f:
    vocab_to_int=json.loads(f.readline())

#On transforme le fichier test_son en liste de notes

notes2= []
testmid= converter.parse("test_son.mid")
parts = instrument.partitionByInstrument(testmid)
if parts: #si le fichier contient plusieurs instrument on prend que les notes du premier
    notes_to_parse2 = parts.parts[0].recurse()   
else:  
    notes_to_parse2 = testmid.flat.notes

# on remplit notes des notes du fichier midi   
for element in notes_to_parse2:   
    if isinstance(element, note.Note):
        notes2.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        for n in element.notes :
            notes2.append(str(n.pitch))

#On associe à chaque notes un nombre selon la table de mappages

encoded=[vocab_to_int[l] for l in notes2]



# On crée la chason song en partant d'une séquence aléatoire de test_son


size_song= 300
song=np.zeros((90,size_song,1))

sequences=np.zeros((90,100))
for b in range (90):
    r = np.random.randint(0, len(encoded) - 100)  
    sequences[b]=encoded[r:r+100]   #séquence aléatoire de test_son
for i in range(size_song):
    if i>0:
        song[:,i-1,:]=sequences
    softmax=predict(sequences)   # On récupère la sortie du modèle
    
    sequences=np.zeros((90,1))
    for b in range(90):
        argsort=np.argsort(softmax[b][0]) # on rcupère les notes suivantes les plus probables selon notre modèle
        sequences[b]=argsort[-1]

        
#On a créé 90 chansons de 300 notes
# On récupère seulement la première qu'on met dans prediction_outputencoded sous forme de liste simple
prediction_outputencoded=[]
for i in range (len(song[0])):
    prediction_outputencoded.append(int(song[0][i][0]))





#Utilisation de la table de mappages pour transformer les nombres en leurs notes associées


prediction_outputdecoded=[int_to_vocab[str(i)] for i in prediction_outputencoded]



#Conversion en fichier midi

output_notes=[]

offset=0
for n in prediction_outputdecoded:
    new_note = note.Note(n)
    new_note.offset = offset
    new_note.storedInstrument = instrument.Piano()
    output_notes.append(new_note)
    offset+=0.5


#On écrit le fichier midi test_output.mid qui est le fichier à écouter

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')