
# DATASETS BASIC INFO

# Edgar Aguilera Tradicional
aguilera_info = {'#_class': 4, # avanzar, retroceder, derecha and izquierda
'target_names': ["Avanzar", "Retroceder", "Derecha", "Izquierda"],
'#_channels': 22, # Deleting the references. Andrea did this, you could try bringing them back maybe.
'samples': 700, # sample_rate * duration in seconds = 500*1.4=700
'sample_rate': 500,
'channels_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz'], # in the channels order, 21 and 22 are  'M1', 'M2'
'subjects':15,
'total_trials':120} # 30 per class

# Nieto
nieto_info = {'#_class': 4, # arriba, abajo, derecha, izquierda
'target_names': ["Arriba", "Abajo", "Derecha", "Izquierda"],
'#_channels': 128,
'samples': 512,
'sample_rate': 256, # in BDF: 1024, but 256 is what the Python extraction tutorial they provided says.
'channels_names': [], # PENDING
'subjects':10,
'total_trials':200} # Not really , it varies because some subjects couldn't finish the experiment

# Coretto
# La frecuencia de muestreo se estrablecio en 1024Hz. De modo que cada intervalo de habla imaginada consta de 4096 muestras (4 segundos). Se implementaron filtrado digital pasabanda con frecuencias de paso de 2 y 45 HZ.
# De modo que el registro de una palabra esta constituido por 24576 muestras correspondientes a los canales de EEG, más tres muestras adicionales que indican la modalidad, estimulo y la presencia de artefactos oculares.
coretto_info = {'#_class': 6, # arriba, abajo, izquierda, derecha, adelante and atrás (last two not counted, but the db also has vocals: a, e, i, o, u)
'target_names': ["Arriba", "Abajo", "Derecha", "Izquierda"],
'#_channels': 6,
'samples': 1365, # =(4096-1)/3 . Originally 4096, that would be 4s. But there were 3 trials inside that, so the first sample was removed and then divided in thirds.
'sample_rate': 1024, # with 1365 samples, it's 1.3 seconds
'channels_names': ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
'subjects':15,
'total_trials':606} # TODO: Read the paper and find if this is right

# Torres
torres_info = {'#_class': 5, # 'arriba', 'abajo', 'izquierda', 'derecha' and 'seleccionar'
'#_channels': 14, # CHECK
'samples': 0, # PENDING
'sample_rate': 128,
'channels_names': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
'subjects':27}

datasets_basic_infos = {'aguilera':aguilera_info, 'nieto':nieto_info, 'coretto':coretto_info, 'torres':torres_info} # If one day I decide to run all experiments at once