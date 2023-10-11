
# DATASETS BASIC INFO

# Edgar Aguilera Tradicional
aguilera_info = {'#_class': 4, # avanzar, retroceder, derecha and izquierda
'#_channels': 22, # Deleting the references. Andrea did this, you could try bringing them back maybe.
'samples':0, # PENDING
'sample_rate': 500,
'channels_names': [], # PENDING
'subjects':15}

# Nieto
nieto_info = {'#_class': 4, # arriba, abajo, derecha, izquierda
'#_channels': 128,
'samples': 0, # PENDING
'sample_rate': 256, # in BDF: 1024, but 256 is what the Python extraction tutorial they provided says.
'channels_names': [], # PENDING
'subjects':10}

# Coretto
# La frecuencia de muestreo se estrablecio en 1024Hz. De modo que cada intervalo de habla imaginada consta de 4096 muestras (4 segundos). Se implementaron filtrado digital pasabanda con frecuencias de paso de 2 y 45 HZ.
# De modo que el registro de una palabra esta constituido por 24576 muestras correspondientes a los canales de EEG, más tres muestras adicionales que indican la modalidad, estimulo y la presencia de artefactos oculares.
coretto_info = {'#_class': 6, # arriba, abajo, izquierda, derecha, adelante and atrás (plus vocals: a, e, i, o, u)
'#_channels': 6,
'samples': 0, # PENDING
'sample_rate': 1024,
'channels_names': ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
'subjects':15}

# Torres
torres_info = {'#_class': 5, # arriba, abajo, izquierda, derecha and seleccionar
'#_channels': 14, # CHECK
'samples': 0, # PENDING
'sample_rate': 128,
'channels_names': [], # PENDING
'subjects':27}

datasets_basic_infos = {'aguilera':aguilera_info, 'nieto':nieto_info, 'coretto':coretto_info, 'torres':torres_info} # If one day I decide to run all experiments at once