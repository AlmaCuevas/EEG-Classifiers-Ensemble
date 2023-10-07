
# DATASETS BASIC INFO

# Edgar Aguilera Tradicional
aguilera_info = {'mi_class': 4, # avanzar, retroceder, derecha and izquierda
'channels': 24,
'samples':0, # PENDING
'sample_rate': 500,
'channels_names': [], # PENDING
'subjects':15}

# Nieto
nieto_info = {'mi_class': 4, # arriba, abajo, derecha, izquierda
'channels': 128,
'samples': 0, # PENDING
'sample_rate':1024, # or 254 pending check
'channels_names': [], # PENDING
'subjects':10}

# Coretto
# La frecuencia de muestreo se estrablecio en 1024Hz. De modo que cada intervalo de habla imaginada consta de 4096 muestras (4 segundos). Se implementaron filtrado digital pasabanda con frecuencias de paso de 2 y 45 HZ.
coretto_info = {'mi_class': 6, # arriba, abajo, izquierda, derecha, adelante and atrás (plus vocals: a, e, i, o, u)
'channels': 6,
'samples': 0, # PENDING
'sample_rate': 1024,
'channels_names': ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
'subjects':15}

# Torres
torres_info = {'mi_class': 5, # arriba, abajo, izquierda, derecha and seleccionar
'channels': 14, # CHECK
'samples': 0, # PENDING
'sample_rate': 128,
'channels_names': [], # PENDING
'subjects':27}

pipeline = [aguilera_info, nieto_info, coretto_info, torres_info] # If one day I decide to run all experiments at once