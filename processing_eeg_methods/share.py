
# DATASETS BASIC INFO

# Edgar Aguilera Traditional
aguilera_traditional_info = {'#_class': 4, # avanzar, retroceder, derecha and izquierda
'target_names': ["Avanzar", "Retroceder", "Derecha", "Izquierda"],
'#_channels': 24,
'samples': 701, # sample_rate * duration in seconds = 500*1.4=700
'sample_rate': 500,
'channels_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz'], # in the channels order, 20 and 21 are  'M1', 'M2'
'subjects':16,
'total_trials':120} # 30 per class

# Edgar Aguilera Gamified
aguilera_gamified_info = {'#_class': 4, # avanzar, retroceder, derecha and izquierda
'target_names': ["Avanzar", "Derecha", "Izquierda", "Retroceder"], # Avanzar:1, Derecha:2, Izquierda:3, Retroceder: 4
'#_channels': 24,
'samples': 701, # sample_rate * duration in seconds = 500*1.4=700
'sample_rate': 500,
'channels_names':['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'M1', 'M2', 'AFz', 'CPz', 'POz'], # in the channels order, 21 and 22 are  'M1', 'M2'
'subjects':15,
'total_trials':120} # 30 per class

# Nieto (no prestar tanta atención)
nieto_info = {'#_class': 4, # arriba, abajo, derecha, izquierda
'target_names': ["Arriba", "Abajo", "Derecha", "Izquierda"],
'#_channels': 128,
'samples': 512,
'sample_rate': 256, # in BDF: 1024, but 256 is what the Python extraction tutorial they provided says.
'channels_names': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32'], # This is 10-5, not 10-20. That's why the channels barely correspond to the other datasets.
'subjects':10,
'total_trials':200} # Not really , it varies because some subjects couldn't finish the experiment

# Coretto (no prestar tanta atención)
# La frecuencia de muestreo se estrablecio en 1024Hz. De modo que cada intervalo de habla imaginada consta de 4096 muestras (4 segundos). Se implementaron filtrado digital pasabanda con frecuencias de paso de 2 y 45 HZ.
# De modo que el registro de una palabra esta constituido por 24576 muestras correspondientes a los canales de EEG, más tres muestras adicionales que indican la modalidad, estimulo y la presencia de artefactos oculares.
coretto_info = {'#_class': 6, # arriba, abajo, izquierda, derecha, adelante and atrás (last two not counted, but the db also has vocals: a, e, i, o, u)
'target_names': ["Arriba", "Abajo", "Derecha", "Izquierda"],
'#_channels': 6,
'samples': 342, # Originally 1365=(4096-1)/3 . Originally 4096, that would be 4s. But there were 3 trials inside that, so the first sample was removed and then divided in thirds.
'sample_rate': 256, # Originally it was 1024 with 1365 samples = 1.3 seconds. But I downsampled to 256.
'channels_names': ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
'subjects':15,
'total_trials':606} # TODO: Read the paper and find if this is right

# Torres
torres_info = {'#_class': 5, # 'arriba', 'abajo', 'izquierda', 'derecha'. They also did 'seleccionar', but we are not going to use that one.
"target_names": {"Arriba", "Abajo", "Izquierda", "Derecha"},
'#_channels': 14, # CHECK
'samples': 421, # PENDING
'sample_rate': 128,
'channels_names': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
'subjects':7, # Total of 27, but 7 because they haven't give me the full array.
'total_trials':132} # why did I thought that my trials were 606?

datasets_basic_infos = {'aguilera_traditional': aguilera_traditional_info, 'aguilera_gamified': aguilera_gamified_info, 'nieto':nieto_info, 'coretto':coretto_info, 'torres':torres_info} # If one day I decide to run all experiments at once