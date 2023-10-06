
# DATASETS BASIC INFO

# Nieto
nieto_mi_class = 4 # arriba, abajo, derecha, izquierda
nieto_channels = 128
nieto_samples =
nieto_sample_rate = 1024 # or 254 pending check

nieto_subjects=10

# Coretto
# La frecuencia de muestreo se estrablecio en 1024Hz. De modo que cada intervalo de habla imaginada consta de 4096 muestras (4 segundos). Se implementaron filtrado digital pasabanda con frecuencias de paso de 2 y 45 HZ.
coretto_mi_class = 6 # arriba, abajo, izquierda, derecha, adelante and atrás (plus vocals: a, e, i, o, u)
coretto_channels = 6
coretto_samples =
coretto_sample_rate = 1024
coretto_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

coretto_subjects=15

# Edgar Aguilera Tradicional
aguilera_mi_class = 4 # avanzar, retroceder, derecha and izquierda
aguilera_channels = 24
aguilera_samples =
aguilera_sample_rate = 500

aguilera_subjects=15

# Torres
torres_mi_class = 5 # arriba, abajo, izquierda, derecha and seleccionar
torres_channels = 14 # CHECK
torres_samples =
torres_sample_rate = 128

torres_subjects=27

# Order of pipeline
pipeline_mi_class=[nieto_mi_class, coretto_mi_class, aguilera_mi_class, torres_mi_class]
pipeline_channels=[nieto_channels, coretto_channels, aguilera_channels, torres_channels]
pipeline_samples=[nieto_samples, coretto_samples, aguilera_samples, torres_samples]
pipeline_sample_rate=[nieto_sample_rate, coretto_sample_rate, aguilera_sample_rate, torres_sample_rate]
pipeline_subjects=[nieto_subjects, coretto_subjects, aguilera_subjects, torres_subjects]