# Supplementary results

Extended results behind the paper. Generated from `Results/ensemble_benchmark/` by the
paper's build script; do not edit by hand. Balanced accuracy in %, gains in percentage
points (pp). Development datasets were used to design the method; confirmatory datasets
were not. Names follow the paper: main ensemble (`v1` in PROTOCOL.md and the code),
curated ensemble (`v2`), classical ensemble (`fast`); experiments 3 and 4 are
PROTOCOL.md Amendments A2 and A3.

## S1. Every subject

| set          | dataset           |   subject | main: voter picked                 |   main: picked (%) |   main ensemble (%) |   main: gain (pp) |   curated: picked (%) |   curated ensemble (%) |   classical ensemble (%) |   tuned ensemble (%) |
|:-------------|:------------------|----------:|:--------------------------------|----------------:|------------------:|---------------:|----------------:|------------------:|--------------------:|-------------------------:|
| Development  | BNCI2014_001      |         1 | green                           |            86.5 |              86.5 |            0.0 |            82.6 |              86.5 |                80.9 |                     88.5 |
| Development  | BNCI2014_001      |         2 | covariance_tangent_space_4_30Hz |            50.3 |              56.9 |            6.6 |            57.6 |              64.9 |                62.5 |                     61.8 |
| Development  | BNCI2014_001      |         3 | green                           |            87.8 |              91.7 |            3.8 |            89.9 |              92.7 |                89.2 |                     91.7 |
| Development  | BNCI2014_001      |         4 | covariance_tangent_space        |            71.9 |              79.2 |            7.3 |            68.8 |              77.1 |                72.2 |                     80.6 |
| Development  | BNCI2014_001      |         5 | green                           |            61.8 |              66.3 |            4.5 |            66.3 |              70.5 |                62.8 |                     59.7 |
| Development  | BNCI2014_001      |         6 | covariance_tangent_space_4_30Hz |            54.9 |              62.2 |            7.3 |            55.9 |              63.2 |                58.7 |                     61.1 |
| Development  | BNCI2014_001      |         7 | covariance_tangent_space        |            84.7 |              88.2 |            3.5 |            84.7 |              89.2 |                83.7 |                     86.8 |
| Development  | BNCI2014_001      |         8 | green                           |            79.9 |              83.7 |            3.8 |            74.3 |              83.0 |                82.3 |                     81.6 |
| Development  | BNCI2014_001      |         9 | shallownet                      |            70.5 |              74.3 |            3.8 |            78.8 |              76.7 |                71.9 |                     76.7 |
| Development  | BNCI2014_004      |         1 | green                           |            71.5 |              75.2 |            3.7 |            71.5 |              75.8 |                75.7 |                     71.7 |
| Development  | BNCI2014_004      |         2 | complexity_boosting             |            56.8 |              59.6 |            2.9 |            56.8 |              60.9 |                63.4 |                     58.7 |
| Development  | BNCI2014_004      |         3 | covariance_tangent_space_4_30Hz |            53.0 |              49.3 |           -3.7 |            52.2 |              49.8 |                56.3 |                     51.0 |
| Development  | BNCI2014_004      |         4 | csp_lda_8_30Hz                  |            93.2 |              93.2 |            0.0 |            88.9 |              94.0 |                91.9 |                     92.7 |
| Development  | BNCI2014_004      |         5 | covariance_tangent_space_4_30Hz |            70.3 |              85.3 |           15.0 |            77.4 |              84.8 |                83.1 |                     84.4 |
| Development  | BNCI2014_004      |         6 | csp_lda_8_30Hz                  |            67.7 |              74.2 |            6.5 |            78.5 |              73.7 |                69.3 |                     72.7 |
| Development  | BNCI2014_004      |         7 | green                           |            74.5 |              73.2 |           -1.3 |            69.5 |              72.2 |                68.0 |                     71.5 |
| Development  | BNCI2014_004      |         8 | relative_spectrum               |            77.5 |              82.5 |            5.0 |            77.5 |              75.5 |                71.3 |                     80.5 |
| Development  | BNCI2014_004      |         9 | green                           |            71.3 |              76.7 |            5.3 |            71.3 |              76.7 |                79.2 |                     77.0 |
| Development  | BNCI2014_009      |         1 | eegnet                          |            80.7 |              85.5 |            4.8 |            82.9 |              85.0 |                82.5 |                     86.8 |
| Development  | BNCI2014_009      |        10 | eegnet                          |            88.2 |              90.9 |            2.8 |            88.2 |              92.1 |                90.2 |                     90.5 |
| Development  | BNCI2014_009      |         2 | erp_time_course                 |            82.7 |              89.0 |            6.2 |            79.8 |              88.6 |                86.0 |                     88.3 |
| Development  | BNCI2014_009      |         3 | eegnet                          |            67.5 |              71.5 |            4.0 |            71.6 |              72.6 |                69.0 |                     69.3 |
| Development  | BNCI2014_009      |         4 | erp_time_course                 |            82.8 |              89.4 |            6.6 |            82.8 |              88.3 |                85.6 |                     88.5 |
| Development  | BNCI2014_009      |         5 | eegnet                          |            81.2 |              87.2 |            5.9 |            81.2 |              88.4 |                87.5 |                     88.1 |
| Development  | BNCI2014_009      |         6 | eegnet                          |            72.3 |              77.7 |            5.3 |            72.3 |              79.6 |                74.0 |                     84.4 |
| Development  | BNCI2014_009      |         7 | eegnet                          |            76.9 |              81.1 |            4.3 |            81.7 |              84.9 |                82.0 |                     82.1 |
| Development  | BNCI2014_009      |         8 | eegnet                          |            77.1 |              79.8 |            2.7 |            77.1 |              83.7 |                77.7 |                     79.7 |
| Development  | BNCI2014_009      |         9 | eegnet                          |            88.1 |              91.0 |            2.9 |            88.1 |              90.5 |                85.7 |                     90.0 |
| Development  | BNCI2015_001      |         1 | green                           |            98.0 |              99.5 |            1.5 |            99.0 |              99.5 |                98.0 |                    100.0 |
| Development  | BNCI2015_001      |        10 | covariance_tangent_space        |            69.2 |              81.0 |           11.8 |            72.8 |              83.2 |                74.2 |                     81.5 |
| Development  | BNCI2015_001      |        11 | green                           |            65.5 |              69.5 |            4.0 |            68.5 |              71.5 |                76.2 |                     72.8 |
| Development  | BNCI2015_001      |        12 | coherence                       |            56.5 |              62.5 |            6.0 |            78.5 |              76.5 |                68.0 |                     62.0 |
| Development  | BNCI2015_001      |         2 | green                           |            97.0 |              96.5 |           -0.5 |            97.0 |              97.5 |                96.5 |                     96.5 |
| Development  | BNCI2015_001      |         3 | covariance_tangent_space_4_30Hz |            71.0 |              92.5 |           21.5 |            71.0 |              95.0 |                79.0 |                     92.5 |
| Development  | BNCI2015_001      |         4 | green                           |            89.0 |              92.5 |            3.5 |            89.0 |              96.0 |                88.0 |                     91.5 |
| Development  | BNCI2015_001      |         5 | green                           |            86.5 |              90.0 |            3.5 |            88.0 |              92.0 |                86.0 |                     92.0 |
| Development  | BNCI2015_001      |         6 | green                           |            68.5 |              74.0 |            5.5 |            72.5 |              72.5 |                61.5 |                     74.0 |
| Development  | BNCI2015_001      |         7 | green                           |            81.0 |              91.5 |           10.5 |            81.0 |              94.0 |                94.5 |                     92.0 |
| Development  | BNCI2015_001      |         8 | covariance_tangent_space_4_30Hz |            76.8 |              81.5 |            4.8 |            73.2 |              83.5 |                77.8 |                     82.8 |
| Development  | BNCI2015_001      |         9 | csp_lda_8_30Hz                  |            80.5 |              84.8 |            4.2 |            87.8 |              88.0 |                86.0 |                     83.2 |
| Confirmatory | BNCI2015_003      |         1 | erp_time_course                 |            69.8 |              69.9 |            0.1 |            69.8 |              64.4 |                70.3 |                     68.9 |
| Confirmatory | BNCI2015_003      |        10 | erp_time_course                 |            79.7 |              79.5 |           -0.2 |            78.1 |              81.1 |                80.6 |                     82.7 |
| Confirmatory | BNCI2015_003      |         2 | erp_time_course                 |            80.5 |              87.4 |            6.9 |            82.4 |              84.7 |                86.3 |                     86.6 |
| Confirmatory | BNCI2015_003      |         3 | erp_time_course                 |            74.7 |              72.3 |           -2.3 |            74.7 |              72.5 |                71.4 |                     71.9 |
| Confirmatory | BNCI2015_003      |         4 | erp_time_course                 |            72.3 |              71.9 |           -0.3 |            72.3 |              72.7 |                74.5 |                     72.9 |
| Confirmatory | BNCI2015_003      |         5 | erp_time_course                 |            73.5 |              73.7 |            0.3 |            73.5 |              76.1 |                77.1 |                     72.5 |
| Confirmatory | BNCI2015_003      |         6 | erp_time_course                 |            58.7 |              58.5 |           -0.2 |            58.7 |              58.4 |                58.3 |                     59.4 |
| Confirmatory | BNCI2015_003      |         7 | eegnet                          |            66.7 |              71.9 |            5.2 |            65.5 |              69.9 |                70.5 |                     71.3 |
| Confirmatory | BNCI2015_003      |         8 | eegnet                          |            78.9 |              83.2 |            4.3 |            78.1 |              82.1 |                79.8 |                     82.6 |
| Confirmatory | BNCI2015_003      |         9 | eegnet                          |            79.0 |              83.7 |            4.7 |            79.0 |              84.1 |                80.4 |                     81.5 |
| Confirmatory | BNCI2015_004      |         1 | complexity_boosting             |            30.5 |              28.5 |           -2.0 |            30.5 |              30.5 |                26.5 |                     32.0 |
| Confirmatory | BNCI2015_004      |         2 | shallownet                      |            22.5 |              26.5 |            4.0 |            24.0 |              25.0 |                29.0 |                     25.5 |
| Confirmatory | BNCI2015_004      |         3 | green                           |            39.0 |              50.5 |           11.5 |            44.0 |              50.0 |                46.5 |                     47.0 |
| Confirmatory | BNCI2015_004      |         4 | covariance_tangent_space_4_30Hz |            42.3 |              47.4 |            5.1 |            37.1 |              43.4 |                44.0 |                     40.6 |
| Confirmatory | BNCI2015_004      |         5 | covariance_tangent_space        |            38.0 |              46.0 |            8.0 |            38.0 |              41.0 |                35.5 |                     41.5 |
| Confirmatory | BNCI2015_004      |         6 | green                           |            43.0 |              30.5 |          -12.5 |            54.5 |              31.0 |                27.0 |                     31.0 |
| Confirmatory | BNCI2015_004      |         7 | relative_spectrum               |            31.4 |              25.7 |           -5.7 |            29.7 |              21.7 |                27.4 |                     28.0 |
| Confirmatory | BNCI2015_004      |         8 | shallownet                      |            37.0 |              45.0 |            8.0 |            41.5 |              40.5 |                38.5 |                     41.5 |
| Confirmatory | BNCI2015_004      |         9 | covariance_tangent_space_4_30Hz |            26.5 |              32.0 |            5.5 |            29.5 |              33.5 |                32.0 |                     36.0 |
| Development  | Kalunga2016       |         1 | csp_lda_8_30Hz                  |            25.0 |              28.1 |            3.1 |            31.2 |              43.8 |                40.6 |                     21.9 |
| Development  | Kalunga2016       |        10 | covariance_tangent_space        |            46.9 |              59.4 |           12.5 |            45.3 |              54.7 |                46.9 |                     56.2 |
| Development  | Kalunga2016       |        11 | covariance_tangent_space        |            40.6 |              40.6 |            0.0 |            40.6 |              40.6 |                43.8 |                     43.8 |
| Development  | Kalunga2016       |        12 | covariance_tangent_space        |            64.6 |              80.2 |           15.6 |            68.8 |              79.2 |                76.0 |                     84.4 |
| Development  | Kalunga2016       |         2 | covariance_tangent_space        |            46.9 |              65.6 |           18.8 |            46.9 |              65.6 |                59.4 |                     68.8 |
| Development  | Kalunga2016       |         3 | relative_spectrum               |            68.8 |              75.0 |            6.2 |            68.8 |              71.9 |                71.9 |                     71.9 |
| Development  | Kalunga2016       |         4 | csp_lda_8_30Hz                  |            50.0 |              50.0 |            0.0 |            46.9 |              71.9 |                65.6 |                     50.0 |
| Development  | Kalunga2016       |         5 | covariance_tangent_space        |            68.8 |              59.4 |           -9.4 |            68.8 |              65.6 |                56.2 |                     59.4 |
| Development  | Kalunga2016       |         6 | covariance_tangent_space        |            68.8 |              46.9 |          -21.9 |            68.8 |              53.1 |                56.2 |                     46.9 |
| Development  | Kalunga2016       |         7 | relative_spectrum               |            31.2 |              34.4 |            3.1 |            31.2 |              29.7 |                34.4 |                     34.4 |
| Development  | Kalunga2016       |         8 | covariance_tangent_space        |            65.6 |              59.4 |           -6.2 |            59.4 |              75.0 |                68.8 |                     62.5 |
| Development  | Kalunga2016       |         9 | riemann_mdm_8_30Hz              |            46.9 |              43.8 |           -3.1 |            50.0 |              71.9 |                65.6 |                     50.0 |
| Confirmatory | MAMEM3            |         1 | relative_spectrum               |            40.7 |              34.3 |           -6.4 |            40.7 |              38.9 |                34.1 |                     34.3 |
| Confirmatory | MAMEM3            |        10 | complexity_boosting             |            21.2 |              24.4 |            3.2 |            21.2 |              20.6 |                21.2 |                     22.3 |
| Confirmatory | MAMEM3            |         2 | relative_spectrum               |            46.4 |              36.4 |          -10.0 |            46.4 |              39.0 |                37.6 |                     38.2 |
| Confirmatory | MAMEM3            |         3 | coherence                       |            17.2 |              17.2 |            0.0 |            19.6 |              19.6 |                28.6 |                     25.1 |
| Confirmatory | MAMEM3            |         4 | coherence                       |            26.9 |              22.7 |           -4.2 |            26.9 |              24.4 |                28.4 |                     24.2 |
| Confirmatory | MAMEM3            |         5 | erp_time_course                 |            22.4 |              20.9 |           -1.5 |            22.4 |              26.9 |                26.9 |                     20.9 |
| Confirmatory | MAMEM3            |         6 | shallownet                      |            28.1 |              41.2 |           13.1 |            28.1 |              36.7 |                39.7 |                     33.3 |
| Confirmatory | MAMEM3            |         7 | relative_spectrum               |            12.4 |              29.0 |           16.6 |            12.4 |              25.8 |                20.3 |                     29.7 |
| Confirmatory | MAMEM3            |         8 | green                           |            19.5 |              22.6 |            3.1 |            25.2 |              25.2 |                26.5 |                     17.3 |
| Confirmatory | MAMEM3            |         9 | complexity_boosting             |            25.1 |              28.3 |            3.2 |            25.1 |              30.5 |                34.3 |                     27.5 |
| Confirmatory | Schirrmeister2017 |         1 | covariance_tangent_space        |            88.1 |              85.0 |           -3.1 |            88.1 |              89.4 |                85.0 |                     85.6 |
| Confirmatory | Schirrmeister2017 |        10 | covariance_tangent_space        |            88.1 |              92.5 |            4.4 |            94.4 |              93.8 |                88.8 |                     93.1 |
| Confirmatory | Schirrmeister2017 |        11 | covariance_tangent_space        |            80.0 |              87.5 |            7.5 |            78.8 |              88.8 |                80.0 |                     83.8 |
| Confirmatory | Schirrmeister2017 |        12 | green                           |            88.8 |              94.4 |            5.6 |            92.5 |              94.4 |                93.1 |                     95.0 |
| Confirmatory | Schirrmeister2017 |        13 | green                           |            98.1 |              98.8 |            0.6 |            98.1 |              95.0 |                95.6 |                     95.6 |
| Confirmatory | Schirrmeister2017 |        14 | covariance_tangent_space        |            62.5 |              63.7 |            1.2 |            62.5 |              76.2 |                63.1 |                     84.4 |
| Confirmatory | Schirrmeister2017 |         2 | covariance_tangent_space        |            85.0 |              92.5 |            7.5 |            85.0 |              93.1 |                87.5 |                     93.8 |
| Confirmatory | Schirrmeister2017 |         3 | covariance_tangent_space        |            97.5 |              98.8 |            1.2 |            96.2 |              98.8 |                96.9 |                     97.5 |
| Confirmatory | Schirrmeister2017 |         4 | green                           |            98.8 |              99.4 |            0.6 |            98.8 |              98.8 |                96.2 |                     99.4 |
| Confirmatory | Schirrmeister2017 |         5 | green                           |            97.5 |             100.0 |            2.5 |            97.5 |              98.8 |                97.5 |                     96.2 |
| Confirmatory | Schirrmeister2017 |         6 | green                           |            95.6 |              97.5 |            1.9 |            96.2 |              98.8 |                90.0 |                     96.9 |
| Confirmatory | Schirrmeister2017 |         7 | covariance_tangent_space        |            85.6 |              91.2 |            5.6 |            85.6 |              89.4 |                85.0 |                     91.9 |
| Confirmatory | Schirrmeister2017 |         8 | green                           |            93.1 |              98.1 |            5.0 |            93.1 |              98.8 |                95.6 |                     97.5 |
| Confirmatory | Schirrmeister2017 |         9 | green                           |            89.4 |              92.5 |            3.1 |            89.4 |              96.2 |                88.1 |                     94.4 |
| Confirmatory | Weibo2014         |         1 | covariance_tangent_space_4_30Hz |            62.5 |              72.5 |           10.0 |            63.7 |              67.5 |                71.2 |                     75.0 |
| Confirmatory | Weibo2014         |        10 | covariance_tangent_space        |            63.7 |              63.7 |            0.0 |            63.7 |              67.5 |                65.0 |                     58.8 |
| Confirmatory | Weibo2014         |         2 | csp_lda_8_30Hz                  |            73.8 |              72.5 |           -1.2 |            77.5 |              85.0 |                83.8 |                     76.2 |
| Confirmatory | Weibo2014         |         3 | minirocket                      |            41.2 |              47.5 |            6.2 |            41.2 |              45.0 |                45.0 |                     47.5 |
| Confirmatory | Weibo2014         |         4 | eegnet                          |            57.5 |              52.5 |           -5.0 |            52.5 |              55.0 |                53.8 |                     53.8 |
| Confirmatory | Weibo2014         |         5 | csp_lda_8_30Hz                  |            67.5 |              73.8 |            6.2 |            57.5 |              63.7 |                68.8 |                     72.5 |
| Confirmatory | Weibo2014         |         6 | csp_lda_8_30Hz                  |            87.0 |              90.0 |            2.9 |            83.1 |              87.3 |                83.3 |                     90.0 |
| Confirmatory | Weibo2014         |         7 | green                           |            82.5 |              92.5 |           10.0 |            85.0 |              88.8 |                82.5 |                     92.5 |
| Confirmatory | Weibo2014         |         8 | csp_lda_8_30Hz                  |            95.0 |              96.2 |            1.2 |            82.5 |              78.8 |                71.2 |                     95.0 |
| Confirmatory | Weibo2014         |         9 | csp_lda_8_30Hz                  |            75.0 |              71.2 |           -3.8 |            70.0 |              75.0 |                70.0 |                     70.0 |

## S2. Combination rules (main ensemble, mean %)

|                   |   Confirmatory |   Development |
|:------------------|---------------:|--------------:|
| best_single       |           61.5 |          70.1 |
| majority          |           59.7 |          68.4 |
| soft              |           59.3 |          69.0 |
| weighted          |           63.6 |          72.3 |
| stacking          |           63.6 |          71.7 |
| auto              |           63.9 |          74.2 |
| oracle_best_voter |           65.4 |          74.8 |

## S3. Leave one voter out (main ensemble): change in ensemble accuracy, pp (negative: the voter helps)

| removed                         |   Development |   Confirmatory |
|:--------------------------------|--------------:|---------------:|
| csp_lda_8_30Hz                  |           0.9 |           -0.9 |
| green                           |          -0.6 |           -0.7 |
| shallownet                      |          -0.3 |           -0.6 |
| erp_time_course                 |          -0.0 |           -0.4 |
| covariance_tangent_space_4_30Hz |          -0.2 |           -0.4 |
| eegnet                          |          -1.7 |           -0.4 |
| relative_spectrum               |          -0.4 |           -0.4 |
| covariance_tangent_space        |          -0.1 |           -0.1 |
| minirocket                      |          -0.0 |           -0.0 |
| complexity_boosting             |          -0.2 |           -0.0 |
| riemann_mdm_8_30Hz              |          -0.0 |            0.1 |
| coherence                       |           0.1 |            0.4 |

## S4. Cost of each voter (subject 1 of each dataset, voters run one after another)

Calibration = out-of-fold estimation plus the final fit. Leave-one-out columns: change in
the curated ensemble when the voter is removed (pp; blank for voters not in the curated ensemble).

| voter                           |   calibration median (s) |   calibration max (s) |   latency median (ms) |   latency max (ms) |   curated leave-one-out, Confirmatory |   curated leave-one-out, Development |
|:--------------------------------|-------------------------:|----------------------:|----------------------:|-------------------:|---------------------------------:|--------------------------------:|
| erp_time_course                 |                      0.5 |                   6.0 |                   0.8 |                1.6 |                             -0.2 |                            -0.1 |
| xdawn_tangent                   |                      0.9 |                   5.8 |                   0.7 |                2.1 |                             -0.0 |                             0.0 |
| csp_lda_8_30Hz                  |                      0.9 |                  35.3 |                   0.9 |                2.1 |                            nan   |                           nan   |
| riemann_mdm_8_30Hz              |                      1.3 |                 121.4 |                   1.1 |               15.1 |                            nan   |                           nan   |
| relative_spectrum               |                      1.4 |                  18.6 |                   1.1 |                5.1 |                             -0.1 |                            -0.4 |
| tangent_svm_8_30Hz              |                      1.6 |                 146.0 |                   1.3 |               20.7 |                              0.2 |                             0.0 |
| erp_covariance_mdm              |                      2.3 |                 203.5 |                   0.6 |               16.6 |                             -0.0 |                            -0.2 |
| augmented_covariance            |                      2.7 |                 138.7 |                   1.7 |               20.3 |                             -0.7 |                            -0.5 |
| bandpower_extratrees            |                      3.0 |                  16.0 |                  10.1 |               14.2 |                            nan   |                           nan   |
| covariance_tangent_space_4_30Hz |                      7.9 |                 543.0 |                   4.4 |               68.5 |                            nan   |                           nan   |
| coherence                       |                      8.3 |                1498.4 |                   2.8 |              427.8 |                              0.0 |                            -0.1 |
| covariance_tangent_space        |                     11.9 |                 921.2 |                   6.2 |              100.6 |                             -0.3 |                            -0.5 |
| complexity_boosting             |                     15.3 |                 138.8 |                  15.1 |               44.5 |                              0.3 |                            -0.6 |
| minirocket                      |                     16.4 |                  53.1 |                   5.8 |               15.9 |                              0.2 |                            -0.2 |
| eegnet                          |                     34.8 |                 454.0 |                   1.6 |                2.4 |                             -0.1 |                            -0.8 |
| shallownet                      |                     44.5 |                 353.3 |                   1.4 |                3.6 |                             -0.4 |                             0.1 |
| fbcnet                          |                     61.1 |                 583.7 |                   3.6 |                4.8 |                             -0.5 |                            -0.6 |
| eeginception_erp                |                    106.7 |                1455.4 |                   2.5 |                3.7 |                            nan   |                           nan   |
| green                           |                    158.6 |                1765.8 |                   5.2 |                9.3 |                             -0.2 |                            -0.4 |
| atcnet                          |                    208.6 |                1348.6 |                   6.1 |                9.2 |                              0.1 |                            -0.2 |

## S5. Gain by how far the picked voter is above chance (all subjects)

| bin      |   subjects |   mean_gain |   improved |
|:---------|-----------:|------------:|-----------:|
| < 5 pp   |          9 |         3.4 |          6 |
| 5-15 pp  |         13 |         2.0 |          7 |
| 15-30 pp |         40 |         3.9 |         28 |
| > 30 pp  |         43 |         3.0 |         36 |

## S6. Network tuning (experiment 3; Amendment A2)

Development out-of-fold balanced accuracy (%) of each configuration, mean over subjects then
datasets. The highest per row was selected (ties to the earlier column); test sessions were
not used. Benchmark = 128 Hz, 1–45 Hz (P300 1–24 Hz), AdamW, 300 epochs. Native rate = up to
250 Hz, 0.5 Hz to min(100 Hz, 0.45 × rate). Published training = Adam, batch 64, 1000 epochs.

| paradigm   | network          |   benchmark |   128 Hz, published training |   native rate, benchmark training |   native rate, published training | selected                        |
|:-----------|:-----------------|------------:|-----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------|
| MI         | atcnet           |        63.7 |                         65.9 |                              64.3 |                              67.5 | native rate, published training |
| MI         | eeginception_erp |        60.3 |                         63.7 |                              61.1 |                              64.1 | native rate, published training |
| MI         | eegnet           |        65.4 |                         68.7 |                              60.9 |                              64.7 | 128 Hz, published training      |
| MI         | fbcnet           |        76.3 |                         78.3 |                              76.9 |                              78.6 | native rate, published training |
| MI         | green            |        75.5 |                         76.4 |                              76.6 |                              76.2 | native rate, benchmark training |
| MI         | shallownet       |        69.0 |                         74.2 |                              65.7 |                              70.2 | 128 Hz, published training      |
| P300       | atcnet           |        56.2 |                         56.4 |                              54.6 |                              56.0 | 128 Hz, published training      |
| P300       | eeginception_erp |        79.2 |                         79.2 |                              81.0 |                              81.1 | native rate, published training |
| P300       | eegnet           |        80.0 |                         80.3 |                              82.4 |                              82.4 | native rate, benchmark training |
| P300       | fbcnet           |        58.4 |                         57.3 |                              58.0 |                              56.7 | benchmark                       |
| P300       | green            |        52.6 |                         53.7 |                              51.9 |                              52.3 | 128 Hz, published training      |
| P300       | shallownet       |        59.1 |                         58.8 |                              69.6 |                              71.6 | native rate, published training |
| SSVEP      | atcnet           |        20.6 |                         22.8 |                              21.1 |                              24.0 | native rate, published training |
| SSVEP      | eeginception_erp |        22.8 |                         26.7 |                              20.7 |                              19.5 | 128 Hz, published training      |
| SSVEP      | eegnet           |        21.5 |                         24.1 |                              19.0 |                              21.9 | 128 Hz, published training      |
| SSVEP      | fbcnet           |        43.8 |                         47.8 |                              42.1 |                              46.4 | 128 Hz, published training      |
| SSVEP      | green            |        45.2 |                         46.9 |                              46.1 |                              47.5 | native rate, published training |
| SSVEP      | shallownet       |        38.2 |                         38.7 |                              32.6 |                              31.5 | 128 Hz, published training      |

### Planned comparisons and per-network results (from `tune_networks analyze`)


Per subject balanced accuracy on the test sessions; two-sided Wilcoxon signed-rank;
ensemble rule auto. Main ensemble = the 12 voters of the main ensemble; tuned = its EEGNet, ShallowNet and
GREEN replaced by the selected configuration (selected.json). Comparisons 1-3 are
planned; the network-vs-network rows at the bottom are descriptive.

#### Confirmatory (the test): 53 subjects

| comparison                                           |   mean A |   mean B |   diff (pp) | A better   |      p |
|:-----------------------------------------------------|---------:|---------:|------------:|:-----------|-------:|
| tuned ensemble vs tuned pool: picked                 |     63.8 |     62.4 |         1.4 | 33/53      | 0.0361 |
| tuned ensemble vs eegnet (tuned) alone               |     63.8 |     55.5 |         8.3 | 46/53      | 0      |
| tuned ensemble vs shallownet (tuned) alone           |     63.8 |     57.6 |         6.2 | 36/53      | 0.0002 |
| tuned ensemble vs green (tuned) alone                |     63.8 |     57.1 |         6.8 | 38/53      | 0      |
| tuned ensemble vs eeginception_erp (tuned) alone     |     63.8 |     55.9 |         7.9 | 44/53      | 0      |
| tuned ensemble vs fbcnet (tuned) alone               |     63.8 |     59.1 |         4.7 | 38/53      | 0.0041 |
| tuned ensemble vs atcnet (tuned) alone               |     63.8 |     53.4 |        10.4 | 44/53      | 0      |
| tuned ensemble vs paper ensemble                     |     63.8 |     63.9 |        -0.1 | 22/53      | 0.4324 |
| paper ensemble vs paper pool: picked                 |     63.9 |     61.5 |         2.4 | 36/53      | 0.0016 |
| eegnet (tuned) alone vs eegnet (paper) alone         |     55.5 |     54.7 |         0.9 | 31/53      | 0.1345 |
| shallownet (tuned) alone vs shallownet (paper) alone |     57.6 |     55.1 |         2.5 | 35/53      | 0.0004 |
| green (tuned) alone vs green (paper) alone           |     57.1 |     56.5 |         0.5 | 23/53      | 0.9795 |
| fbcnet (tuned) alone vs fbcnet (paper) alone         |     59.1 |     57   |         2.1 | 27/53      | 0.0064 |
| atcnet (tuned) alone vs atcnet (paper) alone         |     53.4 |     52.1 |         1.3 | 30/53      | 0.1534 |

Per dataset (mean %):

| dataset           |   tuned ensemble |   paper ensemble |   tuned pool: picked |   atcnet (tuned) alone |
|:------------------|-----------------:|-----------------:|---------------------:|-----------------------:|
| BNCI2015_003      |             75   |             75.2 |                 73.4 |                   51.3 |
| BNCI2015_004      |             35.9 |             36.9 |                 36.5 |                   28.6 |
| MAMEM3            |             27.3 |             27.7 |                 27.4 |                   21.2 |
| Schirrmeister2017 |             93.2 |             92.3 |                 89.8 |                   88.6 |
| Weibo2014         |             73.1 |             73.2 |                 71.6 |                   60.8 |

Voter picked on training data (tuned pool): green_tuned 13, erp_time_course 7, covariance_tangent_space 7, csp_lda_8_30Hz 6, shallownet_tuned 5, eegnet_tuned 5, complexity_boosting 3, covariance_tangent_space_4_30Hz 3, relative_spectrum 2, coherence 1, minirocket 1

#### Development (configurations chosen here): 52 subjects

| comparison                                                       |   mean A |   mean B |   diff (pp) | A better   |      p |
|:-----------------------------------------------------------------|---------:|---------:|------------:|:-----------|-------:|
| tuned ensemble vs tuned pool: picked                             |     74.4 |     70.7 |         3.7 | 43/52      | 0      |
| tuned ensemble vs eegnet (tuned) alone                           |     74.4 |     65.3 |         9.1 | 46/52      | 0      |
| tuned ensemble vs shallownet (tuned) alone                       |     74.4 |     68.1 |         6.3 | 43/52      | 0      |
| tuned ensemble vs green (tuned) alone                            |     74.4 |     64.3 |        10.1 | 41/52      | 0      |
| tuned ensemble vs eeginception_erp (tuned) alone                 |     74.4 |     61   |        13.4 | 44/52      | 0      |
| tuned ensemble vs fbcnet (tuned) alone                           |     74.4 |     69.4 |         5   | 37/52      | 0.0016 |
| tuned ensemble vs atcnet (tuned) alone                           |     74.4 |     59.2 |        15.1 | 46/52      | 0      |
| tuned ensemble vs paper ensemble                                 |     74.4 |     74.2 |         0.1 | 21/52      | 0.7883 |
| paper ensemble vs paper pool: picked                             |     74.2 |     70.1 |         4.1 | 41/52      | 0      |
| eegnet (tuned) alone vs eegnet (paper) alone                     |     65.3 |     62   |         3.3 | 37/52      | 0      |
| shallownet (tuned) alone vs shallownet (paper) alone             |     68.1 |     62.3 |         5.7 | 43/52      | 0      |
| green (tuned) alone vs green (paper) alone                       |     64.3 |     64.4 |        -0.1 | 24/52      | 0.939  |
| eeginception_erp (tuned) alone vs eeginception_erp (paper) alone |     61   |     58.8 |         2.2 | 33/52      | 0.0146 |
| fbcnet (tuned) alone vs fbcnet (paper) alone                     |     69.4 |     67.7 |         1.7 | 31/52      | 0.0001 |
| atcnet (tuned) alone vs atcnet (paper) alone                     |     59.2 |     55.3 |         3.9 | 31/52      | 0.0227 |

Per dataset (mean %):

| dataset      |   tuned ensemble |   paper ensemble |   tuned pool: picked |   atcnet (tuned) alone |
|:-------------|-----------------:|-----------------:|---------------------:|-----------------------:|
| BNCI2014_001 |             76.5 |             76.5 |                 70.2 |                   68.2 |
| BNCI2014_004 |             73.4 |             74.4 |                 73.1 |                   65.5 |
| BNCI2014_009 |             84.8 |             84.3 |                 81.6 |                   57.6 |
| BNCI2015_001 |             85.1 |             84.6 |                 77.7 |                   71.2 |
| Kalunga2016  |             54.2 |             53.6 |                 53.2 |                   37.2 |

Voter picked on training data (tuned pool): covariance_tangent_space 12, green_tuned 12, eegnet_tuned 9, shallownet_tuned 5, csp_lda_8_30Hz 4, relative_spectrum 3, covariance_tangent_space_4_30Hz 3, erp_time_course 2, riemann_mdm_8_30Hz 1, coherence 1


## S7. Paradigm-standard voters (experiment 4; Amendment A3)

Cost of each standard voter, every subject of its paradigm's datasets:

| voter     |   subjects |   calibration median (s) |   calibration max (s) |   latency median (ms) |   latency max (ms) |
|:----------|-----------:|-------------------------:|----------------------:|----------------------:|-------------------:|
| cca       |       22.0 |                      0.0 |                   0.1 |                   0.3 |                0.5 |
| fbcsp     |       63.0 |                     21.1 |                 645.9 |                   9.4 |               16.3 |
| trca      |       22.0 |                      0.0 |                   0.1 |                   0.3 |                0.4 |
| xdawn_lda |       20.0 |                      0.2 |                   0.4 |                   0.1 |                0.2 |


Main + standard pool = the 12 voters of the main ensemble plus FBCSP (MI), xDAWN + LDA (P300) or CCA and TRCA (SSVEP).
Balanced accuracy on the test sessions; ensemble rule auto. First row: H6, one-sided
Wilcoxon. Other rows: two-sided Wilcoxon, Holm-adjusted over those rows.

#### Confirmatory (the test): 53 subjects

| comparison                         |   n |   mean A |   mean B |   diff (pp) | A better   |      p |   Holm p |
|:-----------------------------------|----:|---------:|---------:|------------:|:-----------|-------:|---------:|
| main + standard ensemble vs main + standard pool: picked |  53 |     67.9 |     66.9 |         1.1 | 36/53      | 0.039  | nan      |
| main + standard ensemble vs main ensemble       |  53 |     67.9 |     63.9 |         4.1 | 30/53      | 0.0099 |   0.0298 |
| main + standard ensemble vs fbcsp alone       |  33 |     71.7 |     57.4 |        14.2 | 30/33      | 0      |   0      |
| main + standard ensemble vs xdawn_lda alone   |  10 |     76.6 |     74.6 |         2   | 7/10       | 0.2324 |   0.3203 |
| main + standard ensemble vs cca alone         |  10 |     47   |     52.1 |        -5   | 4/10       | 0.1602 |   0.3203 |
| main + standard ensemble vs trca alone        |  10 |     47   |     20.1 |        27   | 9/10       | 0.0059 |   0.0234 |

Per dataset (mean %):

| dataset           |   main + standard ensemble |   main ensemble |   main + standard pool: picked |   main pool: picked |   cca alone |   trca alone |   fbcsp alone |   xdawn_lda alone |
|:------------------|----------------:|--------------:|--------------------:|------------------:|------------:|-------------:|--------------:|------------------:|
| BNCI2015_003      |            76.6 |          75.2 |                75.4 |              73.4 |       nan   |        nan   |         nan   |              74.6 |
| BNCI2015_004      |            37.2 |          36.9 |                34.5 |              34.5 |       nan   |        nan   |          27.1 |             nan   |
| MAMEM3            |            47   |          27.7 |                50.7 |              26   |        52.1 |         20.1 |         nan   |             nan   |
| Schirrmeister2017 |            92.1 |          92.3 |                89.2 |              89.2 |       nan   |        nan   |          71.7 |             nan   |
| Weibo2014         |            74.1 |          73.2 |                72.5 |              70.6 |       nan   |        nan   |          64.8 |             nan   |

Voter picked on training data (main + standard pool): green 10, cca 9, covariance_tangent_space 9, xdawn_lda 6, csp_lda_8_30Hz 4, erp_time_course 4, covariance_tangent_space_4_30Hz 3, relative_spectrum 2, shallownet 2, fbcsp 1, minirocket 1, eegnet 1, complexity_boosting 1

#### Development: 52 subjects

| comparison                         |   n |   mean A |   mean B |   diff (pp) | A better   |      p |   Holm p |
|:-----------------------------------|----:|---------:|---------:|------------:|:-----------|-------:|---------:|
| main + standard ensemble vs main + standard pool: picked |  52 |     76.8 |     74.2 |         2.7 | 40/52      | 0.0008 | nan      |
| main + standard ensemble vs main ensemble       |  52 |     76.8 |     74.2 |         2.6 | 33/52      | 0.0018 |   0.0054 |
| main + standard ensemble vs fbcsp alone       |  30 |     79.3 |     68.4 |        10.9 | 24/30      | 0      |   0.0001 |
| main + standard ensemble vs xdawn_lda alone   |  10 |     84.5 |     75   |         9.5 | 10/10      | 0.002  |   0.0054 |
| main + standard ensemble vs cca alone         |  12 |     64.2 |     69.1 |        -4.9 | 4/12       | 0.2925 |   0.2925 |
| main + standard ensemble vs trca alone        |  12 |     64.2 |     27.2 |        37   | 12/12      | 0.0005 |   0.002  |

Per dataset (mean %):

| dataset      |   main + standard ensemble |   main ensemble |   main + standard pool: picked |   main pool: picked |   cca alone |   trca alone |   fbcsp alone |   xdawn_lda alone |
|:-------------|----------------:|--------------:|--------------------:|------------------:|------------:|-------------:|--------------:|------------------:|
| BNCI2014_001 |            76.4 |          76.5 |                70.9 |              72   |       nan   |        nan   |          57.8 |               nan |
| BNCI2014_004 |            74.6 |          74.4 |                70.6 |              70.6 |       nan   |        nan   |          68.9 |               nan |
| BNCI2014_009 |            84.5 |          84.3 |                80.7 |              79.8 |       nan   |        nan   |         nan   |                75 |
| BNCI2015_001 |            85   |          84.6 |                78.3 |              78.3 |       nan   |        nan   |          76   |               nan |
| Kalunga2016  |            64.2 |          53.6 |                69.7 |              52   |        69.1 |         27.2 |         nan   |               nan |

Voter picked on training data (main + standard pool): green 13, cca 10, covariance_tangent_space_4_30Hz 6, covariance_tangent_space 5, eegnet 5, csp_lda_8_30Hz 3, xdawn_lda 3, erp_time_course 2, complexity_boosting 1, relative_spectrum 1, fbcsp 1, shallownet 1, coherence 1
