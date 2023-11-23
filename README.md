# Voting system EEG Data Classification

All the codes here are from other contributors:
* CCSPNet -> Only runs for pairs. It is still useful, but multiple trains will be done.
* pyRiemann
* XDAWN -> Good for 4 classes
* TCACNet -> https://github.com/LiuXiaolin-lxl/TCACNet

Abandonded:
* BigProject 
* LMDA-Net -> It didn't run
* EEGNET and cousins
* LSTM, at least that project in particular, you could try with other LSTM libraries. It gave bad results, but maybe other will be better.

They were modified to accomodate the needs for the project.

#  The Datasets are provided by:
* Aguilera
* Nieto
* Coretto
* Torres

#TODO:
* Install the minimum amount of dependencies to make sure those are the ones that the project really needs. You need to go to requirements.txt and check none are missing or are extra.
* Cite code
* Cite datasets and mention why they were chosen
* Try to combine the XDWAN with Riemman
* Standardize the results from all the methods
* Call all the methods from one file
* Run LMDA on Coretto and try Aguilera again. Or maybe just give up on LMDA.