# Pawsey19

## All_Data 

Contains the data that was used to develop the Digital Twin. In order to put into the form expected by the Neural Network programs, run the program `clean.py`, which will unpack, clean and manipulate the data into the required form and produce the csv `FinalData.csv`. 

## ml_code

Contains the code that was used to devleop the Digital Twin. The RBF layer was built using Petra Vidnerova's implementation as a starting point (https://github.com/PetraVidnerova/rbf_for_tf2) and has two main programs: `rbfmodel.py` will run the RBF Neural Network and `dnnmodel.py` will run the DNN neural Network.
