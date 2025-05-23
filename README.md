# NPFTaint： Detecting highly exploitable vulnerabilities in Linux-based IoT firmware with network parsing functions

NPFTaint achieves the accurate identification of network parsing functions through static analysis by extracting structural, behavioral, and semantic features of functions, generating feature vectors, and leveraging a supervised machine learning model. It then uses the breadth-first search algorithm to collect sensitive sinks in the call chain of network parsing functions. Finally, starting from the sensitive sinks in the call chains, NPFTaint performs data dependency analysis from sink to source, effectively achieving the detection goal of highly exploitable vulnerabilities.

## Repository Structure

There are seven main directories:

- **BinarySet**: Dataset
- **Figure**: Analysis and results presentation
- **NPFtaint**: Identify the results of the potential network parsing functions
- **feature vector**: The results of generating the feature vectors after feature extraction and the predicted results
- **feature_extractor**: The results of feature extraction for binaries
- **model**: Network parsing function finder
- **cmdi**: Obtain the detection results of command injection vulnerabilities with the help of Mango
- **overflow**: Obtain the detection results of buffer overflow vulnerabilities with the help of Mango
- **Vulnerability Database**: The collected results of firmware vulnerabilities

### Key Files

- **functionFinder.py**: It is used for the training of the model
- **binary_finder.py**: The method for collecting the executable binaries in the unpacked firmware
- **prediction.py**: The prediction method for determining whether the function is a network parsing function
