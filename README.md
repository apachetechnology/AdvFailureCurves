# Adversarial Failure Curves

## Researcher

| Francesco Bergadano | Sandeep Gupta | Bruno Crispo |
|----------|----------|----------|
| Department of Computer Science,| Centre for Secure Information Technologies (CSIT), | Department of Information Engineering and Computer Science,|
| University of Turin, Italy| Queen's University Belfast, UK | University of Trento, Italy |
    
    https://anonymous.4open.science/r/AdvFailureCurves-EAF8

# Usage details 

## Data
    Download data.tar and keep at the outside the main folder. <br>
    All the path are relative ../DATA.
    Please check the datapath before running the experiments

## Main Experiments
    Run nbMainExperiments.ipynb

## Plot Result
    Run nbPlotResults.ipynb

## Generate Synthetic Data
    Run nbGenSynData.ipynb 

## Generate Adversarial Data
    Run ./AdversarialTraining/nbAdvExData.ipynb 

## Models performance
    Run nbTestClassification.ipynb

# Citation

Please cite our papers if you use this research or the code

[Keyed randomization with adversarial failure curves and moving target defense](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11140525)

## Keyed randomization with adversarial failure curves and moving target defense
@inproceedings{bergadano2025keyed,<br>
  title={Keyed randomization with adversarial failure curves and moving target defense},<br>
  author={Bergadano, Francesco and Gupta, Sandeep and Crispo, Bruno},<br>
  booktitle={Proceedings of the 5th Intelligent Cybersecurity Conference (ICSC)},<br>
  pages={169--176},<br>
  year={2025},<br>
  organization={IEEE}<br>
}

The research paper received the Best Paper Award.
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/4bd4f0d1-6836-447f-8fab-982073ea798d" />


## Evasion resistance via diversity prediction
[Evasion resistance via diversity prediction](https://pure.qub.ac.uk/en/publications/evasion-resistance-via-diversity-prediction/)

@inproceedings{bergadano2025evasion,<br>
  title={Evasion resistance via diversity prediction},<br>
  author={Bergadano, Francesco and Gupta, Sandeep and Crispo, Bruno},<br>
  booktitle={Proceedings of the 1st Iberian Conference on Cybersecurity and Artificial Intelligence},<br>
  pages={5--8},<br>
  year={2025},<br>
  organization={Universidade da Beira Interior}<br>
}

# Requirements
## CTGAN
py -3.9 -m pip install ctgan<br>
py -3.9 -m pip show ctgan

## ART
py -3.9 -m pip install scikit-learn adversarial-robustness-toolbox<br>
py -3.9 -m pip show scikit-learn<br>
py -3.9 -m pip install --upgrade adversarial-robustness-toolbox

## Required package
py -3.9 -m pip install numba<br>
py -3.9 -m pip install rtd


