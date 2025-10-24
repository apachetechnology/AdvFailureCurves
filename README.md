# Adversarial Failure Curves

| Francesco Bergadano | Sandeep Gupta | Bruno Crispo |
|----------|----------|----------|
| Department of Computer Science,| Centre for Secure Information Technologies (CSIT), | Department of Information Engineering and Computer Science,|
| University of Turin, Italy| Queen's University Belfast, UK | University of Trento, Italy |
    
    https://anonymous.4open.science/r/AdvFailureCurves-EAF8

# Usage details 

## Main Experiments
    Run nbMainExperiments.ipynb

## Plot Result
    Run nbPlotResults.ipynb

## Generate Synthetic Data
    Run nbGenSynData.ipynb 

## Generate Adversarial Data
    Run nbAdvExData.ipynb 

## Models performance
    Run nbTestClassification.ipynb

# Citation

Please cite our papers if you use this research or the code

[Keyed randomization with adversarial failure curves and moving target defense](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11140525)
@inproceedings{bergadano2025keyed,
  title={Keyed randomization with adversarial failure curves and moving target defense},
  author={Bergadano, Francesco and Gupta, Sandeep and Crispo, Bruno},
  booktitle={Proceedings of the 5th Intelligent Cybersecurity Conference (ICSC)},
  pages={169--176},
  year={2025},
  organization={IEEE}
}

## Evasion resistance via diversity prediction
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


