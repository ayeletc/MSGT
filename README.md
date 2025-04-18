# Non-Adaptive Multi-Stage Group Testing Algorithm

This repository provides an open-source implementation of the algorithm presented in the paper:

**Non-Adaptive Multi-Stage Algorithm for Group Testing with Prior Statistics**\
Ayelet C. Portnoy, Alejandro Cohen\
[[https://ieeexplore.ieee.org/abstract/document/10735310](https://ieeexplore.ieee.org/abstract/document/10735310)]

## Overview

The algorithm addresses the problem of non-adaptive group testing with general prior statistics. The proposed **Multi-Stage Group Testing (MSGT)** algorithm leverages prior statistics to improve testing efficiency. The key contributions of the algorithm include:

- **Using the List Viterbi Algorithm (LVA):** Adapting LVA for accurate defective item identification with fewer tests.
- **Applicable to General Prior Statistics:** The algorithm works with any prior statistics represented by a Trellis, such as finite-state machines and Markov processes.
- **Reducing the Number of Tests:** The algorithm reduces the required number of tests by at least 25% compared to existing methods in tested scenarios.

## Usage

**Requirements:** Python 3.6 or higher

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ayeletc/GT_with_priors.git
cd GT_with_priors
pip install -r requirements.txt
```

### Running the Algorithm

To run the algorithm, first set the desired parameters in config.yaml and then use the following command:

```bash
python multi_stage_algo.py
```

### Results

The general section in the YAML controls whether the results are saved at the end of the run as a `.mat` file in the results directory or displayed as plots.

```yaml
save_raw: true     
save_fig: false    
plot_res: false    
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{portnoy2024non,
  title={Non-Adaptive Multi-Stage Algorithm for Group Testing with Prior Statistics},
  author={Portnoy, Ayelet C and Cohen, Alejandro},
  booktitle={2024 60th Annual Allerton Conference on Communication, Control, and Computing},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
```

##

