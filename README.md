# Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation

Initial code release for the paper:

**Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation** (ICLR 2025)

Xinpeng Wang, Chengzhi Hu, Paul Rötter and Barbara Plank. 

This code is build on top of the code from the work [Refusal in Language Models Is Mediated by a Single Direction](https://github.com/andyrdt/refusal_direction).

## Environment Setup
```bash
source setup.sh
```
Install the evaluation harness from source

```bash
cd lm-evaluation-harness
pip install -e .
``` 


## Experiments
To run vector extraction, ablation and evaluation, run the script bellow:

```bash
python -m pipeline.run_pipeline --config_path configs/cfg.yaml
```

## Demo 
We also provide a demo notebook [here](demo.ipynb). 

## Cite
```
@inproceedings{wang2025surgical,
    title={Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation},
    author={Xinpeng Wang and Chengzhi Hu and Paul R{\"o}ttger and Barbara Plank},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=SCBn8MCLwc}
}
```