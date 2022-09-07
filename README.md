# Open-world GAN Discovery
PyTorch implementation of [Towards Discovery and Attribution of Open-world GAN Generated Images](https://arxiv.org/abs/2105.04580)
```bash
@inproceedings{girish2021towards,
  title={Towards discovery and attribution of open-world gan generated images},
  author={Girish, Sharath and Suri, Saksham and Rambhatla, Sai Saketh and Shrivastava, Abhinav},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14094--14103},
  year={2021}
}
```

## Installation
Create new virtualenv/conda environment and run the following command:
```
pip install -r requirements.txt
```

## Data
Setup data for train and eval in Pytorch's ImageFolder format:
```
data/train/real_celeba/xxx.png
data/train/attgan_celeba/xxy.png

data/eval/real_celeba/yyy.png
data/eval/began_celeba/yyy.png
```

## Example train commands
Hyperparameters are defined in yaml files in the [cfgs](cfgs/) folder. An example run command with the default config would look like:
```
python main.py --config cfgs/config.yaml
```
Hyperparameters can additionally be overriden with command-line arguments which are dot separated. For e.g. running the pipeline for 3 iterations instead of 4 would look like:
```
python main.py --config cfgs/config.yaml --common.num_iters 3
```

## Evaluation
Clusters can be evaluated at every stage of the pipeline for a given run as follows:
```
python eval_run.py --config cfgs/config.yaml
```

## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
