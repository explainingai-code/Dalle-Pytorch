DallE Implementation in pytorch with generation using mingpt
========

This repository implements DallE-1 on a synthetic dataset of mnist colored numbers on textures/solid background .
[Video on DallE](https://www.youtube.com/watch?v=wX5LP8n9WAQ)

Sample from dataset
<img src="https://github.com/explainingai-code/DallE/assets/144267687/57e3c091-4600-401d-a5a4-52ea5fda3249" width="300">



A lot of parts of the implementation have been taken from below two repositories:
1. GPT from - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
2. Parts of DallE implementation from https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch . 

   I have only kept the minimal version of Dalle which allows us to get decent results(on this dataset) and play around with it. If you are looking for a much more efficient and complete implementation please use this repo.
   

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/DallE.git```
* ```cd DallE```
* ```pip install -r requirements.txt```
* For training/inferencing discrete vae and gpt use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train_dvae``` for training discrete vae
* ```python -m tools.infer_dvae``` for generating reconstructions
* ```python -m tools.train_dalle``` for training minimal version of DallE 
* ```python -m tools.generate_image``` for using the trained DallE to generate images

## Configuration
* ```config/default.yaml``` - Allows you to play with different components of discrete vae as well as DallE and play around with these modifications 

## Data preparation
For setting up the mnist dataset:
Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Download Quarter RGB resolution texture data from [ALOT Homepage](https://aloi.science.uva.nl/public_alot/)
In case you want to train on higher resolution, you can download that as well and code should work for that also

Download train.json and test.json from [Drive](https://drive.google.com/drive/folders/1DSpNaM6hk8VNFVKHs-VK97AlP_8ynRKC?usp=sharing)
Verify the data directory has the following structure after textures download
```
DallE/data/textures/{texture_number}
	*.png
DallE/data/train/images/{0/1/.../9}
	*.png
DallE/data/test/images/{0/1/.../9}
	*.png
DallE/data/train.json
DallE/data/test.json
```

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training of Discrete VAE and DallE the following output will be saved 
* Best Model checkpoints(DVAE and DallE) in ```task_name``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/dvae_reconstruction.png``` 
* GPT generation output in  ```task_name/generation_results.png```


## Sample Output for DallE

Running default config Discrete should give you below reconstructions for both versions
![dvae_reconstructions](https://github.com/explainingai-code/DallE/assets/144267687/fccf876d-fb35-4ed5-9729-b3645690370e)

Sample Generation Output after 40 epochs with 4 layers and 512 hidden dimension and 8 attention heads 

