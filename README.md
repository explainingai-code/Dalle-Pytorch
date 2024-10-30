DallE Implementation in pytorch with generation using mingpt
========

This repository implements DallE-1 [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) on a synthetic dataset of mnist colored numbers on textures/solid background .


## DallE Tutorial Video
<a href="https://www.youtube.com/watch?v=wX5LP8n9WAQ">
   <img alt="DallE Tutorial" src="https://github.com/explainingai-code/Dalle-Pytorch/assets/144267687/5a3af0f0-6b9c-48bc-becb-1d06e6095ada"
   width="300">
</a>

## Sample from dataset

<img src="https://github.com/explainingai-code/DallE/assets/144267687/57e3c091-4600-401d-a5a4-52ea5fda3249" width="300">



A lot of parts of the implementation have been taken from below two repositories:
1. GPT from - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
2. Parts of DallE implementation from https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch . 

   I have only kept the minimal version of Dalle which allows us to get decent results(on this dataset) and play around with it. If you are looking for a much more efficient and complete implementation please use the above repo.

## Data preparation
For setting up the mnist dataset:
Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Download Quarter RGB resolution texture data from [ALOT Homepage](https://aloi.science.uva.nl/public_alot/)

If you are facing issues then use `curl`

`curl -O https://aloi.science.uva.nl/public_alot/tars/alot_png4.tar`


In case you want to train on higher resolution, you can download that as well and but you would have to create new train.json and test.json.
Rest of the code should work fine as long as you create valid json files.

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


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training of Discrete VAE and DallE the following output will be saved 
* Best Model checkpoints(DVAE and DallE) in ```task_name``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/dvae_reconstruction.png``` 
* GPT generation output in  ```task_name/generation_results.png```


## Sample Output for DallE

Running default config DiscreteVAE should give you below reconstructions (left - input | right - reconstruction)

<img src="https://github.com/explainingai-code/DallE/assets/144267687/fccf876d-fb35-4ed5-9729-b3645690370e" width="300">

Sample Generation Output after 40 epochs with 4 layers and 512 hidden dimension and 8 attention heads 

Generate 0 in blue and solid background of olive

Generate 1 in cyan and texture background of cracker

Generate 6 in pink and texture background of stones

Generate 8 in red and texture background of lego

<img src="https://github.com/explainingai-code/DallE/assets/144267687/e5cbc440-9a07-4439-96b8-d9a446e8b293" width="50">
<img src="https://github.com/explainingai-code/DallE/assets/144267687/a6f41119-8cfd-4536-8267-8d05b3a6154f" width="50">
<img src="https://github.com/explainingai-code/DallE/assets/144267687/7bdd3e44-5c3d-46e4-aaaa-67f08a6c6591" width="50">
<img src="https://github.com/explainingai-code/DallE/assets/144267687/6aac0a2e-4264-4691-990c-0f93156ddb7d" width="50">



## Citations

```
@misc{ramesh2021zeroshot,
      title={Zero-Shot Text-to-Image Generation}, 
      author={Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
      year={2021},
      eprint={2102.12092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


