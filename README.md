# AFBT-GAN
Code for AFBT GAN: enhanced explainability and diagnostic performance for cognitive decline by counterfactual generative adversarial network

## __Environment__  
```diff  
torch ==1.13.1
numpy == 1.22.3  
nibabel == 1.10.2  
torchcam == 0.3.2  
torchvision == 0.14.1  
einops == 0.6.0  
python == 3.9.0  
imageio == 2.31.1
``` 
## extract the imaging features
To run the model, you need to extract the dfc by Matlab. You can use the batch operation of spm12 to finish this. Additionally, the input data shape might influence the kernel size of avgpooling in ResNet, you need to change the kernel size, if has bugs.  

## run the model
### __1. Create k fold csv file__  
```diff
generate_csv.py
```
### __2. pretrain classifier model__  
set the mode_net as pretrained classifier  opt.py
```diff
run
main.py
```
### __3. get counterfactual attention__  
set the mode_net as image_generator in opt.py
```diff
run
main.py
```
### __4.train final classifier__  
set the mode_net as region-specific in opt.py
```diff
run
main.py
```
