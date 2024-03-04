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
To run the model, you need to extract the dfc by Matlab. You can use the batch operation of spm12 to finish this. Additionally, the input data shape might influence the kernel size of avgpooling, you need to change the kernel size, if has bugs.  

## run the model
### train and validate the model 
```diff
main.py
```
### __Create k fold csv file__  
```diff
generate_csv.py
```
