# ECE7123MiniProject1



## Dependencies
   1. Python 3.6.9
   2. torch 1.10.1
   3. torchvision  0.11.2

##  Model
   1. Download the trained ResNet model from [here](https://drive.google.com/drive/folders/1WZ0lABx2XZD8ZvsparsjHWLu7tKxYu63?usp=sharing)
and store them under `./pt` directory.
   2. You can use the model to test the inference accuracy.
      - We save our model by `torch.save(net.state_dict(), './project1_model.pt')`.  `test_model.py` shows how to load the model for inference.

## Usage
1. To train the model, run:
```bash
python3 project1_model.py --lr <learning rate> --epoch <epoch>  --model <model>
```

2. Below shows how to train the five ResNet models mentioned in our paper:
```bash
python3 project1_model.py --model 1 # trains ResNet-9 in 200 epoch and learning rate of 0.1 (default)
python3 project1_model.py --model 2 # trains ResNet-9-Dropout in 200 epoch and learning rate of 0.1
python3 project1_model.py --model 3 # trains ResNet-14-CIFAR-LIGHT
python3 project1_model.py --model 4 # trains ResNet-14-CL-1
python3 project1_model.py --model 5 # trains ResNet-14-CL-2


```
Without specifying <tt>epoch</tt> and <tt>lr</tt>, the script will use default values for training (<tt>epoch = 200, lr = 0.1</tt>). 
You can also add or change the parameters as Mini Project 1 required in 
the parameter list of <tt>BuildBasicModelWithParameter()</tt> functions inside <tt>BuildNet()</tt>. 


