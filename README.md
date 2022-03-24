# ECE7123MiniProject1

```bash
ECE7123MiniProject1
├── basecode
│   ├── project1_model_cifar.py
│   └── project1_model.py
├── pt
│   └── project1_model.pt
└── README.md
```


## Dependencies
   1. Python 3.6.9
   2. torch 1.10.1
   3. torchvision  0.11.2

##  Model
   1. Download the ResNet model from [here XXX](https://drive.google.com/???)
and store them under `./pt` directory.
   2. You can use the model to test the inference accuracy.
      - We save our model by `torch.save(net, 'net_model.pt')`. You can load the 
      model by `net_loaded = torch.load('net_model.py')` and evaluate it.

## Train
1. To train the model, execute `project1_model.py` by running:  
      `python3 project1_model.py --lr <your learning rate> --epoch <#>`.

You can also add / change the parameters as Mini Project 1 required in 
the parameter list of `def BuildBasicModelWithParameter()`. 

2. We generate our best performance model by running `Todo XXX`. Model saved to the link above. 
   - You can train the **light** model "ResNet-14-CIFAR-Light" by running `Todo XXX`.

3. If you want to plot the performance, follow these steps: XXX

