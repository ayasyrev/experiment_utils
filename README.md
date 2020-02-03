# Experiment_utils
> Helper utils for track and manage Dl experimets with pytorch.


Very early stage - just draft for my utils.

## Install

`pip install experiment-utils`

`Editeble install: 
git clone https://github.com/ayasyrev/experiment-utils  
cd experiment-utils  
pip install -e .  `

## How to use

Import Experiment:

```python
from experiment_utils.experiment import *
```

After import you has p (stands for Parameters) and e (Experiment) objects.

Name the experiment, later it will be used in logs.

```python
p.exp_name = 'test1'
p.exp_name
```




    'test1'



```python
e.p.exp_name
```




    'test1'



Load learner

```python
e.get_learner()
```

```python
e.p.model
```




    functools.partial(<function resnet18 at 0x7fc8d4dd08c0>, num_classes=10)



```python
e.learn.model.fc
```




    Linear(in_features=512, out_features=10, bias=True)



Short notation for learn - l

```python
e.l.model.conv1
```




    Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)



Now we can easy change some parameter anr start train with pipeline what yuo neg.

`from experiment_utils.utils import train_fc, plot`

`p.pipeline = [train_fc, plot]`

`p.lr = 0.001`

`p.epochs = 10`

`e(repeat_times=2)`
