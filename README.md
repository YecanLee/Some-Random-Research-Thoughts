Some codes/libraries/bolierplates have been seen for many times during my research time, I want to write them down for a long time as some nice resources to be used in the future.

## Import tricks
Some packages can be overlapped with other packages, this means that we are using more memories and making the codes harder to be maintained, to fix this issue, use the following boilerplate codes instead, this will automatically import the correct dependencies in the envs and log all the exceptions.    

```bash
# try import one of the packages
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError as e:
    logger.warning(f"The following packages are missing from the env: {e}")
```

It is also good to offer a fallback implementation.

```bash
try: from torch.nn import MultiheadAttention as slowattn
except ImportError:
    # define the fallback implementation the same name as the import one
    def slowattn(q, k, v):
        pass
```
## Always use a logger
Use a logger to log the errors/informations needed later for better experiments design/debug purpose

```bash
import logging
logger = logging.getLogger(__name__)
```

# Check the I/O bottleneck, this is very important for large experiments
Some people, as Stefan said, like myself, did not study CS before, so there codes has many bottlenecks inside, one of the bottleneck which can be easily fixed is the `DataLoader` from the `torch` package.   
```bash
# ...other import
# we assume that the data is downloaed at the local folder already

from torch.utils.data import DataLoader
train_dataset = DataLoader(dataset_folder, num_workers=4, ...)
```

The I/O bottleneck often happened in the "num_workers" part. Especially during parallel training, if we use the default value for `num_workers=0`, the data will only be loaded in the main process, making the whole training super slow.         
To solve this issue, simply use `num_workers=4` or `num_workers=8`. If we have multiple GPUs training in parallel, we should try with `4*GPU numbers` as the starting point first. Then we can test the speed by scaling up the `num_workers` number until no improvement can be noticed.   

There are other things mentioned in the `Lighting AI`'s [speed up model training](https://lightning.ai/docs/pytorch/stable/advanced/speed.html), two things deserved to be mentioned:   
1. Use `pin_memory=True` if we are using `num_workers` larger than 0 for training, this will pin the different workers to a fixed memory, instead of rerandom at the beggining of each epoch.   
2. Use `persistent_workers=True` if we are using large number of `num_workers`, this will speed up the training at the beginning of every epoch.
