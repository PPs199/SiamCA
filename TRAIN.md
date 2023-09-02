# SiamCA Training Tutorial

This implements training of siamca.
### Add siamca to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/siamca:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.


* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT10K](http://got-10k.aitestunion.com/)
* [LASOT](https://cis.temple.edu/lasot/)

## Download pretrained backbones
Download pretrained backbones from [here](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `pretrained_models` directory

## Training

To train a model, run `train.py` with the desired configs:

```bash
cd experiments/siamca_r50_l234
```

### Multi-processing Distributed Data Parallel Training

Refer to [Pytorch distributed training](https://pytorch.org/docs/stable/distributed.html) for detailed description.

#### Single node(We use 1 GPUs):torch.distributed.launch 
```bash
CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../../tools/train.py --cfg config.yaml
```

## Testing
After training, you can test snapshots on VOT dataset.
For example, you need to test snapshots from 10 to 20 epoch.

```bash 
START=10
END=20
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} \ 
    python -u ../../tools/test.py \
        --snapshot {} \
	--config config.yaml \
	--dataset VOT2018 2>&1 | tee logs/test_dataset.log
```

Or:

```bash
mpiexec -n 1 python ../../tools/test_epochs.py  \
    --start_epoch 10  \
    --end_epoch 20  \
    --gpu_nums 3  \
    --threads 1  \
    --dataset VOT2018
```

## Evaluation

```bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 4 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```

## Hyper-parameter Search

The tuning toolkit will not stop unless you do.

```bash
python ../../tools/tune.py  \
    --dataset VOT2018  \
    --snapshot snapshot/checkpoint_e20.pth  \
    --gpu_id 0
```

