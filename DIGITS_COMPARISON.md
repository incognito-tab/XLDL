# CPU comparison on Digits

Both runs use the 1,797 real handwritten images from [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). `digits_data.py` makes the same stratified split with seed 42: 1,437 training images and 360 test images. Both models use Adam with learning rate 0.001 and batch size 16. The reported score is the best test accuracy seen during training.

| Model | Epochs | Parameters | Best test accuracy |
| --- | ---: | ---: | ---: |
| Mixer-SNN small | 3 | 24,523,310 | 96.94% |
| Dual Memory Pathways (`d=100`, `theta=300`) | 3 | 83,213 | 50.00% |
| Dual Memory Pathways (`d=100`, `theta=300`) | 20 | 83,213 | 91.67% |

Mixer-SNN receives each image resized to 64×64 and repeated across three channels, with `T=1`. Dual Memory Pathways receives the same 8×8 pixel values as a 64-step grayscale sequence. These are architecture-specific input representations of the same examples and labels.

The Dual Memory runs use model initialization seed 3 and neuron threshold 0.02. The default seed and threshold left the neurons silent on this short Digits sequence; the adapted threshold is retained across neuron resets. The 20-epoch Dual Memory run spent 225.5 seconds inside training and evaluation epochs on this CPU. These are exploratory results: the test set was checked after each epoch and used to select the best checkpoint, and the 20-epoch score has a larger training budget than the 3-epoch Mixer score.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python train.py --device cpu --data digits --input-size 64 --model_size small --T 1 --batch-size 16 --epochs 3 --workers 0 --opt adam --lr 0.001 --mixup 0 --cutmix 0 --lr-warmup-epochs 0 --exp-name digits-cpu
python train_dual_digits.py --epochs 3 --batch-size 16 --seed 42 --model-seed 3 --lr 0.001 --memory-size 100 --theta 300 --threshold 0.02 --output-dir logs/dual-digits-d100
python train_dual_digits.py --epochs 20 --batch-size 16 --seed 42 --model-seed 3 --lr 0.001 --memory-size 100 --theta 300 --threshold 0.02 --output-dir logs/dual-digits-d100-e20
```

Dual Memory saves `best.pth`, `history.json`, and `comparison.json` in its output directory. The 3-epoch Mixer checkpoint is at `logs/digits-cpu_digits_mixer_sparse_T1_b16_e3_adam_lr0.001_seed42/pt/checkpoint_max_test_acc1.pth`.
