"""Train Dual Memory Pathways on the same Digits split as Mixer-SNN."""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from spikingjelly.clock_driven import functional

from digits_data import load_digits_split


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'Dual_memory_pathways' / 'smnist' / 'src' / 'dual_pathway.py'
DEFAULT_MIXER_CHECKPOINT = (
    ROOT / 'logs' / 'digits-cpu_digits_mixer_sparse_T1_b16_e3_adam_lr0.001_seed42'
    / 'pt' / 'checkpoint_max_test_acc1.pth'
)


class DigitsSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        self.images = torch.tensor(images, dtype=torch.float32).reshape(-1, 64, 1) / 16.0
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def load_model_class():
    spec = importlib.util.spec_from_file_location('dual_pathway_digits', MODEL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvLMU2


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0

    for images, targets in loader:
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = criterion(output, targets)
            if training:
                loss.backward()
                optimizer.step()
        functional.reset_net(model)
        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        correct += (output.argmax(dim=1) == targets).sum().item()
        count += batch_size

    return {'loss': total_loss / count, 'accuracy_pct': 100.0 * correct / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, default=3)
    parser.add_argument('--memory-size', type=int, default=100)
    parser.add_argument('--theta', type=int, default=300)
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'logs' / 'dual-digits-d100')
    parser.add_argument('--mixer-checkpoint', type=Path, default=DEFAULT_MIXER_CHECKPOINT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.memory_size < 1 or args.theta < 1 or args.threshold <= 0:
        parser.error('epochs, memory-size, theta, and threshold must be positive; batch-size must be at least 2')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_images, test_images, train_targets, test_targets = load_digits_split(args.seed)
    train_dataset = DigitsSequenceDataset(train_images, train_targets)
    test_dataset = DigitsSequenceDataset(test_images, test_targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    model_class = load_model_class()
    torch.manual_seed(args.model_seed)
    model = model_class(args.memory_size, 64, args.theta).to('cpu')
    for layer in (model.LMU1, model.LMU2):
        layer.f_x.v_threshold = args.threshold
        layer.f_x.set_reset_value('v_threshold', args.threshold)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        test_metrics = run_epoch(model, test_loader, criterion)
        elapsed = time.perf_counter() - start
        row = {'epoch': epoch, 'train': train_metrics, 'test': test_metrics, 'seconds': elapsed}
        history.append(row)
        print(
            f'Epoch {epoch}/{args.epochs}: train_acc={train_metrics["accuracy_pct"]:.2f}% '
            f'test_acc1={test_metrics["accuracy_pct"]:.2f}% time={elapsed:.1f}s',
            flush=True,
        )
        if test_metrics['accuracy_pct'] > best_accuracy:
            best_accuracy = test_metrics['accuracy_pct']
            torch.save(
                {'model': model.state_dict(), 'epoch': epoch, 'test_acc1': best_accuracy},
                args.output_dir / 'best.pth',
            )
        (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2))

    comparison = {
        'dataset': 'sklearn Digits',
        'seed': args.seed,
        'model_seed': args.model_seed,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'epochs': args.epochs,
        'dual_memory_test_acc1_pct': best_accuracy,
        'dual_memory_parameters': sum(p.numel() for p in model.parameters()),
        'dual_memory_input': '8x8 grayscale flattened to 64 time steps',
        'dual_memory_threshold': args.threshold,
        'dual_memory_memory_size': args.memory_size,
        'dual_memory_theta': args.theta,
        'mixer_input': '8x8 grayscale resized to 64x64 RGB',
    }
    if args.mixer_checkpoint.is_file():
        mixer_checkpoint = torch.load(args.mixer_checkpoint, map_location='cpu', weights_only=False)
        mixer_args = mixer_checkpoint['args']
        if mixer_args.data != 'digits' or mixer_args.seed != args.seed:
            raise ValueError('Mixer checkpoint must use Digits with the same split seed')
        comparison['mixer_snn_test_acc1_pct'] = float(mixer_checkpoint['max_test_acc1'])
        comparison['mixer_snn_epochs'] = mixer_args.epochs
        if mixer_args.epochs != args.epochs:
            print('Training epoch counts differ; compare accuracies with that in mind.', flush=True)
        print(
            f'Mixer-SNN best test_acc1={comparison["mixer_snn_test_acc1_pct"]:.2f}% | '
            f'Dual Memory best test_acc1={best_accuracy:.2f}%',
            flush=True,
        )
    else:
        print(f'Mixer checkpoint not found: {args.mixer_checkpoint}', flush=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
