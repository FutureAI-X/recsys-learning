"""
diffusion_stable.py
稳定版 Diffusion - 通过固定种子和增加集成提高稳定性
"""

import numpy as np

from data_handler import get_data
from utils import set_seed
from modeling_lstm import train_stable_lstm
from modeling_diffusion import train_diffusion
from modeling_baseline import train_baseline

# ========== 配置 ==========
SLOT_DIM = 288
SEQ_LEN = 10
HIDDEN_DIM = 48  # 稍微增大
DIFFUSION_STEPS = 30  # 增加步数
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
SEED = 42

def print_results(results):
    # 统计运行结果
    print("\n" + "=" * 60)
    print("多次运行统计结果")
    print("=" * 60)
    if results['baseline'] :
        print(f"Baseline:   {np.mean(results['baseline']):.2%} ± {np.std(results['baseline']):.2%}")
    if results['lstm'] :
        print(f"LSTM:      {np.mean(results['lstm']):.2%} ± {np.std(results['lstm']):.2%}")
    if results['diffusion'] :
        print(f"Diffusion: {np.mean(results['diffusion']):.2%} ± {np.std(results['diffusion']):.2%}")
    if results['baseline'] :
        print(f"各次结果 Baseline: {[f'{x:.2%}' for x in results['baseline']]}")
    if results['lstm'] :
        print(f"各次结果 LSTM: {[f'{x:.2%}' for x in results['lstm']]}")
    if results['diffusion'] :
        print(f"各次结果 Diff: {[f'{x:.2%}' for x in results['diffusion']]}")
    print("=" * 60)

# ========== 多次运行取平均 ==========
def run_multiple_times(num_runs=3):
    """
    多次运行

    Args:
        num_runs (int, optional): 默认为 3
    """
    # 记录运行结果
    results = {'baseline': [], 'lstm': [], 'diffusion': []}

    for run in range(num_runs):
        print("\n" + "=" * 60)
        print(f"第 {run+1}/{num_runs} 次运行")
        print('='*60)

        # 设置随机种子
        seed = 42 + run * 10
        set_seed(seed)

        # 获取数据
        X_train, y_train, X_test, y_test = get_data(seq_len=SEQ_LEN)

        # BaseLine
        print("\n训练 BaseLine...")
        baseline_acc = train_baseline(X_train, y_train, X_test, y_test)
        results['baseline'].append(baseline_acc)

        # LSTM
        lstm_acc = train_stable_lstm(X_train, y_train, X_test, y_test, seed=seed, hidden_dim=HIDDEN_DIM, slot_dim=SLOT_DIM)
        results['lstm'].append(lstm_acc)

        # Diffusion
        print("\n训练 Stable Diffusion...")
        diff_acc = train_diffusion(X_train, y_train, X_test, y_test, seed=seed)
        results['diffusion'].append(diff_acc)

    print_results(results)

# ========== 主程序 ==========
def main():
    run_multiple_times(num_runs=1)


if __name__ == "__main__":
    main()