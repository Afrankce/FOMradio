"""Render the full metrics table directly from the archived summary JSON."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
data = root / 'results' / 'beam_matern'
doc = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
labels = [('seen_config', '已见配置'), ('array', '跨阵列'),
          ('angle_interpolation', '角度插值'), ('angle_extrapolation', '角度外推'),
          ('joint', '跨阵列与角度')]
variants = [('A', 'A'), ('uniform_fixed', '均匀 / 固定'), ('beam_fixed', 'Beam / 固定'),
            ('uniform_learned', '均匀 / 可学习'), ('beam_learned', 'Beam / 可学习')]
out = ['# 四组实验完整指标', '',
       '数据来源：`summary.json`。各数值为 seed 42 的场景、角度和 4 次采样的汇总，不能视为四个训练种子。',
       'MAE、RMSE 与边界 MAE 的单位是 dB，越低越好；PSNR 为 dB、SSIM 无量纲，越高越好。', '']
for key, name in labels:
    out += [f'## {name}', '', '| 变体 | MAE ↓ | RMSE ↓ | PSNR ↑ | SSIM ↑ | 边界 MAE ↓ |',
            '|---|---:|---:|---:|---:|---:|']
    for variant, display in variants:
        d = doc['groups'][key]['models'][variant]
        out.append(f"| {display} | {d['mae_db']:.3f} | {d['rmse_db']:.3f} | "
                   f"{d['psnr_db']:.3f} | {d['ssim']:.3f} | {d['boundary_mae_db']:.3f} |")
    out.append('')
(data / 'full_metrics.md').write_bytes(('\n'.join(out).rstrip() + '\n').encode('utf-8'))
