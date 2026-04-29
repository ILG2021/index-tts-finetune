"""
声场匹配 Pipeline v2
将音频 B 的声场特征匹配到音频 A
核心原则：保守处理，宁可欠矫正也不损伤音质

用法:
  python audio_match_v2.py --ref a.wav --target b.wav --output b_matched.wav

可选参数:
  --sr        采样率（默认自动读取，不做重采样）
  --eq        EQ 匹配强度 0.0~1.0（默认 0.8）
  --reverb    是否做混响匹配（默认 yes）
  --noise     是否做噪声底匹配（默认 yes）
"""

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve, sosfilt, butter
import argparse
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════

def load_audio(path: str):
    """读取音频，保留原始采样率，转单声道"""
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), sr


def rms_db(y: np.ndarray) -> float:
    r = np.sqrt(np.mean(y ** 2))
    return 20 * np.log10(r + 1e-10)


def apply_gain_db(y: np.ndarray, db: float) -> np.ndarray:
    return y * (10 ** (db / 20))


# ══════════════════════════════════════════════════════
# Step 1: 响度匹配（最安全，必做）
# ══════════════════════════════════════════════════════

def match_loudness(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """对齐整体 RMS 响度"""
    db_ref = rms_db(ref)
    db_tgt = rms_db(target)
    gain = db_ref - db_tgt
    print(f"  参考 RMS={db_ref:.1f}dB  目标 RMS={db_tgt:.1f}dB  增益={gain:+.1f}dB")
    return apply_gain_db(target, gain)


# ══════════════════════════════════════════════════════
# Step 2: EQ 匹配（频谱包络，最关键）
# ══════════════════════════════════════════════════════

def compute_mean_spectrum(y: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """计算平均幅度谱（正确的帧处理）"""
    hop = n_fft // 4
    window = np.hanning(n_fft)

    # 手动分帧，确保维度正确
    n_frames = (len(y) - n_fft) // hop
    if n_frames < 1:
        # 音频太短，直接 FFT
        padded = np.zeros(n_fft)
        padded[:len(y)] = y
        return np.abs(np.fft.rfft(padded * window))

    specs = []
    for i in range(n_frames):
        frame = y[i * hop: i * hop + n_fft] * window
        specs.append(np.abs(np.fft.rfft(frame)))

    return np.mean(specs, axis=0) + 1e-10


def smooth_spectrum(spec: np.ndarray, octave_fraction: float = 1/3) -> np.ndarray:
    """
    1/3 倍频程平滑（对数域平滑，更符合人耳感知）
    比简单移动平均好得多，不会引入梳状滤波
    """
    n = len(spec)
    smoothed = np.zeros(n)
    for i in range(n):
        # 以当前 bin 为中心，按比例取宽度
        width = max(1, int(i * octave_fraction))
        lo = max(0, i - width)
        hi = min(n, i + width + 1)
        smoothed[i] = np.mean(spec[lo:hi])
    return smoothed


def match_eq(target: np.ndarray, ref: np.ndarray,
             n_fft: int = 2048, strength: float = 0.8) -> np.ndarray:
    """
    EQ 匹配：把 target 的频谱包络调整到接近 ref
    strength: 0=不处理, 1=完全匹配（建议 0.7~0.9）
    """
    ref_spec   = compute_mean_spectrum(ref, n_fft)
    tgt_spec   = compute_mean_spectrum(target, n_fft)

    # 1/3 倍频程平滑后求比值
    ref_smooth = smooth_spectrum(ref_spec)
    tgt_smooth = smooth_spectrum(tgt_spec)

    ratio = ref_smooth / (tgt_smooth + 1e-10)

    # 限幅：±12dB（保守），避免极端频段被过度拉伸
    ratio = np.clip(ratio, 10 ** (-12/20), 10 ** (12/20))

    # strength 插值：不完全应用
    ratio = ratio ** strength    # ratio^1 = 完全匹配, ratio^0 = 不变

    # 统计信息
    ratio_db = 20 * np.log10(ratio)
    print(f"  EQ 调整范围: {ratio_db.min():.1f}dB ~ {ratio_db.max():.1f}dB  强度={strength}")

    # ── 核心：逐帧在频域乘以 ratio ──
    hop = n_fft // 4
    window = np.hanning(n_fft)
    # 输出缓冲（OLA 合成）
    out = np.zeros(len(target) + n_fft)
    norm = np.zeros(len(target) + n_fft)

    n_frames = (len(target) - n_fft) // hop
    for i in range(n_frames):
        start = i * hop
        frame = target[start: start + n_fft] * window
        spec  = np.fft.rfft(frame)
        # 乘以均衡曲线（保留相位）
        spec_eq = spec * ratio
        frame_eq = np.fft.irfft(spec_eq).real * window
        out[start: start + n_fft]  += frame_eq
        norm[start: start + n_fft] += window ** 2

    # OLA 归一化
    norm = np.where(norm < 1e-8, 1.0, norm)
    out = out / norm
    out = out[:len(target)]

    return out


# ══════════════════════════════════════════════════════
# Step 3: 混响匹配（轻度，只加不减）
# ══════════════════════════════════════════════════════

def estimate_rt60_simple(y: np.ndarray, sr: int) -> float:
    """
    简单 RT60 估算：用短时能量包络的衰减斜率
    比直接用累积能量更稳健
    """
    frame_len = int(0.02 * sr)  # 20ms
    hop = frame_len // 2
    n_frames = (len(y) - frame_len) // hop
    if n_frames < 5:
        return 0.3

    energies = []
    for i in range(n_frames):
        f = y[i * hop: i * hop + frame_len]
        energies.append(np.mean(f ** 2))

    energies = np.array(energies) + 1e-10
    db = 10 * np.log10(energies)

    # 取后半段（衰减尾部）估算斜率
    tail = db[n_frames // 2:]
    if len(tail) < 4:
        return 0.3

    t = np.arange(len(tail)) * hop / sr
    try:
        slope, _ = np.polyfit(t, tail, 1)
        if slope >= 0:
            return 0.3  # 没有衰减，无法估算
        rt60 = -60.0 / slope
        return float(np.clip(rt60, 0.05, 4.0))
    except Exception:
        return 0.3


def add_reverb(target: np.ndarray, rt60_add: float, sr: int, wet: float = 0.25) -> np.ndarray:
    """
    叠加简单混响（只在需要增加混响时调用）
    wet: 湿声比例，保守默认 0.25
    """
    ir_len = int(rt60_add * sr)
    t = np.linspace(0, rt60_add, ir_len)
    # 指数衰减 + 随机扩散
    ir = np.random.randn(ir_len) * np.exp(-6.91 * t / rt60_add)
    ir[0] = 1.0   # 直达声
    ir /= np.max(np.abs(ir)) + 1e-10

    reverb = fftconvolve(target, ir, mode='full')[:len(target)]
    # 对齐电平
    reverb = reverb * (np.sqrt(np.mean(target**2)) / (np.sqrt(np.mean(reverb**2)) + 1e-10))
    return (1 - wet) * target + wet * reverb


def match_reverb(target: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    rt60_ref = estimate_rt60_simple(ref, sr)
    rt60_tgt = estimate_rt60_simple(target, sr)
    delta = rt60_ref - rt60_tgt
    print(f"  RT60 估算 参考≈{rt60_ref:.2f}s  目标≈{rt60_tgt:.2f}s  差值={delta:+.2f}s")

    if delta < 0.08:
        print("  差异小于阈值，跳过混响处理（避免损伤音质）")
        return target

    # 只做增加混响（减混响容易引入失真，不做）
    wet = min(delta / rt60_ref * 0.5, 0.35)   # 湿声比例上限 35%
    print(f"  添加混响 rt60_add={delta:.2f}s  wet={wet:.2f}")
    return add_reverb(target, delta, sr, wet)


# ══════════════════════════════════════════════════════
# Step 4: 噪声底匹配（只叠加，不做谱减）
# ══════════════════════════════════════════════════════

def get_noise_level_db(y: np.ndarray, sr: int) -> float:
    """取最安静的 5% 帧的 RMS，作为噪声底估算"""
    frame_len = int(0.05 * sr)
    hop = frame_len
    n_frames = len(y) // frame_len
    if n_frames < 2:
        return rms_db(y)

    rms_list = []
    for i in range(n_frames):
        f = y[i * frame_len: (i+1) * frame_len]
        rms_list.append(np.sqrt(np.mean(f**2)))

    rms_arr = np.array(rms_list) + 1e-10
    n_quiet = max(1, int(len(rms_arr) * 0.05))
    quiet_rms = np.sort(rms_arr)[:n_quiet].mean()
    return 20 * np.log10(quiet_rms)


def match_noise_floor(target: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    noise_db_ref = get_noise_level_db(ref, sr)
    noise_db_tgt = get_noise_level_db(target, sr)
    delta_db = noise_db_ref - noise_db_tgt
    print(f"  噪声底 参考={noise_db_ref:.1f}dB  目标={noise_db_tgt:.1f}dB  差值={delta_db:+.1f}dB")

    # 只在需要增加噪声时处理，且差值 > 3dB 才有意义
    if delta_db < 3.0:
        print("  噪声底差异小，跳过（避免损伤音质）")
        return target

    # 生成与参考噪声底相近的白噪声，叠加到目标
    noise_amp = 10 ** (noise_db_ref / 20)
    noise = np.random.randn(len(target)) * noise_amp
    # 低通滤波（人声噪声底通常不是纯白噪声）
    sos = butter(4, 6000 / (sr / 2), btype='low', output='sos')
    noise = sosfilt(sos, noise)
    print(f"  叠加着色噪声 {noise_db_ref:.1f}dB")
    return target + noise


# ══════════════════════════════════════════════════════
# Step 5: 峰值限幅
# ══════════════════════════════════════════════════════

def safe_normalize(y: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 10 ** (headroom_db / 20):
        y = y * (10 ** (headroom_db / 20)) / peak
    return y


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

def match_audio(ref_path: str, target_path: str, output_path: str,
                eq_strength: float = 0.8,
                do_reverb: bool = True,
                do_noise: bool = True):

    print(f"\n{'='*52}")
    print(f"  参考音频 (A): {ref_path}")
    print(f"  目标音频 (B): {target_path}")
    print(f"  输出:         {output_path}")
    print(f"{'='*52}\n")

    print("📂 加载音频...")
    ref, sr_ref = load_audio(ref_path)
    tgt, sr_tgt = load_audio(target_path)
    print(f"  A: {len(ref)/sr_ref:.2f}s @ {sr_ref}Hz")
    print(f"  B: {len(tgt)/sr_tgt:.2f}s @ {sr_tgt}Hz")

    # 如果采样率不同，把 B 重采样到 A 的采样率
    if sr_ref != sr_tgt:
        print(f"  ⚠️  采样率不一致，将 B 重采样: {sr_tgt}→{sr_ref}Hz")
        tgt = librosa.resample(tgt, orig_sr=sr_tgt, target_sr=sr_ref)
    sr = sr_ref

    print("\n📊 Step 1: 响度匹配...")
    tgt = match_loudness(tgt, ref)

    print(f"\n🎛  Step 2: EQ 匹配 (strength={eq_strength})...")
    tgt = match_eq(tgt, ref, strength=eq_strength)

    if do_reverb:
        print("\n🏠 Step 3: 混响匹配...")
        tgt = match_reverb(tgt, ref, sr)
    else:
        print("\n🏠 Step 3: 混响匹配（已跳过）")

    if do_noise:
        print("\n🔇 Step 4: 噪声底匹配...")
        tgt = match_noise_floor(tgt, ref, sr)
    else:
        print("\n🔇 Step 4: 噪声底匹配（已跳过）")

    print("\n💾 写出文件...")
    tgt = safe_normalize(tgt)
    sf.write(output_path, tgt.astype(np.float32), sr)
    print(f"  ✅ 完成: {output_path}")
    print(f"\n{'='*52}\n")


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="声场匹配 v2：将 B 的声场对齐到 A")
    parser.add_argument("--ref",    required=True,          help="参考音频 A")
    parser.add_argument("--target", required=True,          help="待处理音频 B（TTS）")
    parser.add_argument("--output", default="matched.wav",  help="输出文件")
    parser.add_argument("--eq",     type=float, default=0.8,help="EQ 强度 0~1（默认 0.8）")
    parser.add_argument("--no-reverb", action="store_true", help="跳过混响匹配")
    parser.add_argument("--no-noise",  action="store_true", help="跳过噪声底匹配")
    args = parser.parse_args()

    match_audio(
        ref_path    = args.ref,
        target_path = args.target,
        output_path = args.output,
        eq_strength = args.eq,
        do_reverb   = not args.no_reverb,
        do_noise    = not args.no_noise,
    )
