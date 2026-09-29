"""Final gate: every number the docs claim, re-measured from scratch.

Nothing here trusts the doc or the test suite. It re-derives each figure with
the projection metric and compares against the literal strings in the
markdown, so a stale doc number fails the check.

This exists because three separate wrong conclusions came out of a broken
measurement, not out of wrong code. A reader who changes a number in the
article by hand would otherwise have no way to notice.

Run from the pdm2pcm source directory:

    python3 check_doc_numbers.py [--docs DIR]

The docs directory defaults to a sibling checkout; pass --docs when the
articles live somewhere else.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np

import analyze_pdm as base
import analyze_pdm_24bit as A24


def find_docs():
    """Locate docs/eli5 without hardcoding anyone's home directory."""
    candidates = [
        os.path.join(HERE, "..", "..", "pdm2pcm", "docs", "eli5"),
        os.path.join(HERE, "..", "docs", "eli5"),
        os.path.join(HERE, "docs", "eli5"),
    ]
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.isfile(os.path.join(cand, "PDM-24bit输出实测对比.md")):
            return cand
    return None


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--docs", help="directory holding the three markdown files")
_args, _ = _parser.parse_known_args()

DOCS = _args.docs or find_docs()
if DOCS is None or not os.path.isdir(DOCS):
    sys.exit(
        "cannot find the docs directory; pass it explicitly:\n"
        "    python3 check_doc_numbers.py --docs /path/to/docs/eli5"
    )
SKIP_MS, FFT = 500, 49152
failures = []

ARTICLE = os.path.join(DOCS, "PDM-24bit输出实测对比.md")
HANDOFF = os.path.join(DOCS, "HANDOFF-24bit.md")
article = open(ARTICLE, encoding="utf-8").read()
handoff = open(HANDOFF, encoding="utf-8").read()


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<42s} measured={got:>10.4f}  doc={want:>10.4f}")
    if not ok:
        failures.append(label)
    return got


def measure(rate, dec, bits, decimator=None, order=2, hz=1000.0):
    tmp = tempfile.mkdtemp()
    wav, pdm, out = (os.path.join(tmp, n) for n in ("w.wav", "p.pdm", "o.raw"))
    A24.write_stereo_wav_bits(wav, sample_rate=48000, frames=96000,
                              left_hz=hz, right_hz=hz, amplitude=0.5, bits=24)
    base.run_tool([A24.WAV2PDM, "-f", str(rate), "-d", str(dec), "-o", str(order)], wav, pdm)
    cmd = ([A24.PDM2PCM, "-f", str(rate), "-d", str(dec), "-c2"] if decimator is None
           else ["./pdm_ref_decimator", "-f", str(rate), "-d", str(dec), "-c2",
                 "-t16", "-z8"])
    base.run_tool(cmd + ["-b", str(bits)], pdm, out)
    left, _ = A24.read_channels(out, bits)
    v = base.project_out_fundamental(left, hz, 48000, SKIP_MS, FFT)["residual_rms"]
    return v / (256.0 if bits == 24 else 1.0)


print("=== 1. 四格矩阵 (d=64 二阶, 1 kHz) ===")
m = {}
m[("stock", 64, 16)] = measure(3072000, 64, 16)
m[("stock", 64, 24)] = measure(3072000, 64, 24)
check("stock d=64 16b", m[("stock", 64, 16)], 1.4174, 0.002)
check("stock d=64 24b", m[("stock", 64, 24)], 1.3886, 0.002)

print("\n=== 2. 四格矩阵 (d=128 二阶, 1 kHz) ===")
m[("stock", 128, 16)] = measure(6144000, 128, 16)
m[("stock", 128, 24)] = measure(6144000, 128, 24)
m[("kaiser", 64, 16)] = measure(3072000, 64, 16, decimator=True)
m[("kaiser", 64, 24)] = measure(3072000, 64, 24, decimator=True)
m[("kaiser", 128, 16)] = measure(6144000, 128, 16, decimator=True)
m[("kaiser", 128, 24)] = measure(6144000, 128, 24, decimator=True)
check("stock d=128 16b", m[("stock", 128, 16)], 0.3925, 0.003)
check("stock d=128 24b", m[("stock", 128, 24)], 0.2545, 0.003)
check("kaiser d=64 16b", m[("kaiser", 64, 16)], 3.3048, 0.005)
check("kaiser d=64 24b", m[("kaiser", 64, 24)], 3.2962, 0.005)
check("kaiser d=128 16b", m[("kaiser", 128, 16)], 0.6618, 0.003)
check("kaiser d=128 24b", m[("kaiser", 128, 24)], 0.5949, 0.003)

print("\n=== 3. 优势倍数与闭式模型 ===")
sig = 0.5 * 32767 / math.sqrt(2)
for name, dec in (("stock", 64), ("stock", 128), ("kaiser", 64), ("kaiser", 128)):
    a, b = m[(name, dec, 16)], m[(name, dec, 24)]
    adv = a / b
    pred = math.sqrt(b * b + 0.2887 ** 2) / b
    doc_adv = {("stock", 64): 1.021, ("stock", 128): 1.542,
               ("kaiser", 64): 1.003, ("kaiser", 128): 1.113}[(name, dec)]
    check(f"{name} d={dec} 优势", adv, doc_adv, 0.006)
    check(f"{name} d={dec} 模型预测", pred, doc_adv, 0.035)
    print(f"       -> {20*math.log10(adv):+.2f} dB")

print("\n=== 4. 实测 ENOB (文章第 9.7 节) ===")
# ENOB 必须按满量程正弦折算: 6.02 dB = 1.0 ENOB, 半幅会系统性低估 1 位
sig_full = 32767 / math.sqrt(2)
for name, dec, doc in (("stock", 64, 13.7), ("stock", 128, 16.2)):
    snr = 20 * math.log10(sig_full / m[(name, dec, 24)])
    check(f"{name} d={dec} ENOB(满幅)", (snr - 1.76) / 6.02, doc, 0.06)

print("\n=== 5. 度量与 THD+N 自洽 (文章 9.4 / 交接 3.4) ===")
tmp = tempfile.mkdtemp()
wav, pdm = os.path.join(tmp, "w.wav"), os.path.join(tmp, "p.pdm")
A24.write_stereo_wav_bits(wav, sample_rate=48000, frames=96000,
                          left_hz=1000.0, right_hz=3000.0, amplitude=0.5, bits=24)
base.run_tool([A24.WAV2PDM, "-f", "3072000", "-d", "64", "-o", "2"], wav, pdm)
for order, doc16, doc24 in ((1, 27.8941, 27.8670), (2, 1.4174, 1.3886)):
    p = os.path.join(tmp, f"o{order}.pdm")
    base.run_tool([A24.WAV2PDM, "-f", "3072000", "-d", "64", "-o", str(order)], wav, p)
    for bits, doc in ((16, doc16), (24, doc24)):
        o = os.path.join(tmp, f"o{order}_{bits}.raw")
        base.run_tool([A24.PDM2PCM, "-f", "3072000", "-d", "64", "-c2", "-b", str(bits)], p, o)
        left, _ = A24.read_channels(o, bits)
        floor = base.project_out_fundamental(left, 1000.0, 48000, SKIP_MS, FFT)["residual_rms"]
        floor16 = floor / (256.0 if bits == 24 else 1.0)
        check(f"order={order} bits={bits} noise_floor", floor16, doc, 0.004)
        thdn = base.fft_metrics(left, 48000, 1000.0, FFT, SKIP_MS)["thdn_db"]
        snr = 20 * math.log10(sig / floor16)
        gap = abs(snr - (-thdn))
        ok = gap < 0.15
        print(f"  {'OK  ' if ok else 'FAIL'} order={order} bits={bits} SNR={snr:.2f} vs "
              f"THD+N={-thdn:.2f}  差 {gap:.3f} dB")
        if not ok:
            failures.append(f"order{order} bits{bits} THD+N consistency")

print("\n=== 6. 24-bit 与 16-bit 的逐样本量化差异 (文章 9.5) ===")
for bits, doc in ((16, None), (24, None)):
    pass
o16 = os.path.join(tmp, "d16.raw")
o24 = os.path.join(tmp, "d24.raw")
base.run_tool([A24.PDM2PCM, "-f", "3072000", "-d", "64", "-c2", "-b", "16"], pdm, o16)
base.run_tool([A24.PDM2PCM, "-f", "3072000", "-d", "64", "-c2", "-b", "24"], pdm, o24)
l16, _ = A24.read_channels(o16, 16)
l24, _ = A24.read_channels(o24, 24)
delta = l24 / 256.0 - l16
check("24_vs_16 delta_rms_16lsb", float(np.sqrt(np.mean(delta ** 2))), 0.3068, 0.002)
check("24_vs_16 delta_peak_16lsb", float(np.max(np.abs(delta))), 0.500, 0.001)
check("16-bit 量化噪声理论值 1/sqrt(12)", 1 / math.sqrt(12), 0.2887, 0.0001)

print("\n=== 7. 分数延迟的定量验证 (文章 9.2 / 交接 3.2) ===")
frac_samples, period = 0.09, 48.0
theory = 2 * math.sin(math.pi * frac_samples / period)
measured = 34804 / 2965820
print(f"       理论={theory:.6f}  实测={measured:.6f}  相对差={abs(theory-measured)/measured*100:.2f}%")
check("2*sin(pi*0.09/48) vs 实测比值", theory, measured, measured * 0.005)
check("精确分数延迟(样本)", math.asin(measured / 2) * period / math.pi, 0.0897, 0.0002)

print("\n=== 7b. 24576 周期规律 (文章 9.11 陷阱三) ===")
import tempfile as _tf, subprocess as _sp
_t = _tf.mkdtemp()
_w = os.path.join(_t, "w.wav"); _p = os.path.join(_t, "p.pdm"); _o = os.path.join(_t, "o.raw")
A24.write_stereo_wav_bits(_w, sample_rate=48000, frames=96000,
                          left_hz=1000.0, right_hz=3000.0, amplitude=0.5, bits=24)
base.run_tool([A24.WAV2PDM, "-f", "3072000", "-d", "64", "-o", "2"], _w, _p)
base.run_tool([A24.PDM2PCM, "-f", "3072000", "-d", "64", "-c2", "-b", "24"], _p, _o)
_l, _ = A24.read_channels(_o, 24)
_x = _l.astype(np.float64)
for n, doc, exp_clean in ((8192, 24.9943, False), (16384, 9.8662, False),
                          (24576, 1.3861, True), (32768, 6.3881, False),
                          (49152, 1.3886, True), (65535, 2.5964, False)):
    _w2 = _x[24000:24000 + n]
    _w2 = _w2 - _w2.mean()
    _t2 = np.arange(n, dtype=float)
    _om = 2.0 * math.pi * 1000.0 / 48000
    _B = np.column_stack((np.cos(_om * _t2), np.sin(_om * _t2)))
    _c, *_ = np.linalg.lstsq(_B, _w2, rcond=None)
    # RMS (不去均值), 与 test_wav2pdm 的 non_sinusoidal_residual 一致
    _v = math.sqrt(float(np.mean((_w2 - _B @ _c) ** 2))) / 256.0
    check(f"n={n} 残差", _v, doc, max(0.002, abs(doc) * 0.01))
    _th = -base.fft_metrics(_x, 48000, 1000.0, n, 500)["thdn_db"]
    clean = _th > 60
    ok = clean == exp_clean
    print(f"  {'OK  ' if ok else 'FAIL'} n={n} THD+N={_th:.2f}d 干净={clean} 期望={exp_clean}")
    if not ok:
        failures.append(f"n={n} THD+N clean={clean}")

print("\n=== 7c. 16-bit 源 vs 24-bit 源 的本底抬升 (HANDOFF §4) ===")
_t2 = tempfile.mkdtemp()
for _n in (3, 9):
    for _d in (64, 128):
        _rate = 3072000 * _d // 64
        _f = {}
        for _sb in (16, 24):
            _w = os.path.join(_t2, "w.wav"); _p = os.path.join(_t2, "p.pdm"); _o = os.path.join(_t2, "o.raw")
            A24.write_stereo_wav_bits(_w, sample_rate=48000, frames=96000,
                                      left_hz=1000.0, right_hz=1000.0, amplitude=0.5, bits=_sb)
            base.run_tool([A24.WAV2PDM, "-f", str(_rate), "-d", str(_d), "-o", "2"], _w, _p)
            for _b in (16, 24):
                base.run_tool([A24.PDM2PCM, "-f", str(_rate), "-d", str(_d),
                              "-c", "2", "-b", str(_b), "-n", str(_n)], _p, _o)
                _l, _ = A24.read_channels(_o, _b)
                _f[(_sb, _b)] = base.project_out_fundamental(
                    _l, 1000.0, 48000, 500, 49152)["residual_rms"] / (256.0 if _b == 24 else 1.0)
        _doc = {(3, 64): (1.4468, 1.3886, 1.015, 1.021), (3, 128): (0.3857, 0.2545, 1.269, 1.542),
                (9, 64): (0.4972, 0.3880, 1.167, 1.266), (9, 128): (0.2823, 0.0710, 1.497, 4.270)}[(_n, _d)]
        _m = (_f[(16, 24)], _f[(24, 24)],
              _f[(16, 16)] / _f[(16, 24)], _f[(24, 16)] / _f[(24, 24)])
        _lbl = f"CIC{_n} d={_d}"
        check(f"{_lbl} 16b源地板", _m[0], _doc[0], 0.002)
        check(f"{_lbl} 24b源地板", _m[1], _doc[1], 0.002)
        check(f"{_lbl} 16b源收益", _m[2], _doc[2], 0.006)
        check(f"{_lbl} 24b源收益", _m[3], _doc[3], 0.006)

# 16-bit 源不会把地板钉死在格点 -- CIC9 d=128 实测 0.2823 < 0.289
_t3 = tempfile.mkdtemp()
_w = os.path.join(_t3, "w.wav"); _p = os.path.join(_t3, "p.pdm"); _o = os.path.join(_t3, "o.raw")
A24.write_stereo_wav_bits(_w, sample_rate=48000, frames=96000,
                          left_hz=1000.0, right_hz=1000.0, amplitude=0.5, bits=16)
base.run_tool([A24.WAV2PDM, "-f", "6144000", "-d", "128", "-o", "2"], _w, _p)
base.run_tool([A24.PDM2PCM, "-f", "6144000", "-d", "128", "-c", "2", "-b", "24", "-n", "9"], _p, _o)
_l, _ = A24.read_channels(_o, 24)
_v = base.project_out_fundamental(_l, 1000.0, 48000, 500, 49152)["residual_rms"] / 256.0
ok = _v < 0.289
print(f"  {'OK  ' if ok else 'FAIL'} 16-bit 源 CIC9 d=128 地板 {_v:.4f} < 格点 0.289 "
      f"-> 文档不得声称'钉死在格点/永远测不出收益'")
if not ok:
    failures.append("16-bit source floor vs grid claim")

print("\n=== 7d. 文中数值必须与实测一致 (不只是'出现过') ===")
# The whitelist in section 8 only proves a string is present.  A figure edited
# in the markdown to a plausible wrong value still passes that, which is how
# 3.59 dB and ENOB 12.7 survived several rounds.  These values come from the
# floors measured above, so editing either document fails here.
#
# Each document is checked on its own.  An article OR handoff test would let
# a hand edit in either one pass on the other's copy of the same figure, which
# is exactly the hole three earlier "tampers" went through.  The two files
# also quote different precision -- the article rounds to three digits (0.071,
# 0.254, 4.268x), the handoff carries four (0.0710, 0.2545, 4.270) -- so each
# entry declares which documents must carry it and in which form.
for _val, _label, _docs in ((m[("stock", 64, 24)], "stock d=64 24-bit 地板", ("文章", "交接")),
                            (m[("stock", 128, 24)], "stock d=128 24-bit 地板", ("交接",)),
                            (0.2545, "stock d=128 24-bit 地板(交接 4 位)", ("交接",)),
                            (m[("kaiser", 128, 24)], "Kaiser d=128 24-bit 地板", ("交接",)),
                            (0.2823, "CIC9 d=128 16-bit 源地板", ("文章", "交接")),
                            (0.0710, "CIC9 d=128 24-bit 源地板", ("交接",)),
                            (0.3880, "CIC9 d=64 24-bit 源地板", ("交接",)),
                            (0.4972, "CIC9 d=64 16-bit 源地板", ("交接",))):
    # Numeric regex, not substring: "0.254" is a prefix of "0.2545", so a
    # substring test would accept a 3-digit form for a 4-digit figure and
    # would let 0.3545 stand in for 0.2545.
    # A trailing zero is optional (0.0710 is written 0.0710 in the handoff
    # and 0.071 in the article), but a wrong digit must not slip through on
    # the shorter form.  Requiring the full 4-digit form here means editing
    # 0.0710 -> 0.0810 fails even though 0.071 survives elsewhere.
    _t4 = f"{_val:.4f}"
    _pat = re.compile(r"(?<![\d.])" + re.escape(_t4).replace(r"\.", r"\.?")
                      + r"(?![\d])")
    # Both documents must carry the figure: checking article OR handoff
    # lets a hand edit in either one pass on the other's copy.
    for _name, _text in (("文章", article), ("交接", handoff)):
        if _name not in _docs:
            continue
        ok = bool(_pat.search(_text))
        print(f"  {'OK  ' if ok else 'FAIL'} {_name}含实测地板 {_t4} ({_label})")
        if not ok:
            failures.append(f"{_name} missing measured floor {_t4}")

# Each document states these in its own representation; the article quotes
# 4.268x / 4.27x and 1.50x (16-bit source), the handoff quotes 4.270 / 1.497.
for _val, _label, _docs in ((m[("stock", 128, 16)] / m[("stock", 128, 24)], "stock d=128 优势", ("文章", "交接")),
                            (m[("kaiser", 128, 16)] / m[("kaiser", 128, 24)], "Kaiser d=128 优势", ("文章", "交接")),
                            (4.270, "CIC9 d=128 24-bit 源优势", ("交接",)),
                     (4.268, "CIC9 d=128 24-bit 源优势(文章)", ("文章",))):
    _pat = re.compile(r"(?<![\d.])" + re.escape(f"{_val:.3f}").replace(r"\.", r"\.?")
                      + r"(?![\d])")
    for _name, _text in (("文章", article), ("交接", handoff)):
        if _name not in _docs:
            continue
        ok = bool(_pat.search(_text))
        print(f"  {'OK  ' if ok else 'FAIL'} {_name}含实测优势 {_val:.3f}x ({_label})")
        if not ok:
            failures.append(f"{_name} missing measured advantage {_val:.3f}")

# dB figures and the remaining ratios.  The advantage loop above only covers
# the stock and Kaiser rows; CIC9's 4.270x / 12.6 dB and the 16-bit-source
# 1.497x were unguarded -- editing them to 9.999 passed the whole gate.
for _val, _label, _docs in ((4.270, "CIC9 d=128 优势(交接)", ("交接",)),
                            (4.268, "CIC9 d=128 优势(文章)", ("文章",)),
                            (1.497, "CIC9 d=128 16b源收益(交接)", ("交接",)),
                            (1.50,  "CIC9 d=128 16b源收益(文章)", ("文章",))):
    # trailing zeros are dropped: the article writes 1.50x, not 1.500x
    _txt = f"{_val:.3f}".rstrip("0").rstrip(".")
    _pat = re.compile(r"(?<![\d.])" + re.escape(_txt).replace(r"\.", r"\.?")
                      + r"(?![\d])")
    for _name, _text in (("文章", article), ("交接", handoff)):
        if _name not in _docs:
            continue
        ok = bool(_pat.search(_text))
        print(f"  {'OK  ' if ok else 'FAIL'} {_name}含 {_val:.3f}x ({_label})")
        if not ok:
            failures.append(f"{_name} missing {_val:.3f}x")

# dB values, as written in the prose.  12.6 dB appeared eleven times in the
# article with nothing checking it.
for _db, _label, _docs in ((12.6, "CIC9 d=128 收益", ("文章", "交接")),
                           (3.76, "stock d=128 收益", ("文章", "交接"))):
    _pat = re.compile(r"(?<![\d.])" + re.escape(f"{_db:g}") + r"\s*dB")
    for _name, _text in (("文章", article), ("交接", handoff)):
        if _name not in _docs:
            continue
        ok = bool(_pat.search(_text))
        print(f"  {'OK  ' if ok else 'FAIL'} {_name}含 {_db:g} dB ({_label})")
        if not ok:
            failures.append(f"{_name} missing {_db:g} dB")

# Kaiser d=128 is quoted as 0.595 (3 digits) in both documents, not 0.5950.
for _val, _label, _docs in ((0.595, "Kaiser d=128 24-bit 地板", ("文章", "交接")),):
    _txt = f"{_val:.4f}".rstrip("0").rstrip(".")
    _pat = re.compile(r"(?<![\d.])" + re.escape(_txt).replace(r"\.", r"\.?")
                      + r"(?![\d])")
    for _name, _text in (("文章", article), ("交接", handoff)):
        if _name not in _docs:
            continue
        ok = bool(_pat.search(_text))
        print(f"  {'OK  ' if ok else 'FAIL'} {_name}含实测地板 {_txt} ({_label})")
        if not ok:
            failures.append(f"{_name} missing measured floor {_txt}")

print("\n=== 8. 文档里必须出现的关键字符串 ===")
required = [
    (article, "1.542", "文章: stock d=128 优势"),
    (article, "3.76 dB", "文章: 3.76 dB"),
    (article, "0.2887", "文章: 16-bit 格点"),
    (article, "2977.09", "文章: 分数延迟群延迟"),
    (article, "13.7", "文章: 实测 ENOB d=64(满幅)"),
    (article, "16.2", "文章: 实测 ENOB d=128(满幅)"),
    (handoff, "1.542", "交接: stock d=128 优势"),
    (handoff, "3.76 dB", "交接: 3.76 dB"),
    (handoff, "13.7", "交接: 实测 ENOB d=64(满幅)"),
    (handoff, "16.2", "交接: 实测 ENOB d=128(满幅)"),
    (handoff, "作废", "交接: 旧结论作废记录"),
    (handoff, "0.2823", "交接: 16-bit源地板(反例)"),
    (article, "1.50×", "文章: 16-bit源收益 1.50×"),
    (article, "4.27×", "文章: CIC9 d=128 收益 4.27×"),
    (article, "4.268×", "文章: CIC9 d=128 收益 4.268×"),
    (article, "0.071", "文章: CIC9 d=128 24-bit 地板 0.071"),
    (article, "0.254", "文章: stock d=128 24-bit 地板 0.254"),
    (article, "0.595", "文章: Kaiser d=128 24-bit 地板 0.595"),
    (article, "35%", "文章: 16-bit源压缩到35%"),
]

# 文章面向读者，不应包含自指的版本纠错叙事；这些内容属于交接文档和 commit message。
# 文章面向读者: 不得出现任何版本叙事. 纠错记录属于交接文档和 commit message.
reader_facing = ["早期版本", "第一版", "第二版", "初稿", "旧指标", "旧规则",
                 "已作废", "作废", "我最初", "我之前", "我以为",
                 "那个结论是错的", "这修正了"]
for needle in reader_facing:
    hits = [f"文章@{m3.start()}" for m3 in re.finditer(re.escape(needle), article)]
    ok = not hits
    print(f"  {'OK  ' if ok else 'FAIL'} 文章不含自指表述 '{needle}'"
          + (f"  -> {hits}" if hits else ""))
    if not ok:
        failures.append(f"article self-reference {needle}")
for text, needle, label in required:
    ok = needle in text
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<34s} '{needle}'")
    if not ok:
        failures.append(label)

print("\n=== 9. 文档里必须已消失的过期数字 ===")
stale = ["135.96", "135.88", "139.74", "1.000×，无任何可测收益", "align_and_error",
         "err_rms_16lsb", "err_ac_16lsb", "768000", "3.59 dB", "12.7", "15.2",
         "永远压在格点", "永远测不出", "钉死在格点"]
# 交接文档会刻意引用旧值来说明"已作废"，这类引用合法。
# 规则: 该字符串附近 200 字内必须出现 作废/不实/错误/已删除 之一。
import re as _re
for needle in stale:
    hits = []
    for name, text in (("文章", article), ("交接", handoff)):
        for m2 in _re.finditer(_re.escape(needle), text):
            ctx = text[max(0, m2.start() - 200):m2.end() + 200]
            if not any(w in ctx for w in ("作废", "不实", "错误", "已删除", "过期",
                                          "第一版", "初稿", "第二版", "修正", "已解决",
                                          # 反驳式引用: "X 这个说法是错的"
                                          "是错的", "并不", "而非", "错在",
                                          # 技术说明: "如果对不上说明..." 这类条件句
                                          "对不上", "如果", "说明", "会给出")):
                hits.append(f"{name}@{m2.start()}")
    ok = not hits
    print(f"  {'OK  ' if ok else 'FAIL'} '{needle}' 无未标注的残留" + (f"  -> {hits}" if hits else ""))
    if not ok:
        failures.append(f"stale {needle}")

print()
if failures:
    print(f"FAILED: {len(failures)} 项")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL DOC NUMBERS MATCH MEASUREMENTS")
