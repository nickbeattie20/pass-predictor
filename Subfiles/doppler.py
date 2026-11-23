SPEED_OF_LIGHT_KM_S = 299_792.458


def doppler_shift_hz(freq_hz, range_rate_km_s):
    """
    Frequency shift (observed - transmitted) in Hz, for a given transmit
    frequency and range-rate (km/s, positive = receding).

    Negative range-rate (satellite approaching, distance shrinking) gives
    a positive shift (blueshift, higher frequency) -- matches the familiar
    "pitch rises as it approaches, drops as it passes" Doppler intuition.
    """
    return -freq_hz * (range_rate_km_s / SPEED_OF_LIGHT_KM_S)


def observed_frequency_hz(freq_hz, range_rate_km_s):
    """The actual frequency you'd need to tune to, in Hz."""
    return freq_hz + doppler_shift_hz(freq_hz, range_rate_km_s)


def analyze_pass_doppler(pass_samples, freq_hz):
    """
    Takes the 'samples' list from a pass_predictor.py pass (each with
    'range_rate_km_s') and computes Doppler shift + observed frequency at
    each point in time. Returns the samples annotated with doppler_shift_hz
    and observed_freq_hz, plus summary stats.
    """
    annotated = []
    for s in pass_samples:
        shift = doppler_shift_hz(freq_hz, s["range_rate_km_s"])
        annotated.append({
            **s,
            "doppler_shift_hz": shift,
            "observed_freq_hz": freq_hz + shift,
        })

    shifts = [s["doppler_shift_hz"] for s in annotated]

    summary = {
        "samples": annotated,
        "max_shift_hz": max(shifts),
        "min_shift_hz": min(shifts),
        "total_shift_swing_hz": max(shifts) - min(shifts),
    }
    return summary


if __name__ == "__main__":
    # Quick sanity check: a satellite approaching then receding at typical
    # LEO closing speeds, at a 437 MHz UHF downlink.
    freq = 437_000_000.0  # 437 MHz in Hz
    print(f"{'Range-rate (km/s)':>18} | {'Doppler shift (Hz)':>19} | {'Observed freq (MHz)':>20}")
    print("-" * 63)
    for rate in [-7.0, -3.5, 0.0, 3.5, 7.0]:
        shift = doppler_shift_hz(freq, rate)
        observed = observed_frequency_hz(freq, rate)
        print(f"{rate:>18} | {shift:>19.1f} | {observed / 1e6:>20.6f}")
