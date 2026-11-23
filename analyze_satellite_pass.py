# to run localhost site: streamlit run app.py

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from Subfiles.pass_predictor import load_satellite, find_passes, print_pass_summary
from Subfiles.link_budget import analyze_pass, FREQ_MHZ
from Subfiles.doppler import analyze_pass_doppler


def plot_pass(pass_data, pass_number, output_path):
    """Plot elevation, link margin, and Doppler shift vs. time for one pass."""
    link_analysis = analyze_pass(pass_data["samples"])
    doppler_analysis = analyze_pass_doppler(pass_data["samples"], freq_hz=FREQ_MHZ * 1e6)
    samples = link_analysis["samples"]
    doppler_samples = doppler_analysis["samples"]

    times = [s["time"] for s in samples]
    elevations = [s["elevation_deg"] for s in samples]
    margins = [s["margin_db"] for s in samples]
    shifts_khz = [s["doppler_shift_hz"] / 1000.0 for s in doppler_samples]

    # Convert times to minutes since rise, for a cleaner x-axis
    t0 = times[0]
    minutes = [(t - t0).total_seconds() / 60 for t in times]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax1.plot(minutes, elevations, color="tab:blue")
    ax1.set_ylabel("Elevation (degrees)")
    ax1.set_title(f"Pass {pass_number}: elevation, link margin, and Doppler shift")
    ax1.grid(True, alpha=0.3)

    ax2.plot(minutes, margins, color="tab:orange")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1, label="Link closes here (margin = 0)")
    ax2.fill_between(minutes, margins, 0, where=[m > 0 for m in margins],
                      color="tab:green", alpha=0.15, label="Link closed")
    ax2.fill_between(minutes, margins, 0, where=[m <= 0 for m in margins],
                      color="tab:red", alpha=0.15, label="Link does not close")
    ax2.set_ylabel("Link margin (dB)")
    ax2.legend(loc="lower center", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(minutes, shifts_khz, color="tab:purple")
    ax3.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax3.set_ylabel("Doppler shift (kHz)")
    ax3.set_xlabel("Minutes since rise")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nPass {pass_number} summary:")
    print(f"  Max margin: {link_analysis['max_margin_db']:.1f} dB")
    print(f"  Min margin: {link_analysis['min_margin_db']:.1f} dB")
    print(f"  Link closes for {link_analysis['fraction_of_pass_closed']*100:.0f}% of the pass")
    print(f"  Doppler shift range: {doppler_analysis['min_shift_hz']/1000:+.2f} kHz to "
          f"{doppler_analysis['max_shift_hz']/1000:+.2f} kHz "
          f"(total swing: {doppler_analysis['total_shift_swing_hz']/1000:.2f} kHz)")
    print(f"  Plot saved to: {output_path}")


if __name__ == "__main__":
    sat, ts = load_satellite(use_fallback=False)  # switch to True to force the offline fallback TLE
    passes = find_passes(sat, ts)
    print_pass_summary(passes)

    if passes:
        # Plot the pass with the highest max elevation (the "best" pass)
        best_pass = max(passes, key=lambda p: max(s["elevation_deg"] for s in p["samples"]))
        best_index = passes.index(best_pass) + 1
        plot_pass(best_pass, best_index, "best_pass_link_margin.png")
    else:
        print("No passes to plot.")

