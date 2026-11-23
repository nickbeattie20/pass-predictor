import math

# ---- RF parameters (typical values for a UHF CubeSat downlink) --------

FREQ_MHZ = 437.0          # Common amateur/CubeSat UHF band
TX_POWER_DBM = 30.0       # ~1 Watt transmitter, in dBm (30 dBm = 1 W)
TX_ANTENNA_GAIN_DBI = 0.0   # Simple dipole/monopole on the satellite: ~0 dBi
RX_ANTENNA_GAIN_DBI = 12.0  # A decent ground station Yagi antenna
OTHER_LOSSES_DB = 2.0       # Atmospheric + pointing + cable losses (lumped estimate)
RX_NOISE_FLOOR_DBM = -110.0  # Typical noise floor for a modest ground station receiver


def free_space_path_loss_db(distance_km, freq_mhz=FREQ_MHZ):
    """Free-space path loss in dB. This is usually the dominant loss term."""
    if distance_km <= 0:
        return float("inf")
    return 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz) + 32.44


def received_power_dbm(distance_km,
                        tx_power_dbm=TX_POWER_DBM,
                        tx_gain_dbi=TX_ANTENNA_GAIN_DBI,
                        rx_gain_dbi=RX_ANTENNA_GAIN_DBI,
                        other_losses_db=OTHER_LOSSES_DB,
                        freq_mhz=FREQ_MHZ):
    """Received power at the ground station, in dBm."""
    fspl = free_space_path_loss_db(distance_km, freq_mhz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl - other_losses_db


def link_margin_db(distance_km, noise_floor_dbm=RX_NOISE_FLOOR_DBM, **kwargs):
    """Link margin: how far above the noise floor the received signal is."""
    p_rx = received_power_dbm(distance_km, **kwargs)
    return p_rx - noise_floor_dbm


def analyze_pass(pass_samples):
    """
    Takes the 'samples' list from a pass_predictor.py pass (each with
    'distance_km' and 'elevation_deg') and computes link margin at each
    point in time. Returns the samples annotated with margin_db and
    received_power_dbm, plus summary stats.
    """
    annotated = []
    for s in pass_samples:
        p_rx = received_power_dbm(s["distance_km"])
        margin = p_rx - RX_NOISE_FLOOR_DBM
        annotated.append({**s, "received_power_dbm": p_rx, "margin_db": margin})

    margins = [s["margin_db"] for s in annotated]
    closes = [m > 0 for m in margins]

    summary = {
        "samples": annotated,
        "max_margin_db": max(margins),
        "min_margin_db": min(margins),
        "link_closes_at_all": any(closes),
        "link_closes_fully": all(closes),
        "fraction_of_pass_closed": sum(closes) / len(closes) if closes else 0.0,
    }
    return summary


if __name__ == "__main__":
    # Quick sanity check across a range of distances
    print(f"{'Distance (km)':>15} | {'FSPL (dB)':>10} | {'Rx power (dBm)':>15} | {'Margin (dB)':>12}")
    print("-" * 62)
    for dist in [300, 500, 800, 1200, 2000]:
        fspl = free_space_path_loss_db(dist)
        p_rx = received_power_dbm(dist)
        margin = link_margin_db(dist)
        print(f"{dist:>15} | {fspl:>10.1f} | {p_rx:>15.1f} | {margin:>12.1f}")
