import datetime

import matplotlib.pyplot as plt
import numpy as np


def _trend_xy(x0, x1, intercept, slope_pct):
    slope_per_day = slope_pct / 100
    return [x0, x1], [intercept, intercept + slope_per_day * (x1 - x0)]


def _month_year_ticks(first_date, x0, x1):
    # Ticks only on Jan-1 and Jul-1 of each year covered by [x0, x1], labeled
    # "Jan_YY"/"Jul_YY" (YY = last 2 digits of the year).
    base_date = datetime.datetime.strptime(first_date, '%Y%m%d').date()
    start_date = base_date + datetime.timedelta(days=int(np.floor(x0)))
    end_date = base_date + datetime.timedelta(days=int(np.ceil(x1)))

    ticks, labels = [], []
    for year in range(start_date.year - 1, end_date.year + 2):
        for month, name in ((1, 'Jan'), (7, 'Jul')):
            offset = (datetime.date(year, month, 1) - base_date).days
            if x0 <= offset <= x1:
                ticks.append(offset)
                labels.append(f'{name}_{year % 100:02d}')
    return ticks, labels


def _trend_alpha(significant):
    # Trend line more opaque if Mann-Kendall is significant, fainter otherwise.
    return 0.9 if significant else 0.35


def plot_pixel_summary(
    x_old, evi, x_p_new, evi_periodic, x_filtered,
    lt_evi, lt_evi_slope, lt_A, lt_evi_significant,
    lt_pp, pp_slope, pp_A, pp_significant,
    lt_temp, temp_slope, temp_A, temp_significant,
    residuals, x_valid, residuals_slope, residuals_A, residuals_significant,
    default_value, location_label, first_date,
    output_file,
    data_peaks=None,
):
    """
    Builds the 3-subplot figure (original+interpolated EVI, lt_evi+slope,
    combined detrend) for one pixel. Does not depend on series_analysis_20.py
    or get_slope: it receives everything precomputed (values + Mann-Kendall
    significance flags), so inputs can be cached and re-plotted quickly
    without recomputing DWT/GAM.
    """
    plt.switch_backend('Agg')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Subplot 1: original + interpolated EVI, only within the working period
    ax1.scatter(x_old, evi, c='tab:green', s=10, label='EVI original')
    ax1.plot(x_p_new, evi_periodic, c='tab:red', lw=1, label='EVI interpolated')

    if data_peaks:
        x_old_arr = np.asarray(x_old)
        evi_arr = np.asarray(evi)
        years = sorted({peak['year'] for peak in data_peaks})
        cmap = plt.get_cmap('tab10' if len(years) <= 10 else 'tab20')
        color_by_year = {year: cmap(i % cmap.N) for i, year in enumerate(years)}
        labeled_years = set()
        for peak in data_peaks:
            year = peak['year']
            color = color_by_year[year]
            label = f'Agro year {year}' if year not in labeled_years else None
            labeled_years.add(year)
            ax1.scatter(x_old_arr[peak['max_index']], evi_arr[peak['max_index']],
                        marker='^', s=90, color=color, edgecolors='black', linewidths=0.8, zorder=5, label=label)
            ax1.scatter(x_old_arr[peak['min_index']], evi_arr[peak['min_index']],
                        marker='v', s=90, color=color, edgecolors='black', linewidths=0.8, zorder=5)

    ax1.set_xlim(x_filtered[0], x_filtered[-1])
    ax1.set_ylim(0, 10000)
    ax1.set_ylabel('EVI')
    ax1.set_title('EVI original + interpolated')
    ax1.legend()

    # Subplot 2: lt_evi + slope (working period = x_filtered)
    ax2.plot(x_filtered, lt_evi, c='tab:orange', lw=2, label='lt_evi (DWT)')
    if lt_evi_slope != default_value:
        tx, ty = _trend_xy(x_filtered[0], x_filtered[-1], lt_A, lt_evi_slope)
        ax2.plot(tx, ty, c='tab:red', lw=2, linestyle=(0, (5, 10)), alpha=_trend_alpha(lt_evi_significant),
                 label=f'Slope EVI: {lt_evi_slope:.4f} %/day')
    ax2.set_ylim(0, 10000)
    ax2.set_ylabel('EVI')
    ax2.set_title('lt_evi + slope')
    ax2.legend()

    # Subplot 3: lt_evi, lt_pp, lt_temp (-273), residual + dashed slopes
    ax3_r = ax3.twinx()
    lines3 = ax3.plot(x_filtered, lt_evi, c='tab:green', lw=2, label='lt_evi')
    if lt_evi_slope != default_value:
        tx, ty = _trend_xy(x_filtered[0], x_filtered[-1], lt_A, lt_evi_slope)
        lines3 += ax3.plot(tx, ty, c='tab:green', lw=1.5, linestyle=(0, (5, 10)), alpha=_trend_alpha(lt_evi_significant),
                            label=f'Slope EVI: {lt_evi_slope:.4f} %/day')

    if lt_pp is not None:
        lines3 += ax3_r.plot(x_filtered, lt_pp, c='tab:blue', lw=1, alpha=0.8, label='lt_pp')
        if pp_slope != default_value and pp_A != default_value:
            tx, ty = _trend_xy(x_filtered[0], x_filtered[-1], pp_A, pp_slope)
            lines3 += ax3_r.plot(tx, ty, c='tab:blue', lw=1.2, linestyle=(0, (5, 10)), alpha=_trend_alpha(pp_significant),
                                  label=f'Slope PP: {pp_slope:.4f} %/day')

    if lt_temp is not None:
        lt_temp_celsius = lt_temp - 273
        lines3 += ax3_r.plot(x_filtered, lt_temp_celsius, c='tab:red', lw=1, alpha=0.8, label='lt_temp (-273)')
        if temp_slope != default_value and temp_A != default_value:
            tx, ty = _trend_xy(x_filtered[0], x_filtered[-1], temp_A, temp_slope)
            lines3 += ax3_r.plot(tx, ty, c='tab:red', lw=1.2, linestyle=(0, (5, 10)), alpha=_trend_alpha(temp_significant),
                                  label=f'Slope TEMP: {temp_slope:.4f} %/day')

    if residuals is not None:
        lines3 += ax3.plot(x_valid, residuals, c='tab:purple', lw=2, label='Residual EVI')
        if residuals_slope != default_value:
            tx, ty = _trend_xy(x_valid[0], x_valid[-1], residuals_A, residuals_slope)
            lines3 += ax3.plot(tx, ty, c='tab:purple', lw=1.5, linestyle=(0, (5, 10)), alpha=_trend_alpha(residuals_significant),
                                label=f'Slope Residual: {residuals_slope:.4f} %/day')

    ax3.set_ylim(0, 10000)
    ax3.set_ylabel('EVI / Residual')
    ax3_r.set_ylabel('PP / TEMP (°C)')
    ax3.set_title('Detrend')
    ax3.legend(lines3, [l.get_label() for l in lines3], loc='upper left', fontsize=8)

    tick_positions, tick_labels = _month_year_ticks(first_date, x_filtered[0], x_filtered[-1])
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

    fig.suptitle(f'{location_label} — days since {first_date}')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
