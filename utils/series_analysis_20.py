import pickle
import numpy as np
import utils.dwt_deconstruction as dwt_deconstruction
from utils.plot_pixel_summary import plot_pixel_summary

from utils.peak_detection import  detect_seasonal_term_peaks_mode3
from pygam import LinearGAM, s, te
from sklearn.linear_model import LinearRegression
import datetime
import os
import time
from collections import defaultdict
from contextlib import contextmanager

wt_levels = 5
wt_name = 'db4'#'dmey'

# Optional per-stage profiling inside series_analysis(). Enabled with
# WAVELET_PROFILE=1 in the environment. Meant to run locally with --workers 1:
# each worker process has its own accumulator, so with more than one worker
# the summary is split across processes and cannot be read centrally
# without a Manager.
PROFILE_ENABLED = os.environ.get("WAVELET_PROFILE") == "1"
_PROFILE_STATS = defaultdict(lambda: [0.0, 0])


@contextmanager
def _profile(label):
    if not PROFILE_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        stats = _PROFILE_STATS[label]
        stats[0] += time.perf_counter() - start
        stats[1] += 1


def print_profile_summary():
    if not PROFILE_ENABLED or not _PROFILE_STATS:
        return
    print("=== Profile summary (series_analysis) ===")
    total_all = sum(total for total, _ in _PROFILE_STATS.values())
    for label, (total, count) in sorted(_PROFILE_STATS.items(), key=lambda kv: -kv[1][0]):
        avg_ms = (total / count) * 1000 if count else 0.0
        pct = (total / total_all * 100) if total_all else 0.0
        print(f"{label:24s} total={total:9.3f}s  count={count:7d}  avg={avg_ms:8.3f}ms  {pct:5.1f}%")

# plot_to_screen = True
# plot_to_file = False
# mann_kendall = True
# soil_info = True
# lt_min_period_years = 12
# st_min_period_years = 6



def find_first_one_of_final_chain(arr):
    if arr[-1] == 0:
        return None

    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == 0:
            return i + 1
    return 0


def interpolate_auxiliary_series(series, x_old, x_p_new, nan_values=None):
    if series is None:
        return None

    values = np.asarray(series, dtype=np.float32).copy()
    if nan_values is not None:
        threshold = float(np.min(nan_values))
        values[values <= threshold] = np.nan

    valid_mask = np.isfinite(values)
    if np.sum(valid_mask) < 2:
        return None

    return np.interp(x_p_new, np.asarray(x_old)[valid_mask], values[valid_mask]).astype(np.float32)


def run_detrend_model(lt_evi, lt_pp, lt_temp, x_filtered, default_value, detrend_model="gam_current", debug=False, x=-1, y=-1, new_period=1):
    valid_mask = np.isfinite(lt_evi) & np.isfinite(lt_pp) & np.isfinite(lt_temp)
    if np.sum(valid_mask) < 8:
        return default_value, default_value, default_value, default_value, None, None, default_value, default_value

    y_values = np.asarray(lt_evi, dtype=np.float32)[valid_mask]
    pp_values = np.asarray(lt_pp, dtype=np.float32)[valid_mask]
    temp_values = np.asarray(lt_temp, dtype=np.float32)[valid_mask]
    x_values = np.asarray(x_filtered, dtype=np.float32)[valid_mask]

    # x_values (time) is added as a covariate so that temp/pp do not absorb
    # the secular EVI trend through their own trend (see run_detrend_model
    # notes). In gam_current, s(0)+s(1) ignores this 3rd column without error
    # => same behavior as always for that model.
    X_all = np.column_stack([temp_values, pp_values, x_values])

    detrended_slope_direct = None
    temp_coef = default_value
    pp_coef = default_value
    with _profile(f"detrend_fit_{detrend_model}"):
        if detrend_model == "linear_regression":
            model = LinearRegression()
            model.fit(X_all, y_values)
            y_pred_all = model.predict(X_all)
            # By Frisch-Waugh-Lovell, the coefficient on x_values IS the EVI
            # slope net of the linear temp/pp effect (same *100 convention
            # as get_slope, already expressed per day).
            detrended_slope_direct = float(model.coef_[2]) * 100
            # Local sensitivity of EVI to temp/pp: useful to spot pixels
            # where the detrend fitted something absurd. No simple equivalent
            # in gam_current (splines, not a single coefficient per feature).
            temp_coef = float(model.coef_[0])
            pp_coef = float(model.coef_[1])
        else:
            n_splines = max(4, min(10, len(y_values) - 1))
            model = LinearGAM(
                s(0, n_splines=n_splines) +
                s(1, n_splines=n_splines)
            )
            model.fit(X_all, y_values)
            y_pred_all = model.predict(X_all)
    residuals = y_values - y_pred_all
    with _profile("residuals_senslope"):
        residuals_slope, residuals_A, residuals_B, residuals_porc, _ = get_slope(
            residuals,
            x_values,
            default_value,
            new_period=new_period,
        )

    if detrended_slope_direct is not None:
        residuals_slope = detrended_slope_direct

    return residuals_slope, residuals_A, residuals_B, residuals_porc, residuals, x_values, temp_coef, pp_coef


def series_analysis(
        x_old,
        x_p_new,
        dates,
        dates_periodic,
        evi,
        clouds,
        config,
        type_analysis='point_analysis',
        type_results='raster_slope',
        debug=False,
        x=-1,
        y=-1,
        summer_dates_mask=None,
        pp=None,
        temp=None,
        pp_periodic=None,
        temp_periodic=None,
        pp_lt=None,
        temp_lt=None,
        pp_slope_precomputed=None,
        temp_slope_precomputed=None,
        satellite='sentinel',
        plot_path=None,
        cache_output=None,
        location_label=None,
):

    # Default outcome 
    #####################################################################################
    default_value = -40000
    try:
        _nan_threshold = float(np.min(config.get('nan_values', -20000)))
        dwt_level = int(config.get('level', wt_levels))
        new_period = config.get('new_period', 5)

        if satellite == 'sentinel':

            # Do not process if evi data is not available or not good
            if np.all(evi <= _nan_threshold):
                raise Exception('all EVI discarded')

            # Cloud + Interp with QA Band
            evi_old = evi.copy()
            evi [(clouds > 15) | (clouds <= _nan_threshold)] = np.nan  # Masking based on QA values
            evi [evi<= _nan_threshold] = np.nan
            qa_ok = ~np.isnan(evi)

        elif satellite == 'modis':
            if np.all(evi <= _nan_threshold):
                raise Exception('all EVI discarded')

            # Cloud + Interp with QA Band
            evi_old = evi.copy()

            data = clouds
            data = np.nan_to_num(data, nan=2)
            data = (data.astype(int)) & 0b11
            good = np.asarray((data == 0), dtype='int')

            indices_no_ok = np.where(good == 0)[0]
            indices_ok = np.where(good == 1)[0]

            evi = np.asarray(evi, dtype=np.float32)

            evi[indices_no_ok] = np.nan
            qa_ok = ~np.isnan(evi)


        
    
        with _profile("evi_interp"):
            evi_nonan = evi[~np.isnan(evi)]
            x_old_nonan = np.array(x_old)[~np.isnan(evi)]
            evi_periodic = np.interp(x_p_new, x_old_nonan, evi_nonan)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            #plt.scatter(x_old, evi_old, c='tab:blue', lw=2, label='original series')
            plt.scatter(x_old, evi, c='tab:green', lw=1, label='original series')
            plt.plot(x_p_new, evi_periodic, c='#f005', lw=1,  label='interpolated series')
            plt.xticks(x_p_new, dates_periodic, rotation=45)
            plt.xlabel('Days since ' + dates[0])
            plt.ylabel('EVI')
            plt.title(f'EVI series - Pixel ({x},{y})')
            plt.legend()
            plt.show()


        

        evi_summer_mask, evi_summer = filter_by_dates_sentinel(evi_periodic, dates_periodic, config['summer_crop_season'], dates_doy_mask=summer_dates_mask)
        evi_mean = np.mean(evi_periodic)/10000
        evi_summer_mean = np.nanmean( evi_summer )/10000

        if  'slope' in type_analysis or 'point_analysis' in type_analysis:
            # Search for ruptures
            ############################################################
            # ruptures = detect_ruptures(
            #     evi_int, years, modis_days, start_agro_year_index, config, start_agro_year_doy)
            # last_rupture = ruptures[-1]
            isAgro=False
            data_peaks = detect_seasonal_term_peaks_mode3(evi, dates, config['start_doy_agro'])
            st_max_filtered = [ stp['max'] for stp in data_peaks ]
            st_min_filtered = [ stp['min'] for stp in data_peaks ]
            if (np.nanmean(st_max_filtered) > config['max_evi_threshold'] ) and np.mean(st_min_filtered)< config['min_evi_threshold']:
                isAgro=True  

            last_rupture = {'start':0,'end':len(evi_periodic),'duration':len(evi_periodic),'isAgro':isAgro}

            # Long term trend
            ############################################################
            if ( last_rupture['isAgro'] is True 
                            # At least 8 years of cropLand #(last_rupture['duration']*16/365 > 6 and last_rupture['isAgro'] is True)        #or
                            #(last_rupture['start'] == 0 and last_rupture['isAgro'] is False)        # Natural vegetation
            ):

                start = last_rupture['start']+(last_rupture['duration'] % 2)
                end = last_rupture['end']
                if (start-end) % 2 != 0:
                    end = end - 1
                evi_filtered = evi_periodic[start:end]
                x_filtered = x_p_new[start:end]

                with _profile("evi_dwt"):
                    r = dwt_deconstruction.dwt_deconstruction_all(
                        evi_filtered, wt_name, dwt_level)

                lt_evi=r[0]
                st_evi=np.sum(r[1:],axis=0)

                with _profile("evi_senslope"):
                    lt_evi_slope, lt_A, lt_B, lt_evi_porc, lt_evi_mk = get_slope(
                        lt_evi,
                        x_filtered,
                        default_value,
                        new_period=new_period,
                    )
                # print(lt_evi_slope)
                veg = default_value
                # if crops is not None:
                #     if np.all(np.isin(crops, config['crop_type_natural'])):
                #         veg = 255 # Natural vegetation
                #     elif np.sum(crops == config['crop_type_corn']) >= config['min_corn_years']:
                #         veg = 1 # Corn

                # else:
                st_evi_slope = default_value
                st_A = default_value
                st_B = default_value
                st_evi_porc = default_value
                st_evi_mk = default_value


                pp_slope = default_value
                temp_slope = default_value
                residuals_slope = default_value
                residuals_mean = default_value
                residuals_A = default_value
                residuals_B = default_value
                residuals_porc = default_value
                temp_coef = default_value
                pp_coef = default_value
                _plot_lt_pp = None
                _plot_lt_temp = None
                _plot_residuals = None
                _plot_x_valid = None
                _plot_pp_raw = None
                _plot_temp_raw = None

                if config.get('detrend') and pp is not None and temp is not None:
                    pp_periodic = interpolate_auxiliary_series(pp, x_old, x_p_new, config.get('nan_values'))
                    temp_periodic = interpolate_auxiliary_series(temp, x_old, x_p_new, config.get('nan_values'))
                elif config.get('detrend') and pp_periodic is not None and temp_periodic is not None:
                    pp_periodic = np.asarray(pp_periodic, dtype=np.float32)
                    temp_periodic = np.asarray(temp_periodic, dtype=np.float32)
                else:
                    pp_periodic = None
                    temp_periodic = None

                if pp_periodic is not None and temp_periodic is not None:
                    pp_filtered = pp_periodic[start:end]
                    temp_filtered = temp_periodic[start:end]

                    if np.all(np.isfinite(pp_filtered)) and np.all(np.isfinite(temp_filtered)):
                        if pp_lt is not None and temp_lt is not None and pp_slope_precomputed is not None and temp_slope_precomputed is not None:
                            lt_pp = np.asarray(pp_lt, dtype=np.float32)
                            lt_temp = np.asarray(temp_lt, dtype=np.float32)
                            pp_slope = float(pp_slope_precomputed)
                            temp_slope = float(temp_slope_precomputed)
                        else:
                            with _profile("pp_temp_dwt_senslope_fallback"):
                                r_pp = dwt_deconstruction.dwt_deconstruction_all(pp_filtered, wt_name, dwt_level)
                                lt_pp = r_pp[0]
                                pp_slope, _, _, _, _ = get_slope(lt_pp, x_filtered, default_value, new_period=new_period)

                                r_temp = dwt_deconstruction.dwt_deconstruction_all(temp_filtered, wt_name, dwt_level)
                                lt_temp = r_temp[0]
                                temp_slope, _, _, _, _ = get_slope(lt_temp, x_filtered, default_value, new_period=new_period)

                        _plot_lt_pp = lt_pp
                        _plot_lt_temp = lt_temp
                        _plot_pp_raw = pp_filtered
                        _plot_temp_raw = temp_filtered
                        residuals_slope, residuals_A, residuals_B, residuals_porc, _plot_residuals, _plot_x_valid, temp_coef, pp_coef = run_detrend_model(
                            lt_evi,
                            lt_pp,
                            lt_temp,
                            x_filtered,
                            default_value,
                            detrend_model=config.get('detrend_model', 'gam_current'),
                            debug=debug,
                            x=x,
                            y=y,
                            new_period=new_period,
                        )
                        if residuals_slope > default_value:
                            residuals_mean = float(np.mean(_plot_residuals))

                if plot_path is not None or cache_output is not None:
                    try:
                        pp_A = pp_significant = default_value
                        if _plot_lt_pp is not None and pp_slope != default_value:
                            _, pp_A, _, _, pp_mk = get_slope(_plot_lt_pp, x_filtered, default_value, new_period=new_period)
                            pp_significant = bool(pp_mk is not None and getattr(pp_mk, 'h', False))

                        temp_A = temp_significant = default_value
                        lt_temp_celsius = _plot_lt_temp - 273 if _plot_lt_temp is not None else None
                        if lt_temp_celsius is not None and temp_slope != default_value:
                            _, temp_A, _, _, temp_mk = get_slope(lt_temp_celsius, x_filtered, default_value, new_period=new_period)
                            temp_significant = bool(temp_mk is not None and getattr(temp_mk, 'h', False))

                        residuals_significant = False
                        if _plot_residuals is not None and residuals_slope != default_value:
                            _, _, _, _, residuals_mk = get_slope(_plot_residuals, _plot_x_valid, default_value, new_period=new_period)
                            residuals_significant = bool(residuals_mk is not None and getattr(residuals_mk, 'h', False))

                        plot_kwargs = dict(
                            x_old=x_old, evi=evi, x_p_new=x_p_new, evi_periodic=evi_periodic, x_filtered=x_filtered,
                            lt_evi=lt_evi, lt_evi_slope=lt_evi_slope, lt_A=lt_A,
                            lt_evi_significant=bool(lt_evi_mk is not None and getattr(lt_evi_mk, 'h', False)),
                            lt_pp=_plot_lt_pp, pp_slope=pp_slope, pp_A=pp_A, pp_significant=pp_significant,
                            lt_temp=_plot_lt_temp, temp_slope=temp_slope, temp_A=temp_A, temp_significant=temp_significant,
                            residuals=_plot_residuals, x_valid=_plot_x_valid, residuals_slope=residuals_slope,
                            residuals_A=residuals_A, residuals_significant=residuals_significant,
                            data_peaks=data_peaks,
                            default_value=default_value,
                            location_label=location_label if location_label is not None else f'Pixel ({x},{y})',
                            first_date=dates[0],
                        )

                        if cache_output is not None:
                            # Richer dict than plot_kwargs (which only carries what
                            # plot_pixel_summary() accepts) -- adds what only the data
                            # export needs: raw unmasked EVI + a per-observation QA
                            # flag, and the raw PP/TEMP series (not just their
                            # long-term component).
                            cache_kwargs = dict(
                                plot_kwargs,
                                evi_old=evi_old,
                                qa_ok=qa_ok,
                                pp_raw=_plot_pp_raw,
                                temp_raw=_plot_temp_raw,
                                is_agro=bool(last_rupture['isAgro']),
                                x=x,
                                y=y,
                            )
                            with open(cache_output, 'wb') as f:
                                pickle.dump(cache_kwargs, f)

                        if plot_path is not None:
                            plot_pixel_summary(output_file=f"{plot_path}_summary.jpg", **plot_kwargs)
                    except Exception as _e:
                        print(f"Warning: could not generate the summary plot for pixel ({x},{y}): {_e}")
                
            else:
                x_filtered = default_value
                evi_filtered = default_value
                start = default_value
                end = default_value
                lt_evi = default_value
                st = default_value
                lt_evi_slope = default_value
                lt_A = default_value
                lt_B = default_value
                lt_evi_porc = default_value
                lt_evi_mk = default_value

                st_evi_slope = default_value
                st_A = default_value
                st_B = default_value
                st_evi_porc = default_value
                st_evi_mk = default_value
                veg = default_value

                pp_slope=default_value
                temp_slope=default_value
                residuals_slope=default_value
                residuals_mean=default_value
                temp_coef=default_value
                pp_coef=default_value
                #energy=[default_value for _ in range(wt_levels+1)]
        
    ##################################################################


    except Exception as e:

        if (str(e) != 'all EVI discarded'):
            print("except!", e)
            if debug:
                import traceback
                traceback.print_exc()
                print(f"Pixel ({x},{y}) - No calc, reason: {e}")

            no_calc = 0
        else:
            no_calc = -1  # Will update all defaults to -9999

        evi_mean=default_value+no_calc
        evi_summer_mean=default_value+no_calc
        last_rupture={'isAgro':default_value+no_calc}
        x_filtered=default_value,   # np array with the dates of the continuous-cropping period
        evi_filtered=default_value,  # np array with the EVI of the continuous-cropping period
        start=default_value+no_calc
        end=default_value+no_calc
        lt_evi=default_value+no_calc
        st=default_value+no_calc
        energy=[default_value+no_calc for _ in range(wt_levels+1)]
        lt_evi_slope=default_value+no_calc
        lt_evi_porc=default_value
        lt_A=default_value,
        lt_B=default_value,
    
        st_evi_slope = default_value
        st_A = default_value
        st_B = default_value
        st_evi_porc = default_value
        st_evi_mk = default_value
        veg = default_value+no_calc
        #energy=[default_value for _ in range(wt_levels+1)]
        
        pp_slope=default_value+no_calc
        temp_slope=default_value+no_calc
        residuals_slope=default_value+no_calc
        residuals_mean=default_value+no_calc
        temp_coef=default_value+no_calc
        pp_coef=default_value+no_calc





    
    finally:

        if type_results == 'raster_slope':
            return [
                        lt_evi_slope,
                        #start,
                        #end,
                        float(last_rupture['isAgro']),
                        #lt_evi_porc,
                        float(evi_mean),
                        float(lt_evi_slope * evi_summer_mean) if lt_evi_slope > default_value else default_value,
                        # veg,
                        # st_evi_slope,
                        # st_evi_porc,
                        pp_slope,
                        temp_slope,
                        residuals_slope,
                        residuals_mean,
                        temp_coef,
                        pp_coef,
                    ]

        elif type_results == 'pixel_analysis':    
            return [
                        lt_evi_slope,
                        start,
                        end,
                        last_rupture['isAgro'],
                        lt_evi_porc,
                        evi_mean,
                        lt_evi_slope * evi_summer_mean if lt_evi_slope > default_value else default_value,
                        veg,
                        evi_filtered,
                        x_filtered, 
                        lt_evi
                    ]
# AUX FUNCTIONS ==================================================================================

def filter_by_dates(evi, dates, filter_criteria):

    
    
    evi_filtered = np.asarray(evi, dtype=np.float32)
    dates_doy = np.array([d['doy'] for d in dates])
    if filter_criteria['max']>= filter_criteria['min']:
        dates_doy_mask = (dates_doy >= filter_criteria['min']) & (
            dates_doy <= filter_criteria['max'])
        evi_filtered[~dates_doy_mask] = np.nan
    else:
        dates_doy_mask = (dates_doy <= filter_criteria['min']) & (
            dates_doy >= filter_criteria['max'])
        evi_filtered[~dates_doy_mask] = np.nan
    

    
    return dates_doy_mask,evi_filtered

def filter_by_dates_sentinel(evi, dates, filter_criteria, dates_doy_mask=None):

    evi_filtered = np.asarray(evi, dtype=np.float32)
    if dates_doy_mask is None:
        dates_doy = np.array([datetime.datetime.strptime(d, '%Y%m%d').date().timetuple().tm_yday for d in dates])
        if filter_criteria['max'] >= filter_criteria['min']:
            dates_doy_mask = (dates_doy >= filter_criteria['min']) & (
                dates_doy <= filter_criteria['max'])
        else:
            dates_doy_mask = (dates_doy <= filter_criteria['min']) | (
                dates_doy >= filter_criteria['max'])

    evi_filtered[~dates_doy_mask] = np.nan
    return dates_doy_mask, evi_filtered


def get_slope(evi, x_filtered, default_value, new_period=1):

    try:
        from komanawa.kendall_stats import MannKendall as KomanawaMannKendall

        evi_mk = KomanawaMannKendall(np.asarray(evi, dtype=np.float64))
        slope, intercept, _, _ = evi_mk.calc_senslope()
        # calc_senslope() uses the positional index (0,1,2,...) of the series,
        # not actual days -- since points are spaced every new_period days,
        # the raw slope is "per point" (= per new_period days). Dividing by
        # new_period turns it into an actual per-day rate.
        slope = float(slope) / new_period
        evi_slope = slope * 100

        x_filtered = np.array(x_filtered, dtype=np.float32)
        nans = np.isnan(x_filtered)
        if np.any(nans):
            x_filtered = interp_extrap_respect_valid(x_filtered)

        A = float(intercept)
        B = float(intercept) + slope * (x_filtered[-1] - x_filtered[0])
        evi_porc = (B - A) * 100 / np.abs(A)

    except Exception:
        evi_mk = None
        evi_slope = default_value
        A = default_value
        B = default_value
        evi_porc = default_value

    return evi_slope, A, B, evi_porc, evi_mk


def get_peaks(evi_filtered,
              modis_days,
              config,
              crops,
              veg
          ):
    
    peaks = []
    
    


    for y, c in zip(config['crop_type_years'][config['crop_type_years_filter_start']:config['crop_type_years_filter_end']+1], crops):
        if c == config['crop_type_corn']:
            
            days = [i for i, d in enumerate(modis_days) if d['year'] == y]
            evi_filtered_year = evi_filtered[days]
            peaks.append( { 'value': np.nanmax(evi_filtered_year),
                     'date': modis_days[np.nanargmax(evi_filtered_year)],
                     'index': days[np.nanargmax(evi_filtered_year)],
                     'year':y,
                     'type': c,
                     'veg':veg  }
            )
        else:
            if c in config['crop_type_natural']:
            
                days = [i for i, d in enumerate(modis_days) if d['year'] == y]
                evi_filtered_year = evi_filtered[days]
                peaks.append( { 'value': np.nanmax(evi_filtered_year),
                        'date': modis_days[np.nanargmax(evi_filtered_year)],
                        'index': days[np.nanargmax(evi_filtered_year)],
                        'year':y,
                        'type': c,
                        'veg':veg  }
                )
                         
    return peaks



def interp_extrap_respect_valid(x_filtered, index=None):
    """
    x_filtered: array-like of floats and NaN
    index: optional array-like of the same length, interpolation axis (e.g. times)
    Returns a copy with:
      (1) interior interpolation,
      (2) linear extrapolation at the edges,
      (3) the originally valid values restored.
    """
    y = np.asarray(x_filtered, dtype=float).copy()
    n = y.size

    if index is None:
        idx = np.arange(n, dtype=float)
    else:
        idx = np.asarray(index, dtype=float)
        if idx.size != n:
            raise ValueError("index must have the same length as x_filtered")

    valid = ~np.isnan(y)
    if not valid.any():
        # no data to drive either interpolation or extrapolation
        return y

    # valid indices and values
    iv = np.flatnonzero(valid)
    xv = idx[valid]
    yv = y[valid]

    first, last = iv[0], iv[-1]

    # --- (1) Interpolation ONLY between the first and last valid value ---
    span_mask = (np.arange(n) >= first) & (np.arange(n) <= last)
    y[span_mask] = np.interp(idx[span_mask], xv, yv)

    # --- (2) Linear extrapolation at the edges ---
    if yv.size >= 2:
        # left (before the first valid value)
        if first > 0:
            m_left = (yv[1] - yv[0]) / (xv[1] - xv[0])
            y[:first] = yv[0] + m_left * (idx[:first] - xv[0])

        # right (after the last valid value)
        if last < n - 1:
            m_right = (yv[-1] - yv[-2]) / (xv[-1] - xv[-2])
            y[last+1:] = yv[-1] + m_right * (idx[last+1:] - xv[-1])
    else:
        # Only one valid point: constant extrapolation (no slope available)
        y[:first] = yv[0]
        y[last+1:] = yv[0]

    # --- (3) Restore the originally valid values ---
    y[valid] = np.asarray(x_filtered, dtype=float)[valid]

    return y