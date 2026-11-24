import datetime

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def detect_long_term_peaks(lt):


    lt_smooth = savgol_filter(lt,23*5,2)
    lt_peaks_p = find_peaks(lt_smooth, height=None, threshold=None, distance=int(len(lt)*0.4), prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    lt_peaks_n = find_peaks(-lt_smooth, height=None, threshold=None, distance=int(len(lt)*0.4), prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    lt_peaks = np.sort(np.concatenate((lt_peaks_p[0],lt_peaks_n[0],[0],[len(lt)-1]),axis=0))
    removes=1
    while removes>0:
        removes=0
        for i in range(1, len(lt_peaks)-1):
            if lt_peaks[i] - lt_peaks[i-1]< 46 :
                lt_peaks = np.delete(lt_peaks, i)
                removes=1
                break
            try:
                if np.abs(lt[lt_peaks[i]] - lt[lt_peaks[i-1]]) + np.abs(lt[lt_peaks[i+1]] - lt[lt_peaks[i]])  < 500 :
                    lt_peaks = np.delete(lt_peaks, i)
                    removes=1
                    break
            except:
                pass
        if lt_peaks[-1]-lt_peaks[-2]<46:
                lt_peaks = np.delete(lt_peaks, -2)
                removes=1

    return lt_peaks


def detect_seasonal_term_peaks_mode2(st,years,modis_days,start_agro_year_index,start_agro_year_doy,modis_days_original=None):

    if modis_days_original is None:
        modis_days_original=modis_days
   
    amplitudes = []
    
    
    last_index = len(modis_days)

    for i in range(len(start_agro_year_index)-1):
        try:

            if i+22> last_index:
                break


            start_agro_year = start_agro_year_index[i]
            year=modis_days[start_agro_year]['year']

            try:
                start_agro_year_ = modis_days_original.index({'doy': start_agro_year_doy, 'year': year})
            except:
                start_agro_year_=start_agro_year
            
            end_agro_year = start_agro_year +22
                
            offset=start_agro_year_-start_agro_year

            max = np.max(st[start_agro_year:end_agro_year])
            min = np.min(st[start_agro_year:end_agro_year])
            mean = np.mean(st[start_agro_year:end_agro_year])
            amplitude = max-min
            amplitudes.append({'start_agro_year':start_agro_year+offset,
                            'end_agro_year':end_agro_year+offset,
                            'amplitude':amplitude,
                            'max':max,
                            'min':min,
                            'mean':mean,
                            'max_index':np.argmax(st[start_agro_year:end_agro_year])+start_agro_year+offset,
                            'min_index':np.argmin(st[start_agro_year:end_agro_year])+start_agro_year+offset,
                            'mean_index':int(np.mean([start_agro_year,end_agro_year]))+offset,
                            'year':year
                            })

        except Exception as e: 
            pass


    


    return amplitudes


def detect_seasonal_term_peaks_mode3(evi, dates, start_doy_agro):
    """
    Same as detect_seasonal_term_peaks_mode2 but computed from the actual
    acquisition dates instead of a fixed 22-position index (that constant
    assumed the 16-day MODIS composite, ~22-23 obs/year, and broke with the
    Sentinel cadence). Each "agricultural year" spans from the date closest
    to start_doy_agro in one calendar year to the next one, so it works for
    any satellite/sampling cadence.

    evi: 1D array of values at the original acquisition dates (may contain
        NaN), same length as dates.
    dates: list of '%Y%m%d' dates aligned with evi.
    start_doy_agro: day of year (1-366) that marks the start of the agricultural year.
    """
    evi = np.asarray(evi, dtype=np.float64)
    dates_dt = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in dates]
    doys = np.array([d.timetuple().tm_yday for d in dates_dt])
    years = np.array([d.year for d in dates_dt])

    start_agro_year_index = []
    for year in sorted(set(years.tolist())):
        year_indices = np.flatnonzero(years == year)
        if year_indices.size == 0:
            continue
        closest = year_indices[np.argmin(np.abs(doys[year_indices] - start_doy_agro))]
        start_agro_year_index.append(int(closest))
    start_agro_year_index = sorted(start_agro_year_index)

    amplitudes = []
    for i in range(len(start_agro_year_index) - 1):
        start_agro_year = start_agro_year_index[i]
        end_agro_year = start_agro_year_index[i + 1]

        segment = evi[start_agro_year:end_agro_year]
        if segment.size == 0 or np.all(np.isnan(segment)):
            continue

        try:
            max_value = np.nanmax(segment)
            min_value = np.nanmin(segment)
            mean_value = np.nanmean(segment)
            amplitudes.append({
                'start_agro_year': start_agro_year,
                'end_agro_year': end_agro_year,
                'amplitude': max_value - min_value,
                'max': max_value,
                'min': min_value,
                'mean': mean_value,
                'max_index': int(np.nanargmax(segment)) + start_agro_year,
                'min_index': int(np.nanargmin(segment)) + start_agro_year,
                'mean_index': int((start_agro_year + end_agro_year) / 2),
                'year': int(years[start_agro_year]),
            })
        except Exception:
            pass

    return amplitudes
