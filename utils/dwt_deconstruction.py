import pywt  
import numpy as np  



def dwt_deconstruction(data,wavelet_name, levels, all=False):   
    
    
    n = len(data)
    if n%2==1:
        n=n+1

    wl = pywt.Wavelet(wavelet_name)

    coeff_all = pywt.wavedec(data, wl,  mode='symmetric',level=levels)

    r=np.zeros((levels+1,n))

    for l in range(levels+1):
        aux = coeff_all.copy()
        for c in range(len(coeff_all)):
            if c!=l:
                aux[c]=coeff_all[c]*0
        r[l,:] = pywt.waverec(aux, wavelet= wl)
            

    lt=r[0]
    phenology=np.sum(r[[1,2,3,4,5],:],axis=0)
    return lt,phenology


def dwt_deconstruction_all(data,wavelet_name, levels):   
    
    
    n = len(data)
    if n%2==1:
        n=n+1

    wl = pywt.Wavelet(wavelet_name)

    coeff_all = pywt.wavedec(data, wl,  mode='symmetric',level=levels)

    r=np.zeros((levels+1,n))

    for l in range(levels+1):
        aux = coeff_all.copy()
        for c in range(len(coeff_all)):
            if c!=l:
                aux[c]=coeff_all[c]*0
        r[l,:] = pywt.waverec(aux, wavelet= wl)
            

    return r