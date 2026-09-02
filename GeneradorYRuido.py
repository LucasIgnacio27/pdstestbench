# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:41:38 2026

@author: Betterparts
"""

def snr( vmax=1 , dc=0 , ff=1 , ph=0 , nn=1000 , fs=1000,snr = 10):

        tt=np.arange(0,nn,1)
        xx = vmax * np.sin(2*np.pi*ff*tt/fs + ph) + dc
        pot_sen = np.mean(xx**2)
        pot_ruido =  10**(-snr/10) 
        pot_ruido = pot_ruido *pot_sen
        desvio = np.sqrt(pot_ruido)
        ruido = np.random.normal(0,desvio,nn)
        xx = xx + ruido
        
        return tt , xx
    
tt,xx= gen.senoidal(ff=4)
x = np.fft.fft(xx)
plt.plot(tt,np.imag(x))


tt,xx = gen.snr(ff=20,snr = 10)
x = np.fft.fft(xx)
plt.plot(tt,np.imag(x))