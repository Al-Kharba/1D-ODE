import scipy
import numpy as np

# s(t) = 1
def s(t):
    return 1

# ds/dt = 0
def s_derivative(t, h=1e-5):
    return (s(t + h) - s(t - h)) / (2. * h)

# sigma(t) = t
def sigma(t):
    return t

# dsigma/dt = 1
def sigma_derivative(t, h=1e-5):
    return (sigma(t + h) - sigma(t - h)) / (2. * h)

def gauss_pdf(x, mu, std):
    return 1/(std * np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-mu)**2/std**2)

def data_pdf(x_0):
    return 0.5*(gauss_pdf(x_0, 2, 0.5)+gauss_pdf(x_0, -2, 0.5))

def data_cond_pdf(x_0):
    return 0.5*(gauss_pdf(x_0, 2.5, 0.25)+gauss_pdf(x_0, 1.5, 0.25))

def Denoiser(x_t, sigma, cond=False):
    data_pdf_func = data_pdf if not cond else data_cond_pdf
    nominator = scipy.integrate.quad(lambda x_0: data_pdf_func(x_0)*gauss_pdf(x_t, x_0, sigma)*x_0, -4, 4)[0]
    denominator = scipy.integrate.quad(lambda x_0: data_pdf_func(x_0)*gauss_pdf(x_t, x_0, sigma), -4, 4)[0]
    return nominator / (denominator + 1e-6)

def ODE(x_t, t, cond=False, guidance_scale=1, guidance_interval=(-np.inf, np.inf)):
    if not cond:
        x_0 = Denoiser(x_t/s(t), sigma(t), cond=False)
    else:
        if guidance_scale == 1:
            x_0 = Denoiser(x_t/s(t), sigma(t), cond=True)
        else:
            if t > guidance_interval[0] and t <= guidance_interval[1]:
                x_0 = Denoiser(x_t/s(t), sigma(t), cond=True) * guidance_scale + (1 - guidance_scale) * Denoiser(x_t/s(t), sigma(t), cond=False)
            else:
                x_0 = Denoiser(x_t/s(t), sigma(t), cond=True)
        
    score = (x_0 - x_t) / sigma(t)**2
    dxdt = s_derivative(t) / s(t) * x_t - s(t)**2 * sigma_derivative(t) * sigma(t) * score
    return dxdt
