import scipy
import numpy as np
import plotly.graph_objects as go

from utils.ode import *
from plotly.subplots import make_subplots


def vis(t, xall):
    # grid for 
    Z_grid = np.zeros((30, 11))
    for i in range(Z_grid.shape[0]):
        for j in range(Z_grid.shape[1]):
            x_t = i - 15
            sigma = j
            if j == 0:
                Z_grid[i, j] = data_pdf(x_t)
            else:
                Z_grid[i, j] = scipy.integrate.quad(lambda x_0: data_pdf(x_0)*gauss_pdf(x_t, x_0, sigma), -15, 15)[0]
    for j in range(Z_grid.shape[1]):
        Z_grid[:, j] /= np.max(Z_grid[:, j])
    
    x0 = np.linspace(-4, 4, 100)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('PDFs', 'ODE'))
    
    # data PDF
    fig.add_trace(go.Scatter(x=data_pdf(x0), y=x0, mode='lines', line=dict(color='orange'), name='Unconditional data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_pdf(x0), y=x0, fill='tozerox', mode='none', fillcolor='rgba(255, 165, 0, 0.25)', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_cond_pdf(x0), y=x0, mode='lines', line=dict(color='green'), name='Conditional data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_cond_pdf(x0), y=x0, fill='tozerox', mode='none', fillcolor='rgba(0, 128, 0, 0.25)', showlegend=False), row=1, col=1)

    fig.update_xaxes(range=[0, 0.5], tickvals=[], autorange='reversed', showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title=dict(text='x', font=dict(size=12, weight='bold')),
                     range=[-15, 15], tickvals=[], showgrid=False, zeroline=False, row=1, col=1)

    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, xref="x domain", yref="y domain",
                  fillcolor="white", opacity=1, layer="below", line_width=0, row=1, col=1)
    
    # ODE trajectories
    for i, trajectory in enumerate(xall):
        fig.add_trace(go.Scatter(x=t, y=trajectory[:, 0], mode='lines', line=dict(color='black', width=0.2), showlegend=False), row=1, col=2)

    fig.update_xaxes(title=dict(text='$\sigma$', font=dict(size=12, weight='bold')),
                     range=[0, 10], showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[-15, 15], showgrid=False, zeroline=False, row=1, col=2)

    fig.add_trace(go.Heatmap(x=np.linspace(0, 10, 11), y=np.linspace(-15, 15, 31), z=Z_grid, showscale=False, connectgaps=True, zsmooth='best', colorscale = [[0, 'rgb(254, 245, 236)'], [1, 'rgb(234, 106, 38)']]), row=1, col=2)
    fig.update_layout(legend=dict(x=-0.1))
    
    return fig
