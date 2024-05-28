import scipy
import click
import numpy as np

from tqdm import tqdm
from utils.ode import ODE
from utils.misc import vis
from functools import partial

@click.command()
@click.option('--cond',                 help='Sample from conditional data distribution', metavar='BOOl', type=bool, default=False, show_default=True)
@click.option('--guidance_scale',       help='Scale of applied guidance', metavar='FLOAT', type=click.FloatRange(min=0, max=100), default=1, show_default=True)
@click.option('--guidance_interval',    help='Interval of applied guidance', metavar='START END', type=click.Tuple([float, float]), default=[-np.inf, np.inf], show_default=True)
def main(**kwargs):
    # Solve the ODE
    t = np.linspace(10, 0.001, 25)
    xt_values = np.linspace(-20, 20, 16)
    model = partial(ODE, cond=kwargs['cond'], guidance_scale=kwargs['guidance_scale'], guidance_interval=kwargs['guidance_interval'])
    
    xall = []
    for xt in tqdm(xt_values, total=16, desc='solving the ODE for each initial condition'):
        x = scipy.integrate.odeint(model, xt, t)
        xall.append(x)
    
    fig = vis(t, xall)
    fig.write_html('plotly_graph.html')
    
if __name__ == "__main__":
    main()