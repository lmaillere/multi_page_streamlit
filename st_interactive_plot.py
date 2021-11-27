import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

###############################################################
# plot the logistic equation in pyplot with interaction on param
st.title("Plot of the logistic model")

st.write("This simple app takes parameters from the user and plots the corresponding logistic model, and its numerical integration.")

# some x array
x = np.arange(0, 11, 0.1)

# parameter K is fixed, parameter r through user input via a streamlit slider
K = 10.0
r = st.slider('Enter the Intrinsic growth rate (r):',min_value=0.2, max_value=2.0, step=0.1)

#############################
# example with an integration
# from the Biomaths code / mam3

# densité initiale de la population
x0 = st.number_input('Initial condition:', min_value = 0.0, max_value = K+2, value = 0.5, step = 0.1)

# encapsulation de la densité initiale
etat0 = np.array([x0])

# encapsulation parametres
param_logistic = np.array([r, K])

# temps d'intégration
t_0 = 0.0          
t_fin = 20.0      
pas_t = 0.01  
tspan = np.arange(t_0, t_fin, pas_t)

def m_logistic(etat, t, params):
    x = etat             
    r, K = params    
    xdot = r*x*(1-x/K)    
    return xdot          

# intégration du modèle
int_logistic = odeint(m_logistic, etat0, tspan, args=(param_logistic,), hmax=pas_t)


# creation of the pyplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.plot(x, r*x*(1-x/K), label = '$\dot x = rx(1-x/K)$')
ax1.grid()
ax1.legend()
ax1.set_ylim(bottom = -1, top = 11)
ax1.set_ylabel('dérivée $\dot x$')
ax1.set_xlabel('densité de population $x$')

ax2.plot(tspan, int_logistic, label = '$x(t)$', color = 'C1')
ax2.grid()
ax2.legend()
ax2.set_ylim(bottom = -1, top = 11)
ax2.set_ylabel('$x(t)$')
ax2.set_xlabel('temps $t$')

fig.suptitle('Modèle logistique\n $r = {}, K={}$'.format(r, K))

# show the pyplot figure in the app
st.pyplot(fig)

