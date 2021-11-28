import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from multipage import save, MultiPage, start_app, clear_cache


start_app() #Clears the cache when the app is started

app = MultiPage()
app.start_button = "Let's go!"
app.navbar_name = "Navigation"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"

def startpage():
    st.markdown("""# A multipage streamlit app for basic population dynamics models""")
    st.markdown("""Using the multipage framework by Yan Almeida""")
    st.markdown("""<a href="https://github.com/YanAlmeida/streamlit-multipage-framework">streamlit-multipage-framework</a>""", 
    unsafe_allow_html=True)


# definition du modele logistique
def m_logistic(etat, t, params):
    x = etat             
    r, K = params    
    xdot = r*x*(1-x/K)    
    return xdot          

# definition du modele avec effets Allee
def m_allee(etat, t, params):
    x = etat             
    r, K, Ka = params    
    xdot = r*x*(x/Ka-1)*(1-x/K)    
    return xdot          


#general variables
# temps d'intégration
t_0 = 0.0          
t_fin = 20.0      
pas_t = 0.01  
tspan = np.arange(t_0, t_fin, pas_t)
# some x array
x = np.arange(0, 11, 0.1)
K = 10.0
r = 1.0
Ka = 1.0
# encapsulation parametres
param_logistic = np.array([r, K])
param_allee = np.array([r, K, Ka])

def app1(prev_vars):
    st.title("Logistic model")

    # densité initiale de la population
    x0 = st.slider('Initial condition:', min_value = 0.0, max_value = K+2, value = 0.5, step = 0.1)
    etat0 = np.array([x0])

    # intégration du modèle logistique
    int_logistic = odeint(m_logistic, etat0, tspan, args=(param_logistic,), hmax=pas_t)

    # creation of the pyplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.plot(x, r*x*(1-x/K), label = '$\dot x = rx(1-x/K)$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(bottom = -.5, top = 3)
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


def app2(prev_vars):
    ###############################################################
    # plot the logistic equation in pyplot with interaction on param
    st.title("Allee effects model")

    # densité initiale de la population
    x00 = st.slider('Initial condition:', min_value = 0.0, max_value = K+2, value = 0.5, step = 0.1)

    # encapsulation de la densité initiale
    etat0 = np.array([x00])

    # intégration du modèle
    int_allee = odeint(m_allee, etat0, tspan, args=(param_allee,), hmax=pas_t)

    # creation of the pyplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.plot(x, r*x*(x/Ka-1)*(1-x/K), label = '$\dot x = rx(x/Ka-1)(1-x/K)$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(bottom = -1, top = 15)
    ax1.set_ylabel('dérivée $\dot x$')
    ax1.set_xlabel('densité de population $x$')

    ax2.plot(tspan, int_allee, label = '$x(t)$', color = 'C1')
    ax2.grid()
    ax2.legend()
    ax2.set_ylim(bottom = -1, top = 11)
    ax2.set_ylabel('$x(t)$')
    ax2.set_xlabel('temps $t$')

    fig.suptitle('Allee effects model\n $r = {}, K={}, K_a={}$'.format(r, K, Ka))

    # show the pyplot figure in the app
    st.pyplot(fig)


# setup of the whole multipage app
app.set_initial_page(startpage)
app.add_app("Logistic model", app1)
app.add_app("Allee effects model", app2)
app.run()
