# import different useful stuff
# streamlit for web app development, with multipage framework improvements
# numpy, scipy for computing
# matplotlib for plotting
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from multipage import save, MultiPage, start_app, clear_cache

# recommended start for multipage framework
start_app() #Clears the cache when the app is started

# general configuration of multipage framework
app = MultiPage()
app.start_button = "Let's go!"
app.navbar_name = "Navigation"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"

# defines a starting page for the multipage web app
# not that useful for a deployed app, since it appears to show only the first
# time the app is started 
# in local, the page shows each time the app is executed through > streamlit run foo.py
def startpage():
    st.markdown("""# A multipage streamlit app for basic population dynamics models""")
    st.markdown("""Using the multipage framework by Yan Almeida""")
    st.markdown("""<a href="https://github.com/YanAlmeida/streamlit-multipage-framework">streamlit-multipage-framework</a>""", 
    unsafe_allow_html=True)


# some functions to define models
# logistic model
def m_logistic(etat, t, params):
    x = etat             
    r, K = params    
    xdot = r*x*(1-x/K)    
    return xdot          

# Allee effects model
def m_allee(etat, t, params):
    x = etat             
    r, K, Ka = params    
    xdot = r*x*(x/Ka-1)*(1-x/K)    
    return xdot          

# general parameters for models integration
# related to time
t_0 = 0.0          
t_fin = 20.0      
pas_t = 0.01  
tspan = np.arange(t_0, t_fin, pas_t)
# some x array
x = np.arange(0, 11, 0.1)
# parameters
K = 10.0
r = 1.0
Ka = 1.0
# parameters encapsulation
param_logistic = np.array([r, K])
param_allee = np.array([r, K, Ka])


# definition of a function that computes and displays logistic model and its integration
# to be called later by the multpiage app() function (see bottom of script)
def app1(prev_vars):
    # page title
    st.title("Logistic model")

    # input of the initial density by the user, from a slider
    x0 = st.slider('Initial condition:', min_value = 0.0, max_value = K+2, value = 0.5, step = 0.1)
    etat0 = np.array([x0])

    # model integration
    int_logistic = odeint(m_logistic, etat0, tspan, args=(param_logistic,), hmax=pas_t)

    # creation of the pyplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.plot(x, r*x*(1-x/K), label = '$\dot x = rx(1-x/K)$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(bottom = -.5, top = 3)
    ax1.set_ylabel('time derivative $\dot x$')
    ax1.set_xlabel('population density $x$')

    ax2.plot(tspan, int_logistic, label = '$x(t)$', color = 'C1')
    ax2.grid()
    ax2.legend()
    ax2.set_ylim(bottom = -1, top = 11)
    ax2.set_ylabel('$x(t)$')
    ax2.set_xlabel('time $t$')

    fig.suptitle('Logistic model\n $r = {}, K={}$'.format(r, K))

    # show the pyplot figure in the app
    st.pyplot(fig)

# very similar to app1(), with a second page for the Allee effects model
# to be called later by the multpiage app() function (see bottom of script)
def app2(prev_vars):
    # page title
    st.title("Allee effects model")

    # input of the initial density by the user, from a slider
    x00 = st.slider('Initial condition:', min_value = 0.0, max_value = K+2, value = 0.5, step = 0.1)
    etat0 = np.array([x00])

    # model integration
    int_allee = odeint(m_allee, etat0, tspan, args=(param_allee,), hmax=pas_t)

    # creation of the pyplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.plot(x, r*x*(x/Ka-1)*(1-x/K), label = '$\dot x = rx(x/Ka-1)(1-x/K)$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(bottom = -1, top = 15)
    ax1.set_ylabel('time derivative $\dot x$')
    ax1.set_xlabel('population density $x$')

    ax2.plot(tspan, int_allee, label = '$x(t)$', color = 'C1')
    ax2.grid()
    ax2.legend()
    ax2.set_ylim(bottom = -1, top = 11)
    ax2.set_ylabel('$x(t)$')
    ax2.set_xlabel('time $t$')

    fig.suptitle('Allee effects model\n $r = {}, K={}, K_a={}$'.format(r, K, Ka))

    # show the pyplot figure in the app
    st.pyplot(fig)


# setup of the whole multipage app
# though app.set_initial() and app.add_app() functions of the multipage framework
app.set_initial_page(startpage)
app.add_app("Logistic model", app1)
app.add_app("Allee effects model", app2)
app.run()
