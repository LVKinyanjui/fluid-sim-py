# Real-Time Fluid Dynamics Renderer in Python
This repository contains a Python implementation of modelling real-time fluid dynamics based on the following [paper](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf).

## Installation
To install, press the green "Code" button, download, and extract the zip. Then, navigate to the "dist" folder and run the main.exe file.

## Examples
The simulation accurately predicts real world phenomena such as the [Kármán vortex street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street),

<img width="424" alt="vortex street" src="https://github.com/user-attachments/assets/f3239c8e-b90b-4a94-b4eb-8e1daea9d0f0">

why aerodynamic [airfoils](https://en.wikipedia.org/wiki/NACA_airfoil) generate lift for aircraft and birds,

<img width="395" alt="image" src="https://github.com/user-attachments/assets/0c9f1cf7-23e8-4971-bb4a-057de31113c6">

and the transition from laminar to turbulent one might find from burning [incense](https://en.wikipedia.org/wiki/Incense) (actually the [motivation](https://en.wikipedia.org/wiki/Boswellia_papyrifera) for this project).

<img width="214" alt="smoke" src="https://github.com/user-attachments/assets/8c57c418-2179-417d-8feb-e6bd3eed7257">

## Future
I would like to port this code to C++ using OpenGL or SFML to get higher performance, explore different CFD schemes ([LBM](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods), [DNS](https://en.wikipedia.org/wiki/Direct_numerical_simulation), and [RANS](https://en.wikipedia.org/wiki/Reynolds-averaged_Navier%E2%80%93Stokes_equations)), and add more user customization (colors, solution stepping, drawing boundaries, density sources).
