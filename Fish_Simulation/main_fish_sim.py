"""Port of main.m"""
import time
from Fish_Simulation import setup_var, get_controls, setup_sys, simulate_sys, fem_to_inertial, fish_animation

def main():
    # Setup Script
    t_start = time.time()

    # Setup Variables
    tf = 15.0        # final time
    x_init = -0.5     # initial x position
    y_init = 0.0       # initial y position
    t_pause = 0.01     # time to pause animation for

    # Setup Variables
    p = setup_var(tf, x_init, y_init, t_pause)
    p.controller = get_controls

    # Setup System
    p = setup_sys(p)

    # Simulate System
    Xstates = simulate_sys(p)
    Xinertial = fem_to_inertial(p, Xstates)

    print(f"Elapsed time: {time.time() - t_start:.2f} s")

    # Plot
    fish_animation(p, Xinertial, save_path=None, animation=False)

    return p, Xstates, Xinertial


if __name__ == "__main__":
    main()
