from environment.visualization import FlightGearVisualizer
import time
import numpy as np

def test_visualization():
    visualizer = FlightGearVisualizer()
    for i in range(10,1000):
        visualizer.render_state(np.array([0, 0, 100, 0, 0, i]).T)
        print(i)
        time.sleep(0.01)
    visualizer.close()