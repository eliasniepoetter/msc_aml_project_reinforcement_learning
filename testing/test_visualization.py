from environment.visualization import FlightGearVisualizer
import time

def test_visualization():
    visualizer = FlightGearVisualizer()
    for i in range(10,1000):
        visualizer.render_state(i)
        print(i)
        time.sleep(0.01)
    visualizer.close()