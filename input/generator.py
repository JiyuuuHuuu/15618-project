"""generator.py

A random generator of firework schedules.

Usage:
python3 generator.py
"""
import random
import os
import csv

rd_seed = 15618
random.seed(rd_seed)

# Config
# count of output files
num_schedule = 1
# count of rows
num_firework = 256
# particles
t0_low = 0.0
t0_high = 150.0
p0_x_low = 100.0
p0_x_high = 700.0
p0_y_low = 350.0
p0_y_high = 390.0
v0_y_low = -100.0
v0_y_high = -80.0
v0_x_low = -10.0
v0_x_high = 10.0
r_low = 8.0
r_high = 10.0
eh_low = 50.0
eh_high = 200.0
color_low = 0
color_high = 6

def rd(low, high):
    """Generate a random float inclusively.

    Return random float, format as '0.00'.
    """
    x = random.uniform(low, high)
    return f'{x:.2f}'

def create(t0):
    """Create one row in random.

    t_0, p0_x, p0_y, v0_x, v0_y, r, explosion_height, color
    [0.0, 15.0] [200.0, 1000.0] [500.0, 600.0] 0.0 [-100.0, -50.0] [5.0, 10.0] [50.0, 450.0] [0, 5]
    """
    p0_x = rd(p0_x_low, p0_x_high)
    p0_y = rd(p0_y_low, p0_y_high)
    v0_x = rd(v0_x_low, v0_x_high)
    v0_y = rd(v0_y_low, v0_y_high)
    assert v0_y != '0.00'
    r = rd(r_low, r_high)
    eh = rd(eh_low, eh_high)
    color = random.randint(color_low, color_high)
    return [t0, p0_x, p0_y, v0_x, v0_y, r, eh, color]

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(num_schedule):
        # to reserve the s0\d{2}.csv for preset scenes
        # e.g. $ rm s1*.csv # to remove generated files
        i += 100
        file = f's{i:03}.csv'
        with open(os.path.join(dir_path, file), 'w') as f:
            writer = csv.writer(f)
            t0s = [rd(t0_low, t0_high) for _ in range(num_firework)]
            t0s.sort(key=float)
            for j in range(num_firework):
                writer.writerow(create(t0s[j]))
    print('###### Random schedule files generated ######')
