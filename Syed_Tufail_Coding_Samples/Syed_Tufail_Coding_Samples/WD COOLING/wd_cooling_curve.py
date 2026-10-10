import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import FuncFormatter, FixedLocator

# Using a seaborn style for a clean look
plt.style.use('seaborn-v0_8-whitegrid')

# Discrete data points: cooling time (in Gyr) vs. luminosity (in L/L☉)
time_gyr = np.array([0, 2, 6, 10, 12])
luminosities = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

# We do log10 because luminosity changes a lot (factor of 10 each step)
log_luminosities = np.log10(luminosities)

# PCHIP helps in smoothness without big jumps
pchip = PchipInterpolator(time_gyr, log_luminosities)

# Make more points between 0 and 12 for a nicer curve
time_fine = np.linspace(0, 12, 500)
log_l_fine = pchip(time_fine) # For Interpolating the log-luminosities at these fine times


# Setting up a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the interpolated log-luminosity curve
ax.plot(time_fine, log_l_fine, color='#2a5a99', linewidth=2.5,
        label='0.6 $M_\\odot$ White Dwarf')

# Limit x-axis to 0–12, y-axis to -5.2–-1
ax.set_xlim(0, 12)
ax.set_ylim(-5.2, -1)

# Labels for x and y axes, plus a title
ax.set_xlabel('Cooling Time (Gyr)', fontsize=12)
ax.set_ylabel(r'$\log(L/L_\odot)$', fontsize=12)
ax.set_title('White Dwarf Cooling Curve', fontsize=14)

# Make y-axis ticks at integers: -5, -4, -3, -2, -1
ax.yaxis.set_major_locator(FixedLocator(np.arange(-5, 0, 1)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

# Small texts on the plot to show stages of cooling
ax.text(0.5, -2, 'Rapid Cooling', ha='left', va='bottom', fontsize=10)
ax.text(7, -4, 'Gradual Decline', ha='left', va='top', fontsize=10)
ax.text(10.5, -4.7, 'Crystallization', ha='right', va='bottom', fontsize=10)

# Turn on minor ticks, add a faint grid for them
ax.tick_params(axis='both', which='major', labelsize=10)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# Adjusting layout so nothing is cut off
plt.tight_layout()
plt.show()

