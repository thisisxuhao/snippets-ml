import numpy as np
import matplotlib.pyplot as plt

heart_theta = np.linspace(0, 2*np.pi, 1000 * 10)
heart_rho = 1 - np.cos(heart_theta)
heart_x = heart_rho * np.sin(heart_theta)
heart_y = heart_rho * np.cos(heart_theta)
plt.plot(heart_x, heart_y, color='red')

plt.show()
