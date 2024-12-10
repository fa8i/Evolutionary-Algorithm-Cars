import numpy as np

# Tamaño de ventana
WIDTH, HEIGHT = 1800, 1000

# Parámetros globales
MIN_PROGRESS_INCREMENT = 1.0
MAX_TIME_SINCE_PROGRESS = 100
MAX_ITERATIONS_NEAR_START = 1000
MIN_PROGRESS_TO_SURVIVE = 100.0
MAX_ITERATIONS_PER_LAP = 7000
CELL_SIZE = 100
DELTA_TIME = 0.1
CAR_LENGTH = 40
CAR_WIDTH = 20

# Control de velocidad y dirección
MAX_ACCEL = 4                       # Aceleración máxima cuando la salida de velocidad es 1
FRICTION = 0.1                      # Rozamiento: resta un poco de aceleración según velocidad
BASE_MAX_STEERING = np.radians(15)  # Grados convertidos a radianes
STEERING_FACTOR = 0.15              # A mayor este factor, más reduce el ángulo máximo con la velocidad
MIN_SPEED = 15                      # Velocidad mínima del coche
INITIAL_SPEED = 20.0                # Velocidad inicial de los coches

# Parámetros de la red variable
MIN_HIDDEN = 5
MAX_HIDDEN = 20
GENOME_LENGTH = 203                 # 1 gen para neuronas ocultas + 202 para pesos/bias

# Parámetros de mutación
NGEN = 40
CXPB = 0.8
MUTPB = 0.3

# Colores
BLUE = (0, 200, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
