import os
import sys
import numpy as np
import pygame
import torch
import torch.nn as nn
import sys
import random
from deap import base, creator, tools

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import utils.constants as cte
from utils.circuit_creation import generate_circuit_with_shapely, convert_to_pygame_format, show_initial_menu

pygame.init()
screen = pygame.display.set_mode((cte.WIDTH, cte.HEIGHT))
pygame.display.set_caption("Simulación de Coches Genéticos (Diversidad y Mejor Coche Azul)")
clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def line_intersection(p1, p2, p3, p4):
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    def det(a, b):
        return a[0]*b[1] - a[1]*b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    if (min(p1[0], p2[0]) - 0.01 <= x <= max(p1[0], p2[0]) + 0.01 and
        min(p1[1], p2[1]) - 0.01 <= y <= max(p1[1], p2[1]) + 0.01 and
        min(p3[0], p4[0]) - 0.01 <= x <= max(p3[0], p4[0]) + 0.01 and
        min(p3[1], p4[1]) - 0.01 <= y <= max(p3[1], p4[1]) + 0.01):
        return [x, y]
    else:
        return None

def build_spatial_grid(boundaries):
    grid = {}
    for boundary in boundaries:
        boundary_segments = [(boundary[i], boundary[(i + 1) % len(boundary)]) for i in range(len(boundary))]
        for segment in boundary_segments:
            x_values = [segment[0][0], segment[1][0]]
            y_values = [segment[0][1], segment[1][1]]
            min_cell_x = int(min(x_values) // cte.CELL_SIZE)
            max_cell_x = int(max(x_values) // cte.CELL_SIZE)
            min_cell_y = int(min(y_values) // cte.CELL_SIZE)
            max_cell_y = int(max(y_values) // cte.CELL_SIZE)
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    grid.setdefault((cell_x, cell_y), []).append(segment)
    return grid

def get_nearby_segments(x, y, grid):
    cell_x, cell_y = int(x // cte.CELL_SIZE), int(y // cte.CELL_SIZE)
    nearby_segments = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = (cell_x + dx, cell_y + dy)
            nearby_segments.extend(grid.get(cell, []))
    return nearby_segments

class Car:
    def __init__(self, x, y, angle, network):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = cte.INITIAL_SPEED
        self.color = cte.RED
        self.alive = True
        self.finished = False

        self.distance_traveled = 0
        self.progress = 0
        self.progress_checkpoint = 0
        self.time_since_last_progress = 0
        self.last_index = 0

        self.network = network

        self.laps_completed = 0
        self.lap_times = []
        self.previous_signed_distance = None
        self.total_time = 0.0
        self.lap_start_time = 0.0
        self.last_distances = []
        self.last_sensor_data = []

    def kill(self):
        self.alive = False
        self.color = cte.GRAY

    def finish(self):
        self.finished = True
        self.kill()

    def calculate_distances(self, grid):
        if not self.alive or self.finished:
            return [], []
        max_distance = 200
        distances = []
        sensor_data = []
        sensor_angles = [-60, -30, -15, 0, 15, 30, 60]
        nearby_segments = get_nearby_segments(self.x, self.y, grid)
        for sensor_angle in sensor_angles:
            angle = self.angle + np.radians(sensor_angle)
            sensor_start = np.array([self.x, self.y])
            sensor_end = sensor_start + max_distance * np.array([np.cos(angle), np.sin(angle)])
            min_dist = max_distance
            closest_point = sensor_end
            for segment in nearby_segments:
                p1, p2 = segment
                intersect = line_intersection(sensor_start, sensor_end, p1, p2)
                if intersect is not None:
                    dist = np.linalg.norm(np.array(intersect) - sensor_start)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersect
            distances.append(min_dist / max_distance)
            sensor_data.append((sensor_start, closest_point))
        return distances, sensor_data

    def move(self):
        if not self.alive or self.finished or not self.last_distances:
            return
        distances = torch.tensor(self.last_distances, dtype=torch.float32).to(device)
        output = self.network(distances).detach().cpu().numpy()

        throttle = output[0]
        steering_normalized = output[1]

        acceleration = throttle * cte.MAX_ACCEL
        acceleration -= cte.FRICTION * self.velocity
        self.velocity += acceleration * cte.DELTA_TIME

        if self.velocity < cte.MIN_SPEED:
            self.velocity = cte.MIN_SPEED

        max_steering_angle = cte.BASE_MAX_STEERING / (1 + cte.STEERING_FACTOR * self.velocity)
        steering_change = steering_normalized * max_steering_angle
        self.angle += steering_change

        dx = self.velocity * np.cos(self.angle) * cte.DELTA_TIME
        dy = self.velocity * np.sin(self.angle) * cte.DELTA_TIME
        self.x += dx
        self.y += dy
        self.distance_traveled += np.hypot(dx, dy)
        self.total_time += cte.DELTA_TIME

    def check_collision(self, grid):
        if not self.alive or self.finished:
            return
        car_pos = np.array([self.x, self.y])
        nearby_segments = get_nearby_segments(self.x, self.y, grid)
        for segment in nearby_segments:
            p1, p2 = segment
            p1 = np.array(p1)
            p2 = np.array(p2)
            line_vec = p2 - p1
            pnt_vec = car_pos - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue
            line_unitvec = line_vec / line_len
            pnt_vec_scaled = pnt_vec / line_len
            t = np.dot(line_unitvec, pnt_vec_scaled)
            if t < 0.0 or t > 1.0:
                nearest = p1 if np.linalg.norm(car_pos - p1) < np.linalg.norm(car_pos - p2) else p2
            else:
                nearest = p1 + t * line_vec
            dist = np.linalg.norm(car_pos - nearest)
            if dist < 5:
                self.kill()
                return

    def penalize_if_no_progress(self, progress_increment):
        if progress_increment > cte.MIN_PROGRESS_INCREMENT:
            self.time_since_last_progress = 0
            self.progress_checkpoint = self.progress
        else:
            self.time_since_last_progress += 1
            if self.time_since_last_progress > cte.MAX_TIME_SINCE_PROGRESS:
                self.kill()

    def check_progress_requirements(self, iteration):
        if iteration > cte.MAX_ITERATIONS_NEAR_START and self.progress < cte.MIN_PROGRESS_TO_SURVIVE:
            self.kill()

    def update_lap(self, start_finish_line):
        if not self.alive or self.finished:
            return
        current_pos = np.array([self.x, self.y])
        line_start, line_end = start_finish_line
        line_vec = line_end - line_start
        point_vec = current_pos - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)

        if 0 <= proj_length <= line_len:
            cross = np.cross(line_vec, point_vec)
            current_signed_distance = cross
            if self.previous_signed_distance is not None:
                if self.previous_signed_distance <= 0 and current_signed_distance > 0:
                    self.laps_completed += 1
                    self.lap_times.append(self.total_time)
                    self.lap_start_time = self.total_time
                    if self.laps_completed >= 1 and not self.finished:
                        self.finish()
            else:
                self.lap_start_time = self.total_time
            self.previous_signed_distance = current_signed_distance

            if not self.finished and (self.total_time - self.lap_start_time) > (cte.MAX_ITERATIONS_PER_LAP * cte.DELTA_TIME):
                self.kill()

def calculate_progress_vectorized(cars, center_line, cumulative_distances):
    alive_cars = [car for car in cars if car.alive and not car.finished]
    if not alive_cars:
        return

    car_positions = np.array([[car.x, car.y] for car in alive_cars])
    distances = np.linalg.norm(center_line[:, np.newaxis, :] - car_positions[np.newaxis, :, :], axis=2)
    closest_indices = np.argmin(distances, axis=0)

    for idx, car in enumerate(alive_cars):
        last_index = car.last_index % len(cumulative_distances)
        closest_index = closest_indices[idx] % len(cumulative_distances)
        index_diff = (closest_index - last_index + len(center_line)) % len(center_line)
        if 0 < index_diff < len(center_line) / 2:
            progress_increment = cumulative_distances[closest_index] - cumulative_distances[last_index]
            if progress_increment < 0:
                progress_increment += cumulative_distances[-1]
            car.progress += progress_increment
        else:
            progress_increment = 0
        car.last_index = closest_index
        car.penalize_if_no_progress(progress_increment)

class Simulation:
    def __init__(self, circuit, center_line, cumulative_distances, screen):
        self.screen = screen
        self.inner_boundary, self.outer_boundary = circuit
        self.boundaries = [self.inner_boundary, self.outer_boundary]
        self.grid = build_spatial_grid(self.boundaries)
        self.generation = 0
        self.center_line = center_line
        self.cumulative_distances = cumulative_distances
        self.font = pygame.font.Font(None, 30)
        self.best_progress_ever = 0
        self.best_time_ever = None

        self.inner_boundary_pygame = convert_to_pygame_format(self.inner_boundary)
        self.outer_boundary_pygame = convert_to_pygame_format(self.outer_boundary)

        self.circuit_surface = pygame.Surface((cte.WIDTH, cte.HEIGHT), pygame.SRCALPHA)
        self.draw_circuit_once()

        current_dir = os.path.dirname(os.path.realpath(__file__))
        red_car_raw = pygame.image.load(os.path.join(current_dir, "..", "utils", "red_car.png")).convert_alpha()
        blue_car_raw = pygame.image.load(os.path.join(current_dir, "..", "utils", "blue_car.png")).convert_alpha()

        self.red_car_image = pygame.transform.scale(red_car_raw, (cte.CAR_LENGTH, cte.CAR_WIDTH))
        self.blue_car_image = pygame.transform.scale(blue_car_raw, (cte.CAR_LENGTH, cte.CAR_WIDTH))

        self.grey_car_image = self.red_car_image.copy()
        w, h = self.grey_car_image.get_size()
        for x in range(w):
            for y in range(h):
                r, g, b, a = self.grey_car_image.get_at((x, y))
                gray = (r + g + b) // 3
                self.grey_car_image.set_at((x, y), (gray, gray, gray, a))

        start_index = 5
        self.start_pos = self.center_line[start_index]
        dx = self.center_line[(start_index + 1) % len(self.center_line), 0] - self.center_line[start_index, 0]
        dy = self.center_line[(start_index + 1) % len(self.center_line), 1] - self.center_line[start_index, 1]
        self.initial_angle = np.arctan2(dy, dx)

        inner_start = self.inner_boundary[0]
        outer_start = self.outer_boundary[0]
        self.start_finish_line = (np.array(inner_start), np.array(outer_start))

    def draw_circuit_once(self):
        pygame.draw.lines(self.circuit_surface, cte.WHITE, True, self.inner_boundary_pygame, 2)
        pygame.draw.lines(self.circuit_surface, cte.WHITE, True, self.outer_boundary_pygame, 2)
        line_start = self.inner_boundary[0]
        line_end = self.outer_boundary[0]
        pygame.draw.line(self.circuit_surface, cte.YELLOW, line_start.astype(int), line_end.astype(int), 2)

    def draw_info(self):
        if self.best_time_ever is None:
            info_lines = [
                f"Generación: {self.generation}",
                f"Mejor progreso: {self.best_progress_ever:.2f} unidades"
            ]
        else:
            info_lines = [
                f"Generación: {self.generation}",
                f"Mejor tiempo: {self.best_time_ever:.2f}s"
            ]

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, cte.WHITE)
            self.screen.blit(text, (10, 10 + i * 20))

    def render(self, cars):
        self.screen.fill(cte.BLACK)
        self.screen.blit(self.circuit_surface, (0, 0))

        alive_cars = [car for car in cars if car.alive and not car.finished]
        dead_cars = [car for car in cars if not car.alive and not car.finished]
        finished_cars = [car for car in cars if car.finished]

        # Escoger el mejor coche de la generación actual en base a su fitness momentáneo:
        # Criterio: si está terminado (con lap_times), usar 1000 - tiempo, si no, usar progreso.
        best_car = max(cars, key=lambda c: (1000.0 - c.lap_times[-1] if c.finished and c.lap_times else c.progress), default=None)

        # Dibujar sensores del mejor coche si existe y está vivo o finalizado
        if best_car and (best_car.alive or best_car.finished):
            self.draw_sensors(best_car.last_sensor_data)

        # Dibujar coches
        for car in alive_cars:
            if car == best_car:
                chosen_img = self.blue_car_image
            else:
                chosen_img = self.red_car_image
            self.draw_car(car, chosen_img)

        for car in dead_cars:
            self.draw_car(car, self.grey_car_image)

        for car in finished_cars:
            self.draw_car(car, self.grey_car_image)

        self.draw_info()
        pygame.display.flip()

    def draw_car(self, car, image):
        rotated_surface = pygame.transform.rotate(image, -np.degrees(car.angle))
        rect = rotated_surface.get_rect(center=(car.x, car.y))
        self.screen.blit(rotated_surface, rect)

    def draw_sensors(self, sensor_data):
        for start, end in sensor_data:
            pygame.draw.line(self.screen, cte.BLUE, start, end, 1)
            pygame.draw.circle(self.screen, cte.YELLOW, (int(end[0]), int(end[1])), 3)

    def create_network_from_individual(self, individual):
        h = int(round(individual[0]))
        if h < cte.MIN_HIDDEN:
            h = cte.MIN_HIDDEN
        if h > cte.MAX_HIDDEN:
            h = cte.MAX_HIDDEN

        input_hidden_end = 1 + 8*h
        input_hidden_params = individual[1:input_hidden_end]

        hidden_output_end = input_hidden_end + (2*h + 2)
        hidden_output_params = individual[input_hidden_end:hidden_output_end]

        w_ih = np.array(input_hidden_params[0:7*h]).reshape(h, 7)
        b_ih = np.array(input_hidden_params[7*h:8*h])

        w_ho = np.array(hidden_output_params[0:2*h]).reshape(2, h)
        b_ho = np.array(hidden_output_params[2*h:2*h+2])

        layer1 = nn.Linear(7, h)
        layer2 = nn.Linear(h, 2)
        with torch.no_grad():
            layer1.weight.copy_(torch.tensor(w_ih, dtype=torch.float32))
            layer1.bias.copy_(torch.tensor(b_ih, dtype=torch.float32))
            layer2.weight.copy_(torch.tensor(w_ho, dtype=torch.float32))
            layer2.bias.copy_(torch.tensor(b_ho, dtype=torch.float32))

        network = nn.Sequential(layer1, nn.ReLU(), layer2, nn.Tanh()).to(device)
        return network

    def create_cars_from_population(self, population):
        cars = []
        for ind in population:
            net = self.create_network_from_individual(ind)
            car = Car(self.start_pos[0], self.start_pos[1], self.initial_angle, network=net)
            cars.append(car)
        return cars

    def should_end_generation(self, cars):
        finished_cars = [car for car in cars if car.finished]
        alive_cars = [car for car in cars if car.alive and not car.finished]

        if finished_cars and len(finished_cars) >= 2:
            return True
        if len(alive_cars) == 1 and alive_cars[0].finished:
            return True
        return False

    def run_evaluation(self, population):
        cars = self.create_cars_from_population(population)
        iteration_count = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            iteration_count += 1
            alive_cars = [car for car in cars if car.alive and not car.finished]
            if not alive_cars:
                break

            for car in alive_cars:
                distances, sensor_data = car.calculate_distances(self.grid)
                car.last_distances = distances
                car.last_sensor_data = sensor_data

            for car in alive_cars:
                car.move()
                car.check_collision(self.grid)
                car.update_lap(self.start_finish_line)

            calculate_progress_vectorized(cars, self.center_line, self.cumulative_distances)

            for car in alive_cars:
                car.check_progress_requirements(iteration_count)

            if self.should_end_generation(cars):
                break

            self.render(cars)
            clock.tick(60)

        # Asignar fitness
        fitnesses = []
        for car in cars:
            if car.finished and len(car.lap_times) > 0:
                car_fitness = 1000.0 - car.lap_times[-1]
            else:
                car_fitness = car.progress
            fitnesses.append(car_fitness)

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)

        finished_cars = [c for c in cars if c.finished]
        if finished_cars:
            best_finish_time = min(c.total_time for c in finished_cars)
            if self.best_time_ever is None or best_finish_time < self.best_time_ever:
                self.best_time_ever = best_finish_time
        else:
            current_best_progress = max((c.progress for c in cars), default=0)
            if current_best_progress > self.best_progress_ever:
                self.best_progress_ever = current_best_progress

        self.generation += 1

        # Imprimir info de la generación en la terminal
        best_ind = max(population, key=lambda ind: ind.fitness.values[0])
        h = int(round(best_ind[0]))
        if self.best_time_ever is not None:
            print(f"Generación: {self.generation}, Mejor tiempo: {self.best_time_ever:.2f}s, Neuronas: {h}")
        else:
            print(f"Generación: {self.generation}, Mejor progreso: {self.best_progress_ever:.2f}, Neuronas: {h}")

        return population

# Definición DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_gene():
    return [random.uniform(cte.MIN_HIDDEN, cte.MAX_HIDDEN)] + [random.uniform(-1, 1) for _ in range(cte.GENOME_LENGTH - 1)]

toolbox.register("individual", lambda: creator.Individual(init_gene()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)

def custom_mutate(ind, indpb=0.15, sigma=0.3):
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] += random.gauss(0, sigma)
    # Clampeamos la arquitectura
    if ind[0] < cte.MIN_HIDDEN: ind[0] = cte.MIN_HIDDEN
    if ind[0] > cte.MAX_HIDDEN: ind[0] = cte.MAX_HIDDEN
    return (ind,)

toolbox.register("mutate", custom_mutate, indpb=0.15, sigma=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

def introduce_diversity(population, n_random=3):
    # Introduce individuos nuevos (aleatorios) reemplazando algunos con fitness más bajos
    population.sort(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -9999, reverse=True)
    for i in range(len(population)-n_random, len(population)):
        population[i][:] = init_gene()
        population[i].fitness.values = (0.0,)

if __name__ == "__main__":
    new_circuit = show_initial_menu(screen, cte.WIDTH, cte.HEIGHT)
    if new_circuit:
        inner_boundary, outer_boundary, center_line, cumulative_distances = new_circuit
        cumulative_distances = np.cumsum(np.linalg.norm(np.diff(center_line, axis=0), axis=1))
    else:
        inner_boundary, outer_boundary, center_line, cumulative_distances = generate_circuit_with_shapely(
            scale=1, track_width=85
        )

    sim = Simulation((inner_boundary, outer_boundary), center_line, cumulative_distances, screen)

    def evaluate_population(pop):
        return sim.run_evaluation(pop)

    toolbox.register("evaluate", evaluate_population)

    population = toolbox.population(n=25)  # Población más grande para mayor diversidad

    for gen in range(cte.NGEN):
        # Evaluar la población
        population = toolbox.evaluate(population)
        
        # Selección del mejor individuo (elitismo)
        best = tools.selBest(population, k=1)
        
        # Seleccionar el resto de la población
        offspring = toolbox.select(population, len(population)-1)
        offspring = list(map(toolbox.clone, offspring))
        
        offspring.extend(best) # Añadir el mejor coche sin cambios
         
        # Cruzar
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cte.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # Mutar
        for mutant in offspring:
            if random.random() < cte.MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population = offspring

    pygame.quit()
