import pygame
import sys
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString

pygame.init()

def generate_circuit_with_shapely(control_points=None, track_width=85, points=200, scale=1.0, snap_tolerance=1e-6):
    if control_points is None:
        control_points = np.array([
            [794, 732], [1145, 733], [1233, 576], [1500, 508], [1511, 298], [1285, 157], [1239, 306],
            [1055, 287], [833, 207], [642, 151], [515, 254], [489, 395], [357, 387], [169, 415],
            [127, 621], [305, 635], [311, 720], [532, 740], [794, 732]]) * scale

    tck, _ = splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, points, endpoint=False)
    smooth_points = splev(u, tck)
    center_line = np.vstack(smooth_points).T

    center_line_geom = LineString(center_line)

    inner_boundary_geom = center_line_geom.parallel_offset(track_width / 2, 'left', join_style=2)
    outer_boundary_geom = center_line_geom.parallel_offset(track_width / 2, 'right', join_style=2)

    def close_boundary(boundary, tolerance):
        if not boundary.is_ring:
            coords = list(boundary.coords)
            coords.append(coords[0])
            boundary = LineString(coords)
        return boundary

    inner_boundary_geom = close_boundary(inner_boundary_geom, snap_tolerance)
    outer_boundary_geom = close_boundary(outer_boundary_geom, snap_tolerance)

    inner_boundary = np.array(inner_boundary_geom.coords)
    outer_boundary = np.array(outer_boundary_geom.coords)

    distances = np.linalg.norm(np.diff(center_line, axis=0), axis=1)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    return inner_boundary, outer_boundary, center_line, cumulative_distances


def convert_to_pygame_format(boundary):
    return [(int(x), int(y)) for x, y in boundary]


def create_new_circuit(screen):
    """Permite al usuario marcar nuevos puntos de control para un circuito."""
    control_points = []
    creating_circuit = True
    font = pygame.font.Font(None, 30)
    
    while creating_circuit:
        screen.fill((0, 0, 0))
        for point in control_points:
            pygame.draw.circle(screen, (255, 255, 255), point, 5)
        
        text = font.render("Haz clic para marcar puntos. Presiona Enter para terminar.", True, (255, 255, 255))
        screen.blit(text, (20, 20))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                control_points.append(pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and len(control_points) > 2:
                    control_points.append(control_points[0])
                    creating_circuit = False
    
    return confirm_new_circuit(np.array(control_points), screen)


def confirm_new_circuit(control_points, screen):
    """Genera y confirma el nuevo circuito."""
    font = pygame.font.Font(None, 30)
    track_width = 85
    scale = 1.0
    inner, outer, center_line, cumulative_distances = generate_circuit_with_shapely(
        scale=scale, track_width=track_width, control_points=control_points
    )
    
    while True:
        screen.fill((0, 0, 0))
        pygame.draw.lines(screen, (255, 0, 0), True, convert_to_pygame_format(inner), 2)
        pygame.draw.lines(screen, (0, 255, 0), True, convert_to_pygame_format(outer), 2)
        text1 = font.render("Nuevo circuito generado.", True, (255, 255, 255))
        text2 = font.render("Presiona 'Enter' para confirmar o 'R' para rehacer.", True, (255, 255, 255))
        screen.blit(text1, (20, 20))
        screen.blit(text2, (20, 50))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return inner, outer, center_line, cumulative_distances
                if event.key == pygame.K_r:
                    return create_new_circuit(screen)


def show_initial_menu(screen, WIDTH, HEIGHT):
    """Muestra el menú inicial para elegir entre usar el circuito actual o crear uno nuevo."""
    font = pygame.font.Font(None, 50)
    menu_screen = True
    while menu_screen:
        screen.fill((0, 0, 0))
        text1 = font.render("1. Usar el circuito actual", True, (255, 255, 255))
        text2 = font.render("2. Crear un nuevo circuito", True, (255, 255, 255))
        screen.blit(text1, (WIDTH // 3, HEIGHT // 3))
        screen.blit(text2, (WIDTH // 3, HEIGHT // 3 + 50))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return None
                if event.key == pygame.K_2:
                    return create_new_circuit(screen)
