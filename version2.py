import pygame
from pygame.math import Vector2
import random
import math
import numpy as np
import pickle
import os

pygame.init()

# ==================== CONFIGURACIÓN ====================
ANCHO = 800
ALTO = 800
TAMAÑO_CELDA = 20

# Colores
COLOR_FONDO = (20, 20, 30)
COLOR_MANZANA = (255, 50, 50)

# Paleta de colores para diferentes snakes
COLORES_SNAKE = [
    (50, 200, 50),   # Verde
    (50, 150, 255),  # Azul
    (255, 200, 50),  # Amarillo
    (255, 100, 200), # Rosa
    (150, 50, 255),  # Morado
    (255, 150, 50),  # Naranja
    (50, 255, 200),  # Cyan
    (200, 255, 50),  # Lima
    (255, 50, 150),  # Magenta
    (100, 200, 255), # Celeste
]

# Configuración del Algoritmo Genético
TAMAÑO_POBLACION = 100  # Aumentado para más diversidad genética
GENERACIONES = 100
TASA_MUTACION = 0.2  # Aumentada para más exploración y evitar convergencia prematura
TASA_CROSSOVER = 0.8  # Aumentada para mejor mezcla genética
ELITISMO = 5  # Más individuos élite para preservar buenos comportamientos

# Configuración de la Red Neuronal
INPUTS = 8
HIDDEN = 16
OUTPUTS = 4

WIN = pygame.display.set_mode((ANCHO + 300, ALTO))  # +300 para panel lateral
pygame.display.set_caption("Snake GA - Todas las Snakes en un Tablero")
FONT = pygame.font.SysFont("Arial", 16)
FONT_SMALL = pygame.font.SysFont("Arial", 12)


# ==================== RED NEURONAL ====================
class NeuralNetwork:
    """Red neuronal simple con una capa oculta"""
    
    def __init__(self, weights=None):
        if weights is None:
            self.weights_input_hidden = np.random.randn(INPUTS, HIDDEN) * 0.5
            self.bias_hidden = np.random.randn(HIDDEN) * 0.5
            self.weights_hidden_output = np.random.randn(HIDDEN, OUTPUTS) * 0.5
            self.bias_output = np.random.randn(OUTPUTS) * 0.5
        else:
            self.set_weights(weights)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, inputs):
        """Propagación hacia adelante"""
        inputs = np.array(inputs)
        
        # Capa oculta
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = self.tanh(hidden)
        
        # Capa de salida
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        
        return output
    
    def get_weights(self):
        """Obtener todos los pesos como un array plano (cromosoma)"""
        return np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden.flatten(),
            self.weights_hidden_output.flatten(),
            self.bias_output.flatten()
        ])
    
    def set_weights(self, weights):
        """Establecer pesos desde un array plano (cromosoma)"""
        idx = 0
        
        size = INPUTS * HIDDEN
        self.weights_input_hidden = weights[idx:idx+size].reshape(INPUTS, HIDDEN)
        idx += size
        
        size = HIDDEN
        self.bias_hidden = weights[idx:idx+size]
        idx += size
        
        size = HIDDEN * OUTPUTS
        self.weights_hidden_output = weights[idx:idx+size].reshape(HIDDEN, OUTPUTS)
        idx += size
        
        size = OUTPUTS
        self.bias_output = weights[idx:idx+size]


# ==================== JUEGO SNAKE ====================
class Snake:
    def __init__(self, color_index):
        # Posición inicial aleatoria
        start_x = random.randint(5, int(ANCHO/TAMAÑO_CELDA) - 5)
        start_y = random.randint(5, int(ALTO/TAMAÑO_CELDA) - 5)
        self.body = [Vector2(start_x, start_y), Vector2(start_x, start_y+1), Vector2(start_x, start_y+2)]
        self.direction = Vector2(0, -1)
        self.add = False
        self.steps = 0
        self.steps_without_food = 0
        self.alive = True
        self.score = 0
        self.color = COLORES_SNAKE[color_index % len(COLORES_SNAKE)]
        self.id = color_index
    
    def draw(self, surface):
        if not self.alive:
            return
        
        for i, bloque in enumerate(self.body):
            rect = pygame.Rect(bloque.x * TAMAÑO_CELDA, bloque.y * TAMAÑO_CELDA, 
                             TAMAÑO_CELDA-1, TAMAÑO_CELDA-1)  # -1 para separación visual
            
            if i == 0:
                # Cabeza más brillante
                head_color = tuple(min(c + 50, 255) for c in self.color)
                pygame.draw.rect(surface, head_color, rect)
                # Dibuja un punto para identificar la cabeza
                center = rect.center
                pygame.draw.circle(surface, (255, 255, 255), center, 3)
            else:
                pygame.draw.rect(surface, self.color, rect)
    
    def move(self):
        if not self.alive:
            return
            
        if self.add:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy
            self.add = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy
        
        self.steps += 1
        self.steps_without_food += 1
    
    def set_direction(self, new_direction):
        if not self.alive:
            return
        if new_direction + self.direction != Vector2(0, 0):
            self.direction = new_direction
    
    def check_collision(self, all_snakes):
        """Verifica colisiones con paredes, consigo misma y otras snakes"""
        if not self.alive:
            return True
            
        head = self.body[0]
        
        # Colisión con paredes
        if head.x < 0 or head.x >= ANCHO/TAMAÑO_CELDA or \
           head.y < 0 or head.y >= ALTO/TAMAÑO_CELDA:
            self.alive = False
            return True
        
        # Colisión consigo misma
        if head in self.body[1:]:
            self.alive = False
            return True
        
        # Colisión con otras snakes
        for other_snake in all_snakes:
            if other_snake.id != self.id and other_snake.alive:
                if head in other_snake.body:
                    self.alive = False
                    return True
        
        return False
    
    def get_vision(self, apple_pos, all_snakes):
        """Obtiene información del entorno considerando otras snakes"""
        head = self.body[0]
        
        def check_danger(direction, distance=1):
            check_pos = head + direction * distance
            
            # Pared
            if check_pos.x < 0 or check_pos.x >= ANCHO/TAMAÑO_CELDA or \
               check_pos.y < 0 or check_pos.y >= ALTO/TAMAÑO_CELDA:
                return 1.0
            
            # Su propio cuerpo
            if check_pos in self.body:
                return 1.0
            
            # Otras snakes
            for other_snake in all_snakes:
                if other_snake.id != self.id and other_snake.alive:
                    if check_pos in other_snake.body:
                        return 1.0
            
            return 0.0
        
        # Peligros inmediatos en 4 direcciones
        danger_up = check_danger(Vector2(0, -1))
        danger_down = check_danger(Vector2(0, 1))
        danger_left = check_danger(Vector2(-1, 0))
        danger_right = check_danger(Vector2(1, 0))
        
        # Dirección a la manzana (normalizada y con signo)
        dx = (apple_pos.x - head.x) / (ANCHO/TAMAÑO_CELDA)
        dy = (apple_pos.y - head.y) / (ALTO/TAMAÑO_CELDA)
        
        # Dirección binaria a la manzana
        apple_up = 1.0 if dy < -0.1 else 0.0
        apple_down = 1.0 if dy > 0.1 else 0.0
        apple_left = 1.0 if dx < -0.1 else 0.0
        apple_right = 1.0 if dx > 0.1 else 0.0
        
        return [danger_up, danger_down, danger_left, danger_right,
                apple_up, apple_down, apple_left, apple_right]


class Apple:
    def __init__(self):
        self.generate()
        self.timer = 0
    
    def generate(self):
        self.x = random.randint(0, int(ANCHO/TAMAÑO_CELDA) - 1)
        self.y = random.randint(0, int(ALTO/TAMAÑO_CELDA) - 1)
        self.pos = Vector2(self.x, self.y)
    
    def draw(self, surface):
        # Dibujar manzana con efecto pulsante
        pulse = abs(math.sin(self.timer * 0.1)) * 3
        size = TAMAÑO_CELDA + pulse
        offset = -pulse / 2
        rect = pygame.Rect(self.pos.x * TAMAÑO_CELDA + offset, 
                          self.pos.y * TAMAÑO_CELDA + offset,
                          size, size)
        pygame.draw.rect(surface, COLOR_MANZANA, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2)  # Borde blanco
        self.timer += 1
    
    def check_collision(self, snake):
        if not snake.alive:
            return False
            
        if snake.body[0] == self.pos:
            self.generate()
            snake.add = True
            snake.steps_without_food = 0
            snake.score += 1
            return True
        return False


# ==================== GESTOR DE JUEGO COMPARTIDO ====================
class SharedGameManager:
    def __init__(self, population):
        self.snakes = []
        self.networks = []
        self.genomes = population
        self.position_histories = []
        
        for idx, genome in enumerate(population):
            snake = Snake(idx)
            nn = NeuralNetwork(genome['weights'])
            self.snakes.append(snake)
            self.networks.append(nn)
            self.position_histories.append([])
        
        self.apple = Apple()
    
    def update_all(self):
        """Actualiza todas las snakes simultáneamente"""
        alive_count = 0
        
        for idx, (snake, nn) in enumerate(zip(self.snakes, self.networks)):
            if not snake.alive:
                continue
            
            alive_count += 1
            
            # Obtener decisión
            vision = snake.get_vision(self.apple.pos, self.snakes)
            output = nn.forward(vision)
            
            direction_idx = np.argmax(output)
            directions = [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]
            snake.set_direction(directions[direction_idx])
            
            # Mover
            snake.move()
            
            # Historial de posiciones (más largo para mejor detección)
            self.position_histories[idx].append(snake.body[0].copy())
            if len(self.position_histories[idx]) > 16:  # Aumentado de 8 a 16
                self.position_histories[idx].pop(0)
            
            # Colisión con manzana
            if self.apple.check_collision(snake):
                self.position_histories[idx].clear()
            
            # Límite de pasos más estricto
            max_steps = 150 + snake.score * 80  # Reducido para forzar acción
            
            # Verificar colisiones
            if snake.check_collision(self.snakes) or snake.steps_without_food > max_steps:
                snake.alive = False
                continue
            
            # DETECCIÓN DE BUCLES MEJORADA
            if len(self.position_histories[idx]) >= 12:  # Verificar con más historia
                # Contar posiciones únicas
                unique_positions = len(set([(p.x, p.y) for p in self.position_histories[idx]]))
                
                # Si repite muchas posiciones = bucle
                if unique_positions < 6:  # Más estricto (antes era 4)
                    snake.alive = False
                    continue
                
                # Detectar patrón de movimiento repetitivo (ej: arriba-abajo-arriba-abajo)
                if len(self.position_histories[idx]) >= 8:
                    recent_moves = self.position_histories[idx][-8:]
                    # Si las últimas 4 posiciones se repiten en las 4 anteriores
                    if recent_moves[0:4] == recent_moves[4:8]:
                        snake.alive = False
                        continue
        
        return alive_count
    
    def draw_all(self, surface):
        """Dibuja el tablero compartido"""
        # Fondo
        surface.fill(COLOR_FONDO)
        
        # Línea de separación del panel
        pygame.draw.line(surface, (100, 100, 100), (ANCHO, 0), (ANCHO, ALTO), 2)
        
        # Dibujar grid sutil
        for x in range(0, ANCHO, TAMAÑO_CELDA):
            pygame.draw.line(surface, (30, 30, 40), (x, 0), (x, ALTO))
        for y in range(0, ALTO, TAMAÑO_CELDA):
            pygame.draw.line(surface, (30, 30, 40), (0, y), (ANCHO, y))
        
        # Dibujar manzana
        self.apple.draw(surface)
        
        # Dibujar todas las snakes
        for snake in self.snakes:
            snake.draw(surface)
    
    def draw_stats(self, surface, generation, best_fitness, best_score):
        """Dibuja panel de estadísticas lateral"""
        panel_x = ANCHO + 10
        y_offset = 20
        
        # Título
        title = FONT.render("ESTADÍSTICAS", True, (255, 255, 255))
        surface.blit(title, (panel_x, y_offset))
        y_offset += 40
        
        # Generación
        gen_text = FONT_SMALL.render(f"Generación: {generation}", True, (200, 200, 200))
        surface.blit(gen_text, (panel_x, y_offset))
        y_offset += 25
        
        # Vivos
        alive_count = sum(1 for s in self.snakes if s.alive)
        alive_text = FONT_SMALL.render(f"Vivos: {alive_count}/{len(self.snakes)}", True, (100, 255, 100))
        surface.blit(alive_text, (panel_x, y_offset))
        y_offset += 30
        
        # Récord
        record_text = FONT_SMALL.render(f"Récord: {best_score}", True, (255, 200, 100))
        surface.blit(record_text, (panel_x, y_offset))
        y_offset += 35
        
        # Línea separadora
        pygame.draw.line(surface, (100, 100, 100), (panel_x, y_offset), (panel_x + 270, y_offset))
        y_offset += 15
        
        # Ranking de snakes vivas
        ranking_title = FONT_SMALL.render("RANKING:", True, (255, 255, 255))
        surface.blit(ranking_title, (panel_x, y_offset))
        y_offset += 25
        
        # Ordenar snakes por score
        sorted_snakes = sorted(enumerate(self.snakes), key=lambda x: x[1].score, reverse=True)
        
        for rank, (idx, snake) in enumerate(sorted_snakes[:15], 1):  # Top 15
            if snake.alive:
                status = "🟢"
                color = snake.color
            else:
                status = "⚫"
                color = (100, 100, 100)
            
            # Mostrar ID, color, score
            snake_text = FONT_SMALL.render(
                f"{status} #{idx+1:2d} | Score: {snake.score:3d}", 
                True, color
            )
            surface.blit(snake_text, (panel_x, y_offset))
            y_offset += 22
    
    def calculate_fitness(self):
        """Calcula fitness de todas las snakes con penalización agresiva por bucles"""
        for idx, snake in enumerate(self.snakes):
            # Base: Score es lo MÁS importante
            fitness = snake.score * 2000  # Duplicado el valor de las manzanas
            
            # Bonificación pequeña por sobrevivir
            fitness += snake.steps * 0.3
            
            # Si no comió nada, recompensar acercarse
            if snake.score == 0:
                current_distance = abs(snake.body[0].x - self.apple.pos.x) + \
                                 abs(snake.body[0].y - self.apple.pos.y)
                # Recompensa por acercarse
                max_distance = (ANCHO/TAMAÑO_CELDA) + (ALTO/TAMAÑO_CELDA)
                closeness = max_distance - current_distance
                fitness += closeness * 3
            
            # PENALIZACIÓN SEVERA por movimiento repetitivo
            if len(self.position_histories[idx]) > 0:
                unique_positions = len(set([(p.x, p.y) for p in self.position_histories[idx]]))
                total_positions = len(self.position_histories[idx])
                
                if total_positions > 0:
                    diversity_ratio = unique_positions / total_positions
                    
                    # Si la diversidad es muy baja = bucle
                    if diversity_ratio < 0.5:  # Menos del 50% de posiciones únicas
                        fitness *= 0.3  # Penalización SEVERA del 70%
                    elif diversity_ratio < 0.7:
                        fitness *= 0.7  # Penalización media del 30%
                    else:
                        # Recompensa por buena exploración
                        fitness += unique_positions * 5
            
            # BONIFICACIÓN: Tiempo eficiente (menos pasos para mismo score)
            if snake.score > 0:
                efficiency = snake.score / max(snake.steps, 1)
                fitness += efficiency * 500  # Recompensa eficiencia
            
            self.genomes[idx]['fitness'] = max(0, fitness)
    
    def all_dead(self):
        """Verifica si todas las snakes murieron"""
        return all(not snake.alive for snake in self.snakes)


# ==================== ALGORITMO GENÉTICO ====================
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        self.best_score = 0
        
        for _ in range(population_size):
            nn = NeuralNetwork()
            self.population.append({
                'weights': nn.get_weights(),
                'fitness': 0
            })
    
    def selection(self):
        """Selección por torneo"""
        tournament_size = 5
        selected = []
        
        for _ in range(self.population_size - ELITISMO):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Crossover de un punto"""
        if random.random() < TASA_CROSSOVER:
            point = random.randint(0, len(parent1['weights']) - 1)
            child_weights = np.concatenate([
                parent1['weights'][:point],
                parent2['weights'][point:]
            ])
        else:
            child_weights = parent1['weights'].copy()
        
        return {'weights': child_weights, 'fitness': 0}
    
    def mutate(self, genome):
        """Mutación gaussiana"""
        for i in range(len(genome['weights'])):
            if random.random() < TASA_MUTACION:
                genome['weights'][i] += np.random.randn() * 0.5
        return genome
    
    def evolve_visual(self):
        """Ejecuta una generación con visualización compartida"""
        print(f"\n{'='*60}")
        print(f"🐍 Generación {self.generation} 🐍")
        print(f"{'='*60}")
        
        # Crear gestor de juego compartido
        game_manager = SharedGameManager(self.population)
        
        clock = pygame.time.Clock()
        
        # Ejecutar juego hasta que todos mueran
        while not game_manager.all_dead():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            
            # Actualizar todas las snakes
            alive_count = game_manager.update_all()
            
            # Dibujar
            game_manager.draw_all(WIN)
            game_manager.draw_stats(WIN, self.generation, self.best_fitness, self.best_score)
            
            pygame.display.update()
            clock.tick(300)  # Velocidad del juego
        
        # Calcular fitness
        game_manager.calculate_fitness()
        
        # Ordenar por fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Actualizar mejor
        current_best_score = max([s.score for s in game_manager.snakes])
        if current_best_score > self.best_score:
            self.best_score = current_best_score
        
        if self.population[0]['fitness'] > self.best_fitness:
            self.best_fitness = self.population[0]['fitness']
            self.best_genome = self.population[0]['weights'].copy()
        
        # Estadísticas
        avg_fitness = sum(g['fitness'] for g in self.population) / len(self.population)
        avg_score = sum(s.score for s in game_manager.snakes) / len(game_manager.snakes)
        
        print(f"✓ Mejor Fitness: {self.population[0]['fitness']:.2f}")
        print(f"✓ Fitness Promedio: {avg_fitness:.2f}")
        print(f"✓ Mejor Score: {current_best_score}")
        print(f"✓ Score Promedio: {avg_score:.2f}")
        print(f"✓ Récord Histórico: {self.best_score}")
        
        # Crear nueva población
        new_population = self.population[:ELITISMO]
        selected = self.selection()
        
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return True
    
    def save_best(self, filename='best_snake_shared.pkl'):
        """Guarda el mejor genoma"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.best_genome,
                'fitness': self.best_fitness,
                'score': self.best_score,
                'generation': self.generation
            }, f)
        print(f"💾 Guardado: Score {self.best_score}, Gen {self.generation}")
    
    def load_best(self, filename='best_snake_shared.pkl'):
        """Carga el mejor genoma"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.best_genome = data['weights']
                self.best_fitness = data['fitness']
                self.best_score = data.get('score', 0)
                self.generation = data['generation']
            print(f"📂 Cargado: Score {self.best_score}, Gen {self.generation}")
            return True
        return False


# ==================== MAIN ====================
def main():
    ga = GeneticAlgorithm(TAMAÑO_POBLACION)
    ga.load_best()
    
    print("\n" + "="*60)
    print("🐍 SNAKE - ALGORITMO GENÉTICO (TABLERO COMPARTIDO) 🐍")
    print("="*60)
    print(f"📊 Población: {TAMAÑO_POBLACION} snakes en un tablero")
    print(f"🧠 Arquitectura: {INPUTS}-{HIDDEN}-{OUTPUTS}")
    print(f"🎮 Tablero: {ANCHO}x{ALTO} pixels")
    print(f"🎨 Cada snake tiene su propio color")
    print("="*60)
    
    try:
        for gen in range(GENERACIONES):
            if not ga.evolve_visual():
                break
            
            # Guardar cada 5 generaciones
            if (gen + 1) % 5 == 0:
                ga.save_best()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
    
    ga.save_best()
    print("\n🏆 Entrenamiento completado!")
    print(f"🥇 Mejor score alcanzado: {ga.best_score}")


if __name__ == '__main__':
    main()