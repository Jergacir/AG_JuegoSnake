import pygame
from pygame.math import Vector2
import random
import math
import numpy as np
import pickle
import os

pygame.init()

# ==================== CONFIGURACIÓN ====================
ANCHO = 600
ALTO = 600
TAMAÑO_CELDA = 20

# Colores
COLOR_FONDO = (175, 215, 70)
COLOR_SNAKE = (50, 50, 200)
COLOR_CABEZA = (200, 50, 50)
COLOR_MANZANA = (255, 0, 0)

# Configuración del Algoritmo Genético
TAMAÑO_POBLACION = 100
GENERACIONES = 100
TASA_MUTACION = 0.15
TASA_CROSSOVER = 0.7
ELITISMO = 3

# Configuración de la Red Neuronal
INPUTS = 8
HIDDEN = 16
OUTPUTS = 4

WIN = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Snake - Algoritmo Genético")
FONT = pygame.font.SysFont("Arial", 20)
FONT_SMALL = pygame.font.SysFont("Arial", 16)


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
    def __init__(self):
        self.body = [Vector2(10, 10), Vector2(10, 11), Vector2(10, 12)]
        self.direction = Vector2(0, -1)
        self.add = False
        self.steps = 0
        self.steps_without_food = 0
    
    def draw(self, surface):
        for i, bloque in enumerate(self.body):
            rect = pygame.Rect(bloque.x * TAMAÑO_CELDA, bloque.y * TAMAÑO_CELDA, 
                             TAMAÑO_CELDA, TAMAÑO_CELDA)
            if i == 0:
                pygame.draw.rect(surface, COLOR_CABEZA, rect)
            else:
                pygame.draw.rect(surface, COLOR_SNAKE, rect)
    
    def move(self):
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
        if new_direction + self.direction != Vector2(0, 0):
            self.direction = new_direction
    
    def check_collision(self):
        """Verifica si la snake choca con paredes o consigo misma"""
        head = self.body[0]
        
        if head.x < 0 or head.x >= ANCHO/TAMAÑO_CELDA or \
           head.y < 0 or head.y >= ALTO/TAMAÑO_CELDA:
            return True
        
        if head in self.body[1:]:
            return True
        
        return False
    
    def get_vision(self, apple_pos):
        """Obtiene información del entorno para la red neuronal"""
        head = self.body[0]
        
        def check_danger(direction, distance=1):
            check_pos = head + direction * distance
            if check_pos.x < 0 or check_pos.x >= ANCHO/TAMAÑO_CELDA or \
               check_pos.y < 0 or check_pos.y >= ALTO/TAMAÑO_CELDA:
                return 1.0
            if check_pos in self.body:
                return 1.0
            return 0.0
        
        danger_up = check_danger(Vector2(0, -1))
        danger_down = check_danger(Vector2(0, 1))
        danger_left = check_danger(Vector2(-1, 0))
        danger_right = check_danger(Vector2(1, 0))
        
        dx = (apple_pos.x - head.x) / (ANCHO/TAMAÑO_CELDA)
        dy = (apple_pos.y - head.y) / (ALTO/TAMAÑO_CELDA)
        
        apple_up = 1.0 if dy < 0 else 0.0
        apple_down = 1.0 if dy > 0 else 0.0
        apple_left = 1.0 if dx < 0 else 0.0
        apple_right = 1.0 if dx > 0 else 0.0
        
        return [danger_up, danger_down, danger_left, danger_right,
                apple_up, apple_down, apple_left, apple_right]


class Apple:
    def __init__(self):
        self.generate()
    
    def generate(self):
        self.x = random.randint(0, int(ANCHO/TAMAÑO_CELDA) - 1)
        self.y = random.randint(0, int(ALTO/TAMAÑO_CELDA) - 1)
        self.pos = Vector2(self.x, self.y)
    
    def draw(self, surface):
        rect = pygame.Rect(self.pos.x * TAMAÑO_CELDA, self.pos.y * TAMAÑO_CELDA,
                          TAMAÑO_CELDA, TAMAÑO_CELDA)
        pygame.draw.rect(surface, COLOR_MANZANA, rect)
    
    def check_collision(self, snake):
        if snake.body[0] == self.pos:
            self.generate()
            snake.add = True
            snake.steps_without_food = 0
            return True
        return False


# ==================== ALGORITMO GENÉTICO ====================
class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        self.best_score = 0  # NUEVO: Mejor score alcanzado
        
        for _ in range(population_size):
            nn = NeuralNetwork()
            self.population.append({
                'weights': nn.get_weights(),
                'fitness': 0
            })
    
    def evaluate_genome(self, genome, render=False):
        """Evalúa un genoma jugando Snake"""
        snake = Snake()
        apple = Apple()
        nn = NeuralNetwork(genome['weights'])
        
        score = 0
        max_steps = 200
        
        clock = pygame.time.Clock()
        position_history = []
        
        while True:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return genome['fitness']
                
                WIN.fill(COLOR_FONDO)
                snake.draw(WIN)
                apple.draw(WIN)
                
                # Información principal
                text = FONT.render(f"Generación: {self.generation}", True, (255, 255, 255))
                WIN.blit(text, (10, 10))
                
                score_text = FONT.render(f"Score Actual: {score}", True, (255, 255, 255))
                WIN.blit(score_text, (10, 40))
                
                # NUEVO: Mostrar mejor score
                best_score_text = FONT.render(f"Mejor Score: {self.best_score}", True, (184, 149, 22))
                WIN.blit(best_score_text, (10, 70))
                
                # Fitness
                fitness_text = FONT_SMALL.render(f"Mejor Fitness: {int(self.best_fitness)}", True, (10, 10, 10))
                WIN.blit(fitness_text, (10, 100))
                
                pygame.display.update()
                clock.tick(50)
            
            vision = snake.get_vision(apple.pos)
            output = nn.forward(vision)
            
            direction_idx = np.argmax(output)
            directions = [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]
            snake.set_direction(directions[direction_idx])
            
            snake.move()
            
            position_history.append(snake.body[0].copy())
            if len(position_history) > 8:
                position_history.pop(0)
            
            if apple.check_collision(snake):
                score += 1
                max_steps = 200 + score * 100
                position_history.clear()
            
            current_distance = abs(snake.body[0].x - apple.pos.x) + abs(snake.body[0].y - apple.pos.y)
            
            if snake.check_collision() or snake.steps_without_food > max_steps:
                break
            
            if len(position_history) == 8:
                unique_positions = len(set([(p.x, p.y) for p in position_history]))
                if unique_positions < 4:
                    break
        
        # Calcular fitness
        fitness = score * 1000
        fitness += snake.steps * 0.5
        
        if score == 0:
            fitness += (500 - current_distance * 2)
        
        if len(position_history) > 0:
            unique_positions = len(set([(p.x, p.y) for p in position_history]))
            fitness += unique_positions * 2
        
        genome['fitness'] = max(0, fitness)
        
        return fitness, score  # NUEVO: Devolver también el score
    
    def selection(self):
        tournament_size = 5
        selected = []
        
        for _ in range(self.population_size - ELITISMO):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
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
        for i in range(len(genome['weights'])):
            if random.random() < TASA_MUTACION:
                genome['weights'][i] += np.random.randn() * 0.5
        return genome
    
    def evolve(self):
        """Ejecuta una generación completa del algoritmo genético"""
        print(f"\n{'='*50}")
        print(f"Generación {self.generation}")
        print(f"{'='*50}")
        
        max_score_this_gen = 0  # NUEVO: Rastrear mejor score de esta generación
        
        for i, genome in enumerate(self.population):
            fitness, score = self.evaluate_genome(genome, render=False)
            max_score_this_gen = max(max_score_this_gen, score)
            print(f"Individuo {i+1}/{self.population_size} - Fitness: {fitness:.2f} | Score: {score}", end='\r')
        
        # Ordenar por fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Actualizar mejor
        if self.population[0]['fitness'] > self.best_fitness:
            self.best_fitness = self.population[0]['fitness']
            self.best_genome = self.population[0]['weights'].copy()
        
        # NUEVO: Actualizar mejor score
        if max_score_this_gen > self.best_score:
            self.best_score = max_score_this_gen
        
        avg_fitness = sum(g['fitness'] for g in self.population) / len(self.population)
        print(f"\nMejor Fitness: {self.population[0]['fitness']:.2f}")
        print(f"Fitness Promedio: {avg_fitness:.2f}")
        print(f"Mejor Score de Esta Gen: {max_score_this_gen}")  # NUEVO
        print(f"Récord de Score: {self.best_score}")  # NUEVO
        
        # Elitismo
        new_population = self.population[:ELITISMO]
        
        # Selección
        selected = self.selection()
        
        # Crossover y mutación
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def save_best(self, filename='best_snake.pkl'):
        """Guarda el mejor genoma"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.best_genome,
                'fitness': self.best_fitness,
                'score': self.best_score,  # NUEVO
                'generation': self.generation
            }, f)
        print(f"\n✓ Mejor snake guardada - Score: {self.best_score}, Gen: {self.generation}")
    
    def load_best(self, filename='best_snake.pkl'):
        """Carga el mejor genoma"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.best_genome = data['weights']
                self.best_fitness = data['fitness']
                self.best_score = data.get('score', 0)  # NUEVO
                self.generation = data['generation']
            print(f"✓ Snake cargada: Gen {self.generation}, Score {self.best_score}, Fitness {self.best_fitness:.2f}")
            return True
        return False
    
    def play_best(self):
        """Visualiza al mejor individuo jugando"""
        if self.best_genome is None:
            print("No hay mejor genoma para mostrar")
            return
        
        best_genome = {'weights': self.best_genome, 'fitness': 0}
        self.evaluate_genome(best_genome, render=True)


# ==================== MAIN ====================
def main():
    ga = GeneticAlgorithm(TAMAÑO_POBLACION)
    
    ga.load_best()
    
    print("\n🐍 SNAKE - ALGORITMO GENÉTICO 🐍")
    print("================================")
    print(f"Población: {TAMAÑO_POBLACION}")
    print(f"Generaciones: {GENERACIONES}")
    print(f"Tasa de mutación: {TASA_MUTACION}")
    print(f"Arquitectura: {INPUTS}-{HIDDEN}-{OUTPUTS}")
    
    try:
        for gen in range(GENERACIONES):
            ga.evolve()
            
            if (gen + 1) % 5 == 0:
                ga.save_best()
            
            if (gen + 1) % 10 == 0:
                print("\n🎮 Mostrando mejor individuo...")
                ga.play_best()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido")
    
    ga.save_best()
    
    print("\n🏆 Reproduciendo mejor snake...")
    while True:
        ga.play_best()


if __name__ == '__main__':
    main()