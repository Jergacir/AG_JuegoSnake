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

# Configuración del Algoritmo Genético - MEJORADO
TAMAÑO_POBLACION = 100
GENERACIONES = 100
TASA_MUTACION = 0.2  # Aumentada para romper patrones subóptimos
TASA_CROSSOVER = 0.8  # Aumentada para mejor mezcla genética
ELITISMO = 5  # Más individuos élite

# Configuración de la Red Neuronal
INPUTS = 10  # 4 peligros + 4 direcciones binarias + 2 distancias continuas
HIDDEN = 32  # Aumentado para mayor capacidad de aprendizaje
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
        """Obtiene información del entorno para la red neuronal - MEJORADO"""
        head = self.body[0]
        
        def check_danger(direction, max_distance=5):
            """Verifica peligro hasta max_distance celdas adelante"""
            for dist in range(1, max_distance + 1):
                check_pos = head + direction * dist
                # Pared
                if check_pos.x < 0 or check_pos.x >= ANCHO/TAMAÑO_CELDA or \
                   check_pos.y < 0 or check_pos.y >= ALTO/TAMAÑO_CELDA:
                    return 1.0 / dist  # Más cerca = más peligroso
                # Cuerpo
                if check_pos in self.body:
                    return 1.0 / dist
            return 0.0
        
        # Peligros en 4 direcciones (mira hasta 5 celdas adelante para mejor anticipación)
        danger_up = check_danger(Vector2(0, -1), 5)
        danger_down = check_danger(Vector2(0, 1), 5)
        danger_left = check_danger(Vector2(-1, 0), 5)
        danger_right = check_danger(Vector2(1, 0), 5)
        
        # Distancia normalizada (continua) a la manzana
        dx = (apple_pos.x - head.x) / (ANCHO/TAMAÑO_CELDA)
        dy = (apple_pos.y - head.y) / (ALTO/TAMAÑO_CELDA)
        
        # Dirección binaria (como antes)
        apple_up = 1.0 if dy < 0 else 0.0
        apple_down = 1.0 if dy > 0 else 0.0
        apple_left = 1.0 if dx < 0 else 0.0
        apple_right = 1.0 if dx > 0 else 0.0
        
        # NUEVO: Distancias continuas normalizadas (-1 a 1)
        # Esto le dice a la red "qué tan lejos" está la manzana
        normalized_dx = np.clip(dx, -1.0, 1.0)
        normalized_dy = np.clip(dy, -1.0, 1.0)
        
        return [danger_up, danger_down, danger_left, danger_right,
                apple_up, apple_down, apple_left, apple_right,
                normalized_dx, normalized_dy]  # +2 inputs


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
        self.best_score = 0
        
        for _ in range(population_size):
            nn = NeuralNetwork()
            self.population.append({
                'weights': nn.get_weights(),
                'fitness': 0
            })
    
    def evaluate_genome(self, genome, render=False):
        """Evalúa un genoma jugando Snake con fitness mejorado para eficiencia"""
        snake = Snake()
        apple = Apple()
        nn = NeuralNetwork(genome['weights'])
        
        score = 0
        max_steps = 120  # Balance entre eficiencia y aprendizaje
        
        clock = pygame.time.Clock()
        position_history = []
        distance_history = []
        direction_changes = 0  # NUEVO: Contador de cambios de dirección
        last_direction = snake.direction.copy()
        stuck_counter = 0  # NUEVO: Contador para detectar loops persistentes
        min_distance_to_apple = float('inf')  # NUEVO: Rastrea si se aleja de la manzana
        
        while True:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return genome['fitness'], 0
                
                WIN.fill(COLOR_FONDO)
                snake.draw(WIN)
                apple.draw(WIN)
                
                # Información principal
                text = FONT.render(f"Generación: {self.generation}", True, (255, 255, 255))
                WIN.blit(text, (10, 10))
                
                score_text = FONT.render(f"Score Actual: {score}", True, (255, 255, 255))
                WIN.blit(score_text, (10, 40))
                
                best_score_text = FONT.render(f"Mejor Score: {self.best_score}", True, (184, 149, 22))
                WIN.blit(best_score_text, (10, 70))
                
                fitness_text = FONT_SMALL.render(f"Mejor Fitness: {int(self.best_fitness)}", True, (10, 10, 10))
                WIN.blit(fitness_text, (10, 100))
                
                # NUEVO: Mostrar eficiencia
                if score > 0:
                    efficiency = snake.steps / score
                    eff_text = FONT_SMALL.render(f"Pasos/Manzana: {int(efficiency)}", True, (10, 10, 10))
                    WIN.blit(eff_text, (10, 120))
                
                pygame.display.update()
                clock.tick(50)
            
            vision = snake.get_vision(apple.pos)
            output = nn.forward(vision)
            
            direction_idx = np.argmax(output)
            directions = [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]
            new_direction = directions[direction_idx]
            
            # NUEVO: Detectar cambios de dirección
            if new_direction != last_direction and new_direction + last_direction != Vector2(0, 0):
                direction_changes += 1
            last_direction = new_direction.copy()
            
            snake.set_direction(new_direction)
            
            # Mover
            snake.move()
            
            # Distancia a la manzana
            current_distance = abs(snake.body[0].x - apple.pos.x) + abs(snake.body[0].y - apple.pos.y)
            distance_history.append(current_distance)
            if len(distance_history) > 10:
                distance_history.pop(0)
            
            # NUEVO: Detectar si se aleja continuamente
            if current_distance < min_distance_to_apple:
                min_distance_to_apple = current_distance
                stuck_counter = 0  # Reset si se acerca
            else:
                stuck_counter += 1  # Incrementar si no mejora
            
            position_history.append(snake.body[0].copy())
            if len(position_history) > 20:  # Aumentado de 8 a 20
                position_history.pop(0)
            
            if apple.check_collision(snake):
                score += 1
                max_steps = 120 + score * 60  # Balance entre eficiencia y supervivencia
                position_history.clear()
                distance_history.clear()
                stuck_counter = 0  # Reset al comer
                min_distance_to_apple = float('inf')
            
            if snake.check_collision() or snake.steps_without_food > max_steps:
                break
            
            # ========== DETECCIÓN DE BUCLES MEJORADA ==========
            # SOLO durante entrenamiento (no en visualización)
            if not render:
                # 1. Bucle pequeño: Repite pocas posiciones
                if len(position_history) >= 12:
                    unique_positions = len(set([(p.x, p.y) for p in position_history[-12:]]))
                    if unique_positions < 5:  # Si solo visita 5 posiciones diferentes en 12 pasos
                        break
                
                # 2. Bucle largo: Detectar patrón repetitivo en los últimos 20 pasos
                if len(position_history) == 20:
                    # Comparar primera mitad vs segunda mitad
                    first_half = [(p.x, p.y) for p in position_history[:10]]
                    second_half = [(p.x, p.y) for p in position_history[10:]]
                    
                    # Si hay mucho overlap entre ambas mitades = bucle
                    overlap = len(set(first_half) & set(second_half))
                    if overlap > 7:  # 70% de overlap
                        break
                
                # 3. Estancamiento: No se acerca a la manzana en mucho tiempo
                if stuck_counter > 40:  # 40 pasos sin acercarse (balanceado)
                    break
                
                # 4. Alejamiento continuo: Si tiene historia de distancias
                if len(distance_history) == 10:
                    # Si la distancia promedio aumenta = se está alejando
                    avg_recent = sum(distance_history[-5:]) / 5
                    avg_old = sum(distance_history[:5]) / 5
                    if avg_recent > avg_old + 3:  # Se alejó más de 3 celdas
                        break
        
        # ==================== CÁLCULO DE FITNESS MEJORADO ====================
        
        # 0. PENALIZACIÓN PROGRESIVA POR MUERTE (mientras más score, más grave es morir)
        death_penalty = 0
        if score < 3 and snake.steps < 50:  # Murió muy rápido
            death_penalty = 3000
        elif score < 5 and snake.steps < 100:
            death_penalty = 1500
        elif score >= 10:  # NUEVO: Si ya tenía buen score y murió
            # Penalización que escala con el score alcanzado
            death_penalty = score * 500  # Por cada manzana, 500 puntos de penalización
        elif score >= 5:
            death_penalty = score * 300
        
        # 1. BASE: Manzanas valen MUCHO más
        fitness = score * 10000  # 10x más que antes
        fitness -= death_penalty  # Restar penalización progresiva por muerte
        
        # 2. BONUS POR EFICIENCIA: Recompensar llegar rápido (pero menos que antes)
        if score > 0:
            steps_per_apple = snake.steps / score
            
            # Ideal: menos de 45 pasos por manzana (más permisivo para esquivar)
            if steps_per_apple < 45:
                # BONUS: Reducido para no arriesgar tanto
                efficiency_bonus = (45 - steps_per_apple) * 80  # Antes era 120
                fitness += efficiency_bonus
                print(f"  [Eficiencia ALTA: {steps_per_apple:.1f} pasos/manzana → +{efficiency_bonus:.0f} bonus]", end='')
            else:
                # PENALIZACIÓN: Suavizada
                efficiency_penalty = (steps_per_apple - 45) * 50  # Antes era 80
                fitness -= efficiency_penalty
                print(f"  [Eficiencia BAJA: {steps_per_apple:.1f} pasos/manzana → -{efficiency_penalty:.0f} penalty]", end='')
        
        # 2b. PENALIZACIÓN POR ZIGZAG: Pero balanceada
        if score > 0:
            changes_per_apple = direction_changes / score
            if changes_per_apple > 10:  # Más de 10 giros por manzana = zigzag
                zigzag_penalty = (changes_per_apple - 10) * 150  # Penalización fuerte pero no brutal
                fitness -= zigzag_penalty
                print(f" [ZIGZAG: {changes_per_apple:.1f} giros/manzana → -{zigzag_penalty:.0f}]", end='')
        elif direction_changes > 15:  # Penalizar zigzag incluso sin score
            fitness -= direction_changes * 15
        
        # 3. PENALIZACIÓN POR VAGAR: Si da muchos pasos sin comer
        if snake.steps_without_food > 100 and score == 0:
            wandering_penalty = (snake.steps_without_food - 100) * 5
            fitness -= wandering_penalty
        
        # 4. BONUS por acercarse (solo si no comió)
        if score == 0:
            closeness_bonus = (500 - current_distance * 3)
            fitness += max(0, closeness_bonus)
        
        # 5. BONUS por movimiento variado
        if len(position_history) > 0:
            unique_positions = len(set([(p.x, p.y) for p in position_history]))
            fitness += unique_positions * 3
        
        # 6. Recompensa por supervivencia (escalada con score)
        survival_bonus = snake.steps * 0.3 * (1 + score * 0.15)  # Aumentado: más score = mucho más valor por sobrevivir
        fitness += survival_bonus
        
        # 7. NUEVO: Bonus extra por sobrevivir mucho tiempo con buen score
        if score >= 15:
            longevity_bonus = score * 100  # 100 puntos por cada manzana si sobrevivió bien
            fitness += longevity_bonus
        
        genome['fitness'] = max(0, fitness)
        
        return fitness, score
    
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
        
        max_score_this_gen = 0
        total_efficiency = 0
        count_with_score = 0
        
        for i, genome in enumerate(self.population):
            fitness, score = self.evaluate_genome(genome, render=False)
            max_score_this_gen = max(max_score_this_gen, score)
            
            print(f"\rIndividuo {i+1}/{self.population_size} - Fitness: {fitness:.2f} | Score: {score}", end='')
        
        print()  # Nueva línea después del loop
        
        # Ordenar por fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Actualizar mejor
        if self.population[0]['fitness'] > self.best_fitness:
            self.best_fitness = self.population[0]['fitness']
            self.best_genome = self.population[0]['weights'].copy()
        
        if max_score_this_gen > self.best_score:
            self.best_score = max_score_this_gen
        
        avg_fitness = sum(g['fitness'] for g in self.population) / len(self.population)
        
        print(f"\n📊 RESULTADOS:")
        print(f"   Mejor Fitness: {self.population[0]['fitness']:.2f}")
        print(f"   Fitness Promedio: {avg_fitness:.2f}")
        print(f"   Mejor Score Esta Gen: {max_score_this_gen}")
        print(f"   🏆 Récord de Score: {self.best_score}")
        
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
                'score': self.best_score,
                'generation': self.generation
            }, f)
        print(f"\n💾 Mejor snake guardada - Score: {self.best_score}, Gen: {self.generation}")
    
    def load_best(self, filename='best_snake.pkl'):
        """Carga el mejor genoma"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.best_genome = data['weights']
                self.best_fitness = data['fitness']
                self.best_score = data.get('score', 0)
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
    
    print("\n🐍 SNAKE - ALGORITMO GENÉTICO (OPTIMIZADO PARA EFICIENCIA) 🐍")
    print("="*65)
    print(f"Población: {TAMAÑO_POBLACION}")
    print(f"Generaciones: {GENERACIONES}")
    print(f"Tasa de mutación: {TASA_MUTACION} (aumentada)")
    print(f"Tasa de crossover: {TASA_CROSSOVER} (aumentada)")
    print(f"Elitismo: {ELITISMO}")
    print(f"Arquitectura: {INPUTS}-{HIDDEN}-{OUTPUTS}")
    print("\n🎯 NUEVO: Fitness optimizado para rutas DIRECTAS")
    print("   - Bonus por eficiencia (< 50 pasos/manzana)")
    print("   - Penalización por zigzag (> 50 pasos/manzana)")
    print("   - Manzanas valen 5x más que antes")
    print("="*65)
    
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