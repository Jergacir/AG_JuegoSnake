import pygame
from pygame.math import Vector2
import random
import numpy as np
import pickle
import os

pygame.init()

# ==================== CONFIGURACIÓN ====================
ANCHO_TABLERO = 400
ALTO_TABLERO = 400
TAMAÑO_CELDA = 20
ESPACIO_CENTRO = 100  # Espacio entre los dos tableros

# Ventana total
ANCHO_VENTANA = ANCHO_TABLERO * 2 + ESPACIO_CENTRO + 40  # +40 para márgenes
ALTO_VENTANA = ALTO_TABLERO + 200  # +200 para títulos y estadísticas

# Colores
COLOR_FONDO = (20, 20, 30)
COLOR_FONDO_TABLERO_IA = (30, 20, 40)  # Morado oscuro para IA
COLOR_FONDO_TABLERO_HUMANO = (20, 40, 30)  # Verde oscuro para humano
COLOR_SNAKE_IA = (150, 50, 255)  # Morado
COLOR_CABEZA_IA = (200, 100, 255)
COLOR_SNAKE_HUMANO = (50, 200, 100)  # Verde
COLOR_CABEZA_HUMANO = (100, 255, 150)
COLOR_MANZANA = (255, 50, 50)

# Configuración de la Red Neuronal
INPUTS = 10  # 4 peligros + 4 direcciones binarias + 2 distancias continuas
HIDDEN = 32  # Igual que main_2.py
OUTPUTS = 4

WIN = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption("Snake: HUMANO vs IA 🧠")
FONT_TITLE = pygame.font.SysFont("Arial", 28, bold=True)
FONT = pygame.font.SysFont("Arial", 20)
FONT_SMALL = pygame.font.SysFont("Arial", 16)


# ==================== RED NEURONAL ====================
class NeuralNetwork:
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
        inputs = np.array(inputs)
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = self.tanh(hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        return output
    
    def set_weights(self, weights):
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
    def __init__(self, is_ai=False):
        start_x = 10
        start_y = 10
        self.body = [Vector2(start_x, start_y), Vector2(start_x, start_y+1), Vector2(start_x, start_y+2)]
        self.direction = Vector2(0, -1)
        self.add = False
        self.score = 0
        self.alive = True
        self.is_ai = is_ai
    
    def draw(self, surface, offset_x, offset_y):
        if not self.alive:
            return
        
        color_body = COLOR_SNAKE_IA if self.is_ai else COLOR_SNAKE_HUMANO
        color_head = COLOR_CABEZA_IA if self.is_ai else COLOR_CABEZA_HUMANO
        
        for i, bloque in enumerate(self.body):
            x = offset_x + int(bloque.x * TAMAÑO_CELDA)
            y = offset_y + int(bloque.y * TAMAÑO_CELDA)
            rect = pygame.Rect(x, y, TAMAÑO_CELDA-1, TAMAÑO_CELDA-1)
            
            if i == 0:
                pygame.draw.rect(surface, color_head, rect)
                # Punto blanco en la cabeza
                center = rect.center
                pygame.draw.circle(surface, (255, 255, 255), center, 3)
            else:
                pygame.draw.rect(surface, color_body, rect)
    
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
    
    def set_direction(self, new_direction):
        if not self.alive:
            return
        # Evitar dirección opuesta
        if new_direction + self.direction != Vector2(0, 0):
            self.direction = new_direction
    
    def check_collision(self):
        if not self.alive:
            return True
            
        head = self.body[0]
        
        # Colisión con paredes
        if head.x < 0 or head.x >= ANCHO_TABLERO/TAMAÑO_CELDA or \
           head.y < 0 or head.y >= ALTO_TABLERO/TAMAÑO_CELDA:
            self.alive = False
            return True
        
        # Colisión consigo misma
        if head in self.body[1:]:
            self.alive = False
            return True
        
        return False
    
    def get_vision(self, apple_pos):
        """Obtiene información del entorno para la red neuronal - IGUAL QUE MAIN_2"""
        head = self.body[0]
        
        def check_danger(direction, max_distance=5):
            """Verifica peligro hasta max_distance celdas adelante"""
            for dist in range(1, max_distance + 1):
                check_pos = head + direction * dist
                # Pared
                if check_pos.x < 0 or check_pos.x >= ANCHO_TABLERO/TAMAÑO_CELDA or \
                   check_pos.y < 0 or check_pos.y >= ALTO_TABLERO/TAMAÑO_CELDA:
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
        dx = (apple_pos.x - head.x) / (ANCHO_TABLERO/TAMAÑO_CELDA)
        dy = (apple_pos.y - head.y) / (ALTO_TABLERO/TAMAÑO_CELDA)
        
        # Dirección binaria (como antes)
        apple_up = 1.0 if dy < 0 else 0.0
        apple_down = 1.0 if dy > 0 else 0.0
        apple_left = 1.0 if dx < 0 else 0.0
        apple_right = 1.0 if dx > 0 else 0.0
        
        # Distancias continuas normalizadas (-1 a 1)
        normalized_dx = np.clip(dx, -1.0, 1.0)
        normalized_dy = np.clip(dy, -1.0, 1.0)
        
        return [danger_up, danger_down, danger_left, danger_right,
                apple_up, apple_down, apple_left, apple_right,
                normalized_dx, normalized_dy]  # +2 inputs = 10 total


class Apple:
    def __init__(self):
        self.generate()
    
    def generate(self):
        self.x = random.randint(0, int(ANCHO_TABLERO/TAMAÑO_CELDA) - 1)
        self.y = random.randint(0, int(ALTO_TABLERO/TAMAÑO_CELDA) - 1)
        self.pos = Vector2(self.x, self.y)
    
    def draw(self, surface, offset_x, offset_y):
        x = offset_x + int(self.pos.x * TAMAÑO_CELDA)
        y = offset_y + int(self.pos.y * TAMAÑO_CELDA)
        rect = pygame.Rect(x, y, TAMAÑO_CELDA, TAMAÑO_CELDA)
        pygame.draw.rect(surface, COLOR_MANZANA, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2)
    
    def check_collision(self, snake):
        if not snake.alive:
            return False
            
        if snake.body[0] == self.pos:
            self.generate()
            snake.add = True
            snake.score += 1
            return True
        return False


# ==================== JUEGO DE COMPARACIÓN ====================
class ComparisonGame:
    def __init__(self, ai_weights):
        # Snake IA (izquierda)
        self.snake_ai = Snake(is_ai=True)
        self.apple_ai = Apple()
        self.nn = NeuralNetwork(ai_weights)
        
        # Snake Humano (derecha)
        self.snake_human = Snake(is_ai=False)
        self.apple_human = Apple()
        
        self.game_over = False
        self.winner = None
    
    def update_ai(self):
        """Actualiza la snake de IA"""
        if not self.snake_ai.alive:
            return
        
        # Obtener visión y decisión
        vision = self.snake_ai.get_vision(self.apple_ai.pos)
        output = self.nn.forward(vision)
        
        direction_idx = np.argmax(output)
        directions = [Vector2(0, -1), Vector2(0, 1), Vector2(-1, 0), Vector2(1, 0)]
        self.snake_ai.set_direction(directions[direction_idx])
        
        # Mover
        self.snake_ai.move()
        
        # Verificar colisiones
        self.apple_ai.check_collision(self.snake_ai)
        self.snake_ai.check_collision()
    
    def update_human(self):
        """Actualiza la snake humana"""
        if not self.snake_human.alive:
            return
        
        # Mover
        self.snake_human.move()
        
        # Verificar colisiones
        self.apple_human.check_collision(self.snake_human)
        self.snake_human.check_collision()
    
    def handle_input(self, event):
        """Maneja input del teclado para el humano"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.snake_human.direction.y != 1:
                self.snake_human.set_direction(Vector2(0, -1))
            elif event.key == pygame.K_DOWN and self.snake_human.direction.y != -1:
                self.snake_human.set_direction(Vector2(0, 1))
            elif event.key == pygame.K_LEFT and self.snake_human.direction.x != 1:
                self.snake_human.set_direction(Vector2(-1, 0))
            elif event.key == pygame.K_RIGHT and self.snake_human.direction.x != -1:
                self.snake_human.set_direction(Vector2(1, 0))
    
    def check_game_over(self):
        """Verifica si el juego terminó - El juego continúa hasta determinar ganador"""
        # Si ambos están muertos, comparar scores
        if not self.snake_ai.alive and not self.snake_human.alive:
            self.game_over = True
            if self.snake_ai.score > self.snake_human.score:
                self.winner = "IA"
            elif self.snake_human.score > self.snake_ai.score:
                self.winner = "HUMANO"
            else:
                self.winner = "EMPATE"
        
        # Si solo el humano murió, verificar si la IA ya ganó o debe seguir
        elif not self.snake_human.alive and self.snake_ai.alive:
            if self.snake_ai.score > self.snake_human.score:
                # IA ya superó al humano, puede terminar
                self.game_over = True
                self.winner = "IA"
            # Si no ha superado, el juego continúa (IA sigue jugando)
        
        # Si solo la IA murió, verificar si el humano ya ganó o debe seguir
        elif not self.snake_ai.alive and self.snake_human.alive:
            if self.snake_human.score > self.snake_ai.score:
                # Humano ya superó a la IA, puede terminar
                self.game_over = True
                self.winner = "HUMANO"
            # Si no ha superado, el juego continúa (humano sigue jugando)
    
    def draw(self, surface):
        """Dibuja ambos tableros y estadísticas"""
        surface.fill(COLOR_FONDO)
        
        # Offset para tablero IA (izquierda)
        offset_ai_x = 20
        offset_ai_y = 80
        
        # Offset para tablero humano (derecha)
        offset_human_x = ANCHO_TABLERO + ESPACIO_CENTRO + 20
        offset_human_y = 80
        
        # Dibujar fondos de tableros
        pygame.draw.rect(surface, COLOR_FONDO_TABLERO_IA, 
                        (offset_ai_x, offset_ai_y, ANCHO_TABLERO, ALTO_TABLERO))
        pygame.draw.rect(surface, COLOR_FONDO_TABLERO_HUMANO,
                        (offset_human_x, offset_human_y, ANCHO_TABLERO, ALTO_TABLERO))
        
        # Bordes
        pygame.draw.rect(surface, (150, 50, 255), 
                        (offset_ai_x, offset_ai_y, ANCHO_TABLERO, ALTO_TABLERO), 3)
        pygame.draw.rect(surface, (50, 200, 100),
                        (offset_human_x, offset_human_y, ANCHO_TABLERO, ALTO_TABLERO), 3)
        
        # Dibujar grids
        for x in range(0, ANCHO_TABLERO, TAMAÑO_CELDA):
            pygame.draw.line(surface, (40, 40, 50), 
                           (offset_ai_x + x, offset_ai_y), 
                           (offset_ai_x + x, offset_ai_y + ALTO_TABLERO))
            pygame.draw.line(surface, (40, 50, 40),
                           (offset_human_x + x, offset_human_y),
                           (offset_human_x + x, offset_human_y + ALTO_TABLERO))
        
        for y in range(0, ALTO_TABLERO, TAMAÑO_CELDA):
            pygame.draw.line(surface, (40, 40, 50),
                           (offset_ai_x, offset_ai_y + y),
                           (offset_ai_x + ANCHO_TABLERO, offset_ai_y + y))
            pygame.draw.line(surface, (40, 50, 40),
                           (offset_human_x, offset_human_y + y),
                           (offset_human_x + ANCHO_TABLERO, offset_human_y + y))
        
        # Dibujar manzanas y snakes
        self.apple_ai.draw(surface, offset_ai_x, offset_ai_y)
        self.snake_ai.draw(surface, offset_ai_x, offset_ai_y)
        
        self.apple_human.draw(surface, offset_human_x, offset_human_y)
        self.snake_human.draw(surface, offset_human_x, offset_human_y)
        
        # Títulos
        title_ai = FONT_TITLE.render("🤖 IA", True, (200, 100, 255))
        title_human = FONT_TITLE.render("👤 HUMANO", True, (100, 255, 150))
        
        surface.blit(title_ai, (offset_ai_x + ANCHO_TABLERO//2 - title_ai.get_width()//2, 30))
        surface.blit(title_human, (offset_human_x + ANCHO_TABLERO//2 - title_human.get_width()//2, 30))
        
        # Scores
        score_ai = FONT.render(f"Score: {self.snake_ai.score}", True, (255, 255, 255))
        score_human = FONT.render(f"Score: {self.snake_human.score}", True, (255, 255, 255))
        
        surface.blit(score_ai, (offset_ai_x + 10, offset_ai_y + ALTO_TABLERO + 20))
        surface.blit(score_human, (offset_human_x + 10, offset_human_y + ALTO_TABLERO + 20))
        
        # Estado
        if not self.snake_ai.alive:
            dead_text = FONT_SMALL.render("MUERTA", True, (255, 100, 100))
            surface.blit(dead_text, (offset_ai_x + 10, offset_ai_y + ALTO_TABLERO + 50))
        else:
            alive_text = FONT_SMALL.render("VIVA", True, (100, 255, 100))
            surface.blit(alive_text, (offset_ai_x + 10, offset_ai_y + ALTO_TABLERO + 50))
        
        if not self.snake_human.alive:
            dead_text = FONT_SMALL.render("MUERTO", True, (255, 100, 100))
            surface.blit(dead_text, (offset_human_x + 10, offset_human_y + ALTO_TABLERO + 50))
        else:
            alive_text = FONT_SMALL.render("VIVO", True, (100, 255, 100))
            surface.blit(alive_text, (offset_human_x + 10, offset_human_y + ALTO_TABLERO + 50))
        
        # Controles
        controls = FONT_SMALL.render("Usa las flechas del teclado ⬆️⬇️⬅️➡️", True, (200, 200, 200))
        surface.blit(controls, (ANCHO_VENTANA//2 - controls.get_width()//2, ALTO_VENTANA - 30))
        
        # Game Over
        if self.game_over:
            # Overlay semi-transparente
            overlay = pygame.Surface((ANCHO_VENTANA, ALTO_VENTANA))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            surface.blit(overlay, (0, 0))
            
            # Mensaje de ganador
            if self.winner == "IA":
                winner_text = FONT_TITLE.render("🤖 ¡LA IA GANA! 🤖", True, (200, 100, 255))
            elif self.winner == "HUMANO":
                winner_text = FONT_TITLE.render("👤 ¡GANASTE! 🎉", True, (100, 255, 150))
            else:
                winner_text = FONT_TITLE.render("⚖️ EMPATE ⚖️", True, (255, 255, 100))
            
            surface.blit(winner_text, 
                        (ANCHO_VENTANA//2 - winner_text.get_width()//2, ALTO_VENTANA//2 - 80))
            
            # Scores finales
            final_scores = FONT.render(
                f"IA: {self.snake_ai.score}  vs  HUMANO: {self.snake_human.score}",
                True, (255, 255, 255)
            )
            surface.blit(final_scores,
                        (ANCHO_VENTANA//2 - final_scores.get_width()//2, ALTO_VENTANA//2 - 20))
            
            # Instrucción para reiniciar
            restart_text = FONT_SMALL.render("Presiona ESPACIO para jugar de nuevo o ESC para salir",
                                            True, (200, 200, 200))
            surface.blit(restart_text,
                        (ANCHO_VENTANA//2 - restart_text.get_width()//2, ALTO_VENTANA//2 + 40))


# ==================== FUNCIÓN PRINCIPAL ====================
def load_ai_weights(filename='best_snake.pkl'):
    """Carga los pesos de la mejor IA"""
    if not os.path.exists(filename):
        print(f"❌ Error: No se encontró el archivo '{filename}'")
        print("   Ejecuta primero el entrenamiento para generar la IA.")
        return None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(f"✓ IA cargada exitosamente")
            print(f"  - Generación: {data['generation']}")
            print(f"  - Score récord: {data.get('score', 'N/A')}")
            return data['weights']
    except Exception as e:
        print(f"❌ Error al cargar la IA: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("🐍 SNAKE: HUMANO vs IA 🧠")
    print("="*60)
    
    # Cargar IA
    ai_weights = load_ai_weights()
    if ai_weights is None:
        print("\n⚠️  No se puede iniciar sin una IA entrenada.")
        print("   Ejecuta primero el script de entrenamiento.")
        return
    
    print("\n🎮 Controles:")
    print("   - Flechas del teclado: Mover tu snake")
    print("   - ESPACIO: Reiniciar partida")
    print("   - ESC: Salir")
    print("\n¡Buena suerte! 🍀")
    print("="*60 + "\n")
    
    clock = pygame.time.Clock()
    game = ComparisonGame(ai_weights)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Input del humano
            game.handle_input(event)
            
            # Reiniciar o salir en game over
            if game.game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game = ComparisonGame(ai_weights)
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not game.game_over:
            # Actualizar ambas snakes
            game.update_ai()
            game.update_human()
            game.check_game_over()
        
        # Dibujar
        game.draw(WIN)
        pygame.display.update()
        clock.tick(10)  # Velocidad del juego (ajustable)
    
    pygame.quit()
    print("\n👋 ¡Gracias por jugar!")


if __name__ == '__main__':
    main()