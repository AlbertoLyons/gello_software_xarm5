# Importación de liberías necesarias
import pygame
"""
Script para manejar la interfaz de teclado para iniciar, continuar o detener la grabación de datos.
Se utiliza la librería pygame para detectar las teclas presionadas.
"""
# Definición de colores para la interfaz visual
NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
# Definición de teclas para iniciar, continuar y detener la grabación
KEY_START = pygame.K_s
KEY_CONTINUE = pygame.K_c
KEY_QUIT_RECORDING = pygame.K_q

"""
Clase para manejar la interfaz de teclado para iniciar, continuar o detener la grabación de datos.
- La pantalla cambia de color según el estado: gris para normal, verde para iniciar, rojo para detener.
"""
class KBReset:
    # Inicialización de la clase, configuración de la pantalla y estado inicial
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)
        self._saved = False
    # Función para actualizar el estado de la grabación según las teclas presionadas
    def update(self) -> str:
        pressed_last = self._get_pressed()
        if KEY_QUIT_RECORDING in pressed_last:
            self._set_color(RED)
            self._saved = False
            return "normal"

        if self._saved:
            return "save"

        if KEY_START in pressed_last:
            self._set_color(GREEN)
            self._saved = True
            return "start"

        self._set_color(NORMAL)
        return "normal"
    # Función para detectar las teclas presionadas utilizando pygame
    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed
    # Función para cambiar el color de la pantalla según el estado de la grabación
    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()

# Función principal para ejecutar la interfaz de teclado y mostrar el estado de la grabación
def main():
    kb = KBReset()
    while True:
        state = kb.update()
        if state == "start":
            print("start")


if __name__ == "__main__":
    main()
