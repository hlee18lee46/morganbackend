import sys
import math
from PyQt6.QtWidgets import QOpenGLWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class AnimatedBackground(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set OpenGL format for better rendering
        fmt = QSurfaceFormat()
        fmt.setSamples(4)  # Enable antialiasing
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        
        # Animation parameters
        self.angle = 0.0
        self.particles = []
        self.num_particles = 50
        self.particle_speed = 0.5
        
        # Initialize particles
        for _ in range(self.num_particles):
            self.particles.append({
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10),
                'z': np.random.uniform(-20, 0),
                'size': np.random.uniform(0.1, 0.3),
                'speed': np.random.uniform(0.2, 0.8),
                'color': QColor(
                    np.random.randint(100, 200),  # Soft blue range
                    np.random.randint(180, 255),
                    np.random.randint(200, 255),
                    150  # Semi-transparent
                )
            })
        
        # Start animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 FPS
        
        # Enable transparency
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
    def update_animation(self):
        self.angle += 0.5
        
        # Update particle positions
        for particle in self.particles:
            particle['z'] += particle['speed'] * self.particle_speed
            if particle['z'] > 5:  # Reset particle when it gets too close
                particle['z'] = -20
                particle['x'] = np.random.uniform(-10, 10)
                particle['y'] = np.random.uniform(-10, 10)
        
        self.update()
    
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Transparent background
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Set camera position
        glTranslatef(0, 0, -30)
        glRotatef(self.angle * 0.1, 0, 1, 0)
        
        # Draw particles
        for particle in self.particles:
            glPushMatrix()
            glTranslatef(particle['x'], particle['y'], particle['z'])
            
            # Apply pulsing effect
            pulse = 0.7 + 0.3 * math.sin(self.angle * 0.05)
            size = particle['size'] * pulse
            
            # Set particle color
            glColor4f(
                particle['color'].redF(),
                particle['color'].greenF(),
                particle['color'].blueF(),
                particle['color'].alphaF() * (1 - particle['z']/5)  # Fade out as particles get closer
            )
            
            # Draw particle as a smooth sphere
            quad = gluNewQuadric()
            gluSphere(quad, size, 16, 16)
            gluDeleteQuadric(quad)
            
            glPopMatrix()
        
        # Draw connecting lines between nearby particles
        glBegin(GL_LINES)
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                dist = math.sqrt(
                    (p1['x'] - p2['x'])**2 +
                    (p1['y'] - p2['y'])**2 +
                    (p1['z'] - p2['z'])**2
                )
                if dist < 5:  # Only connect nearby particles
                    alpha = (1 - dist/5) * 0.3  # Fade out with distance
                    glColor4f(0.5, 0.8, 1.0, alpha)
                    glVertex3f(p1['x'], p1['y'], p1['z'])
                    glVertex3f(p2['x'], p2['y'], p2['z'])
        glEnd() 