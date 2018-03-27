__author__ = 'justinarmstrong'

import os
import pygame as pg
from pygame.surfarray import pixels3d, array3d
from . import constants as c
import platform
from . import setup
from collections import deque

p_name = platform.system()

keybinding = {
    'action': pg.K_s,
    'jump': pg.K_a,
    'left': pg.K_LEFT,
    'right': pg.K_RIGHT,
    'down': pg.K_DOWN
}


class Control(object):
    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""

    def __init__(self, caption, env):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.caption = caption
        # self.fps = 60
        self.fps = 100000
        self.show_fps = False
        self.current_time = 0.0
        self.keys = pg.key.get_pressed()
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.ml_done = False
        self.max_posision_x = 200
        self.correct_x = 80
        self.before_x = 200

    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]

    def update(self):
        self.current_time = pg.time.get_ticks()
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        self.state.update(self.screen, self.keys, self.current_time)

        # position start = 200 / end=8519
        if self.state.mario.dead:
            self.ml_done = True

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)
        self.state.previous = previous

    def get_step(self):
        if p_name == "Darwin":
            next_state = pixels3d(self.screen)
        else:
            next_state = array3d(setup.SCREEN)

        reward = 0
        score = self.state.get_score()
        position_x = self.state.last_x_position
        if position_x > self.max_posision_x:
            reward += (position_x - self.max_posision_x) * 2
            self.max_posision_x = position_x
        else:
            reward = 0

        reward = reward + score

        # time penalty
        # reward -= 0.1
        # if self.keys[275] == 1:
        #    reward += 1
        '''
        if self.keys[276] == 1:
            reward -= 1
        elif self.keys[275] == 1:
            reward += 1
        '''

        if self.keys[275] == 1:
            reward += 1
        else:
            reward -= 5

        '''
        if self.before_x < position_x:
            reward += position_x - self.before_x
        self.before_x = position_x
        '''

        # if position_x < 70 and position_x != 0:
        #    self.ml_done = True

        return (next_state, reward, self.ml_done, self.state.clear,
                self.max_posision_x, self.state.timeout, position_x)

    def event_loop(self, key):
        if key != None and self.keys != key:
            self.keys = key

        else:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.done = True
                elif event.type == pg.KEYDOWN:
                    self.keys = pg.key.get_pressed()
                    self.toggle_show_fps(event.key)
                elif event.type == pg.KEYUP:
                    self.keys = pg.key.get_pressed()
                self.state.get_event(event)

    def toggle_show_fps(self, key):
        if key == pg.K_F5:
            self.show_fps = not self.show_fps
            if not self.show_fps:
                pg.display.set_caption(self.caption)

    def main(self):
        """Main loop for entire program"""
        while not self.done:
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)
            if self.show_fps:
                fps = self.clock.get_fps()
                with_fps = "{} - {:.2f} FPS".format(self.caption, fps)
                pg.display.set_caption(with_fps)


class _State(object):
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.quit = False
        self.next = None
        self.previous = None
        self.persist = {}
        self.score = 0
        self.last_x_position = 0

    def get_score(self):
        tmp = self.score
        self.score = 0
        return tmp

    def get_event(self, event):
        pass

    def startup(self, current_time, persistant):
        self.persist = persistant
        self.start_time = current_time

    def cleanup(self):
        self.done = False
        return self.persist

    def update(self, surface, keys, current_time):
        pass


def load_all_gfx(directory, colorkey=(255, 0, 255),
                 accept=('.png', 'jpg', 'bmp')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name] = img
    return graphics


def load_all_music(directory, accept=('.wav', '.mp3', '.ogg', '.mdi')):
    songs = {}
    for song in os.listdir(directory):
        name, ext = os.path.splitext(song)
        if ext.lower() in accept:
            songs[name] = os.path.join(directory, song)
    return songs


def load_all_fonts(directory, accept=('.ttf')):
    return load_all_music(directory, accept)


def load_all_sfx(directory, accept=('.wav', '.mpe', '.ogg', '.mdi')):
    effects = {}
    for fx in os.listdir(directory):
        name, ext = os.path.splitext(fx)
        if ext.lower() in accept:
            effects[name] = pg.mixer.Sound(os.path.join(directory, fx))
    return effects
