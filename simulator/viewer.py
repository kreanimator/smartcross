"""The live window: runs the simulation and shows what the cameras make of it.

Every car count on screen comes from the camera sensor. The simulation's own list of vehicles is
never read here, so the numbers can only be what vision found.
"""

import cv2
import numpy as np
from panda3d.core import Texture
from ursina import EditorCamera, Entity, Text, camera, time, window

from controller.traffic_lights import Approach
from simulator.simulation import Playback, Simulation
from vision.camera_sensor import CameraSensor

SPEED_KEYS = {"1": 1, "2": 2, "3": 4, "4": 8}
ORBIT_VIEW = {"rotation": (35, 30, 0), "zoom": -90}
TOP_DOWN_VIEW = {"rotation": (90, 0, 0), "zoom": -230}  # Shows the whole map, north up.
WIDEST_STATUS = "adaptive   0000.0 s   NS_YELLOW   cameras see 000 cars   PAUSED"  # Sizes the background box.
HELP = "space pause   r reset   1-4 speed   tab overview   right-drag orbit   scroll zoom"
FEED_SIZE = (340, 255)  # Pixels of each camera feed on screen.
FEED_HEIGHT = 0.24  # Fraction of the window's height.
GREEN, YELLOW, RED, WHITE = (80, 220, 80), (0, 210, 255), (60, 60, 230), (255, 255, 255)


def annotate(frame, tracks, roi, scale_x, scale_y):
    """A camera frame at feed size, with the watched lane and every tracked vehicle marked."""
    image = cv2.resize(frame, FEED_SIZE)
    corners = np.array([(x * scale_x, y * scale_y) for x, y in roi], np.int32)
    cv2.polylines(image, [corners], True, YELLOW, 1, cv2.LINE_AA)
    for track in tracks:
        left, top, right, bottom = track.box
        start = (int(left * scale_x), int(top * scale_y))
        cv2.rectangle(image, start, (int(right * scale_x), int(bottom * scale_y)), GREEN, 1, cv2.LINE_AA)
    return image


def caption(image, approach, seen, crossed) -> None:
    cv2.rectangle(image, (0, 0), (FEED_SIZE[0], 24), (0, 0, 0), -1)
    if seen.available:
        text = f"{approach.value}: {seen.queued} queued, {seen.approaching} moving, {crossed} crossed"
        cv2.putText(image, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(image, f"{approach.value}: CAMERA UNAVAILABLE", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)


class CameraFeeds:
    """The four roadside feeds down the right edge of the window."""

    def __init__(self):
        self.textures = {}
        self.shown_look = -1
        width = FEED_HEIGHT * FEED_SIZE[0] / FEED_SIZE[1]
        for index, approach in enumerate(Approach):
            texture = Texture(f"feed-{approach.value}")
            texture.setup_2d_texture(*FEED_SIZE, Texture.T_unsigned_byte, Texture.F_rgb)
            self.textures[approach] = texture
            feed = Entity(
                parent=camera.ui,
                model="quad",
                scale=(width, FEED_HEIGHT),
                position=(window.right.x - width / 2 - 0.005, 0.5 - (index + 0.5) * (FEED_HEIGHT + 0.005)),
            )
            feed.setTexture(texture, 1)

    def show(self, sensor: CameraSensor) -> None:
        if sensor.looks == self.shown_look:
            return
        self.shown_look = sensor.looks
        for approach, texture in self.textures.items():
            image = self._image(sensor, approach)
            texture.set_ram_image_as(image[::-1].tobytes(), "BGR")

    @staticmethod
    def _image(sensor: CameraSensor, approach: Approach) -> np.ndarray:
        frame = sensor.last_frames.get(approach)
        if frame is None:
            image = np.zeros((FEED_SIZE[1], FEED_SIZE[0], 3), np.uint8)
        else:
            scale_x, scale_y = FEED_SIZE[0] / frame.shape[1], FEED_SIZE[1] / frame.shape[0]
            analyzer = sensor.analyzers[approach]
            image = annotate(frame, sensor.last_tracks.get(approach, []), analyzer.roi, scale_x, scale_y)
        caption(image, approach, sensor.seen[approach], len(sensor.analyzers[approach].crossings))
        return image


class Viewer(Entity):
    """Ursina calls update() every frame and input() on every key press."""

    def __init__(self, app, world, detector, seed: int, strategy: str):
        super().__init__()
        self.app, self.scene_view, self.detector, self.seed, self.strategy = app, world, detector, seed, strategy
        self.simulation = self._new_simulation()
        self.playback = Playback()

        camera.clip_plane_near = 1  # A closer near plane makes the road flicker into the grass from afar.
        camera.clip_plane_far = 1000
        self.editor_camera = EditorCamera()
        self.top_down = False
        self._apply_view()

        self.feeds = CameraFeeds()
        self.status = Text(WIDEST_STATUS, parent=camera.ui, position=window.top_left + (0.02, -0.02), background=True)
        Text(HELP, parent=camera.ui, position=window.bottom_left + (0.02, 0.05), scale=0.8, background=True)

    def _new_simulation(self) -> Simulation:
        sensor = CameraSensor(self.app, self.scene_view, self.detector)
        return Simulation(self.seed, self.strategy, sensor=sensor)

    def update(self):
        for _ in range(self.playback.steps_due(time.dt)):
            self.simulation.step()
        self.scene_view.show(self.simulation)
        self.feeds.show(self.simulation.sensor)
        self._update_status()

    def input(self, key):
        if key == "space":
            self.playback.paused = not self.playback.paused
        elif key == "r":
            self.reset()
        elif key == "tab":
            self.top_down = not self.top_down
            self._apply_view()
        elif key in SPEED_KEYS:
            self.playback.speed = SPEED_KEYS[key]

    def reset(self) -> None:
        """Start the same run again, keeping the current pause and speed."""
        self.scene_view.vehicles.clear()
        self.simulation = self._new_simulation()

    def _apply_view(self) -> None:
        view = TOP_DOWN_VIEW if self.top_down else ORBIT_VIEW
        self.editor_camera.position = (0, 0, 0)
        self.editor_camera.rotation = view["rotation"]
        self.editor_camera.target_z = view["zoom"]

    def _update_status(self) -> None:
        lights = self.simulation.lights
        state = "PAUSED" if self.playback.paused else f"x{self.playback.speed}"
        seen = self.simulation.sensor.seen.values()
        cars = sum(approach.queued + approach.approaching for approach in seen)
        status = f"{self.strategy}   {self.simulation.traffic.now_s:6.1f} s   {lights.phase.value}   cameras see {cars} cars   {state}"
        if status != self.status.text:  # Rebuilding text every frame is slow.
            self.status.text = status
