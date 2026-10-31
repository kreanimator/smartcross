"""The interactive window: runs the simulation live and handles the keyboard."""

from ursina import EditorCamera, Entity, Text, camera, time, window

from simulator.simulation import Playback, Simulation

SPEED_KEYS = {"1": 1, "2": 2, "3": 4, "4": 8}
ORBIT_VIEW = {"rotation": (35, 30, 0), "zoom": -90}
TOP_DOWN_VIEW = {"rotation": (90, 0, 0), "zoom": -230}  # Shows the whole map, north up.
WIDEST_STATUS = "0000.0 s   NS_YELLOW   000 vehicles   PAUSED"  # Sizes the background box.
HELP = "space pause   r reset   1-4 speed   tab overview   right-drag orbit   scroll zoom"


class Viewer(Entity):
    """Ursina calls update() every frame and input() on every key press."""

    def __init__(self, world, seed: int):
        super().__init__()
        self.scene_view, self.seed = world, seed
        self.simulation = Simulation(seed)
        self.playback = Playback()

        camera.clip_plane_near = 1  # A closer near plane makes the road flicker into the grass from afar.
        camera.clip_plane_far = 1000
        self.editor_camera = EditorCamera()
        self.top_down = False
        self._apply_view()

        self.status = Text(WIDEST_STATUS, parent=camera.ui, position=window.top_left + (0.02, -0.02), background=True)
        Text(HELP, parent=camera.ui, position=window.bottom_left + (0.02, 0.05), scale=0.8, background=True)

    def update(self):
        for _ in range(self.playback.steps_due(time.dt)):
            self.simulation.step()
        self.scene_view.show(self.simulation)
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
        self.simulation = Simulation(self.seed)

    def _apply_view(self) -> None:
        view = TOP_DOWN_VIEW if self.top_down else ORBIT_VIEW
        self.editor_camera.position = (0, 0, 0)
        self.editor_camera.rotation = view["rotation"]
        self.editor_camera.target_z = view["zoom"]

    def _update_status(self) -> None:
        lights = self.simulation.lights
        state = "PAUSED" if self.playback.paused else f"x{self.playback.speed}"
        vehicles = len(self.simulation.traffic.vehicles)
        status = f"{self.simulation.traffic.now_s:6.1f} s   {lights.phase.value}   {vehicles} vehicles   {state}"
        if status != self.status.text:  # Rebuilding text every frame is slow.
            self.status.text = status
