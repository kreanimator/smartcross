"""The traffic signal heads shown in the scene. They only display what they are given."""

from ursina import Entity, color
from ursina.shaders import unlit_shader

from controller.traffic_lights import Approach, Lamp
from simulator.roads import OUTWARD, ROAD_WIDTH, STOP_LINE, heading_degrees

POLE_HEIGHT = 6
LIT = {Lamp.RED: color.rgb32(255, 40, 30), Lamp.YELLOW: color.rgb32(255, 200, 0), Lamp.GREEN: color.rgb32(40, 255, 80)}
OFF = color.rgb32(35, 35, 35)


class SignalHead:
    """A pole on the driver's right at the stop line, with one bulb per lamp colour."""

    def __init__(self, approach: Approach):
        out_x, out_z = OUTWARD[approach]
        right_x, right_z = -out_z, out_x
        side, along = ROAD_WIDTH / 2 + 1, STOP_LINE
        root = Entity(
            position=(out_x * along + right_x * side, 0, out_z * along + right_z * side),
            rotation_y=heading_degrees(approach),
        )
        Entity(parent=root, model="cube", y=POLE_HEIGHT / 2, scale=(0.2, POLE_HEIGHT, 0.2), color=color.dark_gray)
        Entity(parent=root, model="cube", y=POLE_HEIGHT + 0.4, scale=(0.7, 2, 0.4), color=color.black)

        self.bulbs = {}
        for lamp, height in ((Lamp.RED, 1.0), (Lamp.YELLOW, 0.4), (Lamp.GREEN, -0.2)):
            # Unlit, so a bulb glows the same in sun and shadow. It shows through both faces.
            self.bulbs[lamp] = Entity(
                parent=root, model="sphere", y=POLE_HEIGHT + height, scale=0.55, color=OFF, shader=unlit_shader
            )

    def show(self, lamp: Lamp) -> None:
        for bulb_lamp, bulb in self.bulbs.items():
            bulb.color = LIT[bulb_lamp] if bulb_lamp is lamp else OFF
