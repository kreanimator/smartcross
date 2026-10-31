"""Sunlight with shadows. Each surface is its own colour times the light reaching it."""

from panda3d.core import AmbientLight, DirectionalLight, Shader
from ursina import scene

SHADOW_MAP_SIZE = 4096
SHADOW_AREA = 240  # Metres of ground covered by the shadow map.
SHADOW_BIAS = 0.0005  # Stops surfaces from shadowing themselves.

# Both shader stages must declare this uniform identically.
LIGHT_SOURCE = """uniform struct {
    vec4 position;
    vec3 color;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];"""

VERTEX_SHADER = f"""#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
{LIGHT_SOURCE}
in vec4 vertex;
in vec3 normal;
in vec2 p3d_MultiTexCoord0;
out vec2 texcoord;
out vec3 view_normal;
out vec4 shadow_coord;

void main() {{
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    texcoord = p3d_MultiTexCoord0;
    view_normal = p3d_NormalMatrix * normal;
    shadow_coord = p3d_LightSource[0].shadowViewMatrix * p3d_ModelViewMatrix * vertex;
}}
"""

FRAGMENT_SHADER = f"""#version 150
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform struct {{ vec4 ambient; }} p3d_LightModel;
{LIGHT_SOURCE}
in vec2 texcoord;
in vec3 view_normal;
in vec4 shadow_coord;
out vec4 fragment_color;

void main() {{
    vec3 to_sun = normalize(p3d_LightSource[0].position.xyz);
    float facing_sun = max(dot(normalize(view_normal), to_sun), 0.0);
    vec4 biased_coord = shadow_coord;
    biased_coord.z -= {SHADOW_BIAS} * biased_coord.w;
    float in_sun = textureProj(p3d_LightSource[0].shadowMap, biased_coord);
    vec3 light = p3d_LightModel.ambient.rgb + p3d_LightSource[0].color * facing_sun * in_sun;
    vec4 surface = texture(p3d_Texture0, texcoord) * p3d_ColorScale;
    fragment_color = vec4(surface.rgb * light, surface.a);
}}
"""


def add_sunlight() -> None:
    ambient = AmbientLight("ambient")
    ambient.set_color((0.5, 0.5, 0.55, 1))
    scene.set_light(scene.attach_new_node(ambient))

    sun = DirectionalLight("sun")
    sun.set_color((0.7, 0.68, 0.62, 1))
    sun.set_shadow_caster(True, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
    sun.get_lens().set_film_size(SHADOW_AREA, SHADOW_AREA)
    sun.get_lens().set_near_far(10, 400)
    sun_node = scene.attach_new_node(sun)
    sun_node.set_pos(-80, 150, -60)
    sun_node.look_at(0, 0, 0)
    scene.set_light(sun_node)

    scene.set_shader(Shader.make(Shader.SL_GLSL, VERTEX_SHADER, FRAGMENT_SHADER))
