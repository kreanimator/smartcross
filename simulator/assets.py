"""Loads Kenney models (CC0) from the zip archives in assets/models."""

from pathlib import Path
from zipfile import ZipFile

from panda3d.core import Filename
from ursina import Entity

MODELS = Path(__file__).parent.parent / "assets" / "models"
ARCHIVES = {"cars": "kenney_car-kit.zip", "city": "kenney_city-kit-industrial_2.0.zip"}
GLB_FOLDER = "Models/GLB format/"


def extract_models() -> None:
    """Unpack the GLB models once, into a folder git ignores."""
    for kit, archive in ARCHIVES.items():
        target = MODELS / "extracted" / kit
        if target.exists():
            continue
        with ZipFile(MODELS / archive) as zip_file:
            for name in zip_file.namelist():
                if name.startswith(GLB_FOLDER) and not name.endswith("/"):
                    path = target / name.removeprefix(GLB_FOLDER)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(zip_file.read(name))


def model_entity(kit: str, name: str, **kwargs) -> Entity:
    path = MODELS / "extracted" / kit / f"{name}.glb"
    model = loader.load_model(Filename.from_os_specific(str(path)))  # noqa: F821 (Panda3D builtin)
    return Entity(model=model, **kwargs)
