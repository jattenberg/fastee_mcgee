import taichi as ti
import numpy as np
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@ti.kernel
def compute(phase: int, uv, W, H):
    for i, j in ti.ndrange(W, H):
        cen = uv[phase, i, j]
        lapl = uv[phase, i + 1, j] + uv[phase, i, j + 1] + uv[phase, i - 1, j] + uv[phase, i, j - 1] - 4.0 * cen
        du = Du * lapl[0] - cen[0] * cen[1] * cen[1] + feed * (1 - cen[0])
        dv = Dv * lapl[1] + cen[0] * cen[1] * cen[1] - (feed + kill) * cen[1]
        val = cen + 0.5 * tm.vec2(du, dv)
        uv[1 - phase, i, j] = val

    return uv

@app.get("/diffusion/{W}/{H}")
def diffusion(W: int, H, int):
    uv = ti.Vector.field(2, float, shape=(2, W, H))

    uv_grid = np.zeros((2, W, H, 2), dtype=np.float32)
    uv_grid[0, :, :, 0] = 1.0
    rand_rows = np.random.choice(range(W), 50)
    rand_cols = np.random.choice(range(H), 50)
    uv_grid[0, rand_rows, rand_cols, 1] = 1.0
    uv.from_numpy(uv_grid)
