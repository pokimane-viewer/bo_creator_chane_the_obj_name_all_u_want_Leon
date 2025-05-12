import os
import sys
import math
import json
import time
import queue
import random
import pathlib
import subprocess
import threading
import webbrowser
from typing import Any, Dict, List, Tuple, Callable
from contextlib import suppress
from dataclasses import dataclass, field as dataclass\_field
from numpy.typing import NDArray
import numpy as np
try:
import cupy as cp
except ImportError:
cp = np
def \_ensure\_submodule(name: str): return sys.modules\[name]
\_PKG\_ROOT = "pl15\_j20\_sim"
def \_identity\_eq\_hash(cls):
def \_eq(self, other): return self is other
def \_hash(self): return id(self)
cls.**eq**, cls.**hash** = \_eq, \_hash
return cls
class \_FallbackAircraft:
def **init**(self, state: Any, config: Dict\[str, Any] = None, additional\_weight: float = 1.0):
self.state = state
self.config = config or {}
self.destroyed = False
self.additional\_weight = additional\_weight
self.identity\_verified = False
def update(self, dt: float = .05):
if self.destroyed:
return
st = \_resolve\_state(self)
accel = cp.asarray(self.config.get("gravity", \[0, 0, -9.81]), dtype=cp.float32)
accel += cp.asarray(self.config.get("extra\_accel", \[0, 0, 0]), dtype=cp.float32)
st.velocity += accel \* dt
st.position += st.velocity \* dt
st.time += dt
self.state = st
class CombatResultsTable:
\_KILL\_PROB\_TABLE = {50.: .9, 100.: .7, 150.: .3, 250.: .05}
\_MAX\_ENGAGE\_RANGE = 250.
@staticmethod
def \_kill\_probability(r: float) -> float:
if r >= CombatResultsTable.\_MAX\_ENGAGE\_RANGE:
return 0.
ks = sorted(CombatResultsTable.\_KILL\_PROB\_TABLE)
for lo, hi in zip(ks\[:-1], ks\[1:]):
if lo <= r < hi:
p\_lo = CombatResultsTable.\_KILL\_PROB\_TABLE\[lo]
p\_hi = CombatResultsTable.\_KILL\_PROB\_TABLE\[hi]
α = (r - lo) / (hi - lo)
return p\_lo + α \* (p\_hi - p\_lo)
return CombatResultsTable.\_KILL\_PROB\_TABLE\[ks\[0]]
def evaluate\_engagement(self, a1, a2):
if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
return
s1 = \_resolve\_state(a1)
s2 = \_resolve\_state(a2)
d = float(cp.linalg.norm(s1.position - s2.position))
if d > self.\_MAX\_ENGAGE\_RANGE:
return
if random.random() < (pk := self.\_kill\_probability(d)):
loser = a2 if random.random() < .5 else a1
loser.destroyed = True
loser.state.velocity \*= 0
print(f"\[CRT] Engagement at {d:.1f} m – {loser.**class**.**name**} destroyed (Pk={pk:.2f})")
class TaiwanConflictCRTManager:
def **init**(self, env, aircraft, crt):
self.environment = env
self.aircraft = aircraft
self.crt = crt
def step(self, dt: float):
for ac in self.aircraft:
if getattr(ac, "destroyed", False):
continue
if hasattr(ac, "update"):
ac.update(dt)
n = len(self.aircraft)
for i in range(n):
for j in range(i + 1, n):
ai = self.aircraft\[i]
aj = self.aircraft\[j]
if getattr(ai, "destroyed", False) or getattr(aj, "destroyed", False):
continue
self.crt.evaluate\_engagement(ai, aj)
class agi:
monitor\_states = staticmethod(lambda \*a, \*\*k: None)
apply\_failsafe = staticmethod(lambda \*a, \*\*k: None)
advanced\_cooperation = staticmethod(lambda \*a, **k: None)
@dataclass
class RadarSeeker:
update\_rate: float = 100.
sensitivity: float = 1e-6
active: bool = True
def track\_target(self, tp: cp.ndarray, op: cp.ndarray) -> bool:
if not self.active:
return False
d = cp.linalg.norm(tp - op)
snr = 1 / (d**4 \* self.sensitivity + 1e-9)
return snr > 1.
@dataclass
class RadarSeekerFast(RadarSeeker):
...
@dataclass
class PL15MissileFast:
state: Any
config: Dict\[str, Any]
seeker: RadarSeekerFast = dataclass\_field(init=False)
def **post\_init**(self):
self.seeker = RadarSeekerFast(\*\*self.config.get("seeker", {}))
def update(self, dt: float = .05):
if hasattr(self.state, "velocity") and hasattr(self.state, "position"):
self.state.position += self.state.velocity \* dt
if hasattr(self.state, "time"):
self.state.time += dt
@dataclass(slots=True)
class EntityState:
position: NDArray\[np.float32] | cp.ndarray
velocity: NDArray\[np.float32] | cp.ndarray
orientation: float = 0.
time: float = 0.
acceleration: cp.ndarray = dataclass\_field(default\_factory=lambda: cp.zeros(3, dtype=cp.float32))
def as\_gpu(self):
self.position = cp.asarray(self.position() if callable(self.position) else self.position, dtype=cp.float32)
self.velocity = cp.asarray(self.velocity() if callable(self.velocity) else self.velocity, dtype=cp.float32)
if not isinstance(self.acceleration, cp.ndarray):
self.acceleration = cp.asarray(self.acceleration, dtype=cp.float32)
@property
def pos(self):
return self.position
@pos.setter
def pos(self, v):
self.position = v
@property
def vel(self):
return self.velocity
@vel.setter
def vel(self, v):
self.velocity = v
def get\_targets():
return cp.array(\[\[50., 0., 0.]], dtype=cp.float32)
class Terrain:
def height\_at(self, x, y):
return 0.
class Weather:
def **init**(self, wind=(0, 0, 0), visibility=100000.):
self.wind = cp.asarray(wind, dtype=cp.float32)
self.visibility = visibility
class RFEnvironment:
def **init**(self, t, w):
self.terrain = t
self.weather = w
class SimpleEnvironment:
def **init**(self):
self.terrain = Terrain()
self.weather = Weather()
self.rf\_env = RFEnvironment(self.terrain, self.weather)
self.get\_targets = get\_targets
def \_ensure\_array(x, shape=(3,), dtype=cp.float32):
if callable(x):
with suppress(Exception):
x = x()
if x is None:
x = cp.zeros(shape, dtype=dtype)
if not isinstance(x, cp.ndarray):
x = cp.asarray(x, dtype=dtype)
return x
def \_resolve\_state(obj):
st = getattr(obj, "state", None)
if callable(st):
with suppress(Exception):
st = st()
if st is None or not (hasattr(st, "position") and hasattr(st, "velocity")):
if hasattr(obj, "position") and hasattr(obj, "velocity"):
st = obj
else:
st = EntityState(cp.zeros(3, dtype=cp.float32), cp.zeros(3, dtype=cp.float32))
st.position = \_ensure\_array(getattr(st, "position", None))
st.velocity = \_ensure\_array(getattr(st, "velocity", None))
return st
class EngagementManager:
def **init**(self, env, ac, ms):
self.env = env
self.aircraft = ac
self.missiles = ms
def step(self, dt):
for m in self.missiles:
if hasattr(m, "update"):
m.update(dt)
class \_GraphStub:
def launch(self):
print("\[GRAPH] Graph visualisation (stub) launched.")
class capture\_graph:
def **enter**(self):
return None, \_GraphStub()
def **exit**(self, \*a):
...
def \_air\_density(alt: float) -> float:
ρ0, h = 1.225, 8500.
return ρ0 \* math.exp(-alt / h)
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class F22Aircraft(\_FallbackAircraft):
additional\_weight: float = 1.0
def update(self, dt: float = .05):
if self.destroyed:
return
lift = .05 \* self.state.velocity\[0]
drag = .01 \* (self.state.velocity\[0] \*\* 2)
thrust = cp.array(\[20., 0., 0.], dtype=cp.float32)
acc = thrust - cp.array(\[drag, 0., 9.81], dtype=cp.float32)
self.state.velocity += acc \* dt
self.state.position += self.state.velocity \* dt
self.state.time += dt
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(\_FallbackAircraft):
additional\_weight: float = 1.0
def **init**(self, st, cfg=None, additional\_weight: float = 1.0):
base = {
"mass": 25000.,
"wing\_area": 73.,
"thrust\_max": 2 \* 147000,
"Cd0": .02,
"Cd\_supersonic": .04,
"service\_ceiling": 20000.,
"radar": {"type": "KLJ-5A", "range\_fighter": 200000.},
"irst": {"range\_max": 100000.}
}
if cfg:
for k, v in cfg.items():
if isinstance(v, dict) and isinstance(base.get(k), dict):
base\[k].update(v)
else:
base\[k] = v
super().**init**(st, base, additional\_weight=additional\_weight)
def \_drag(self) -> cp.ndarray:
v = cp.linalg.norm(self.state.velocity) + 1e-6
Cd = self.config\["Cd\_supersonic"] if v / 343. > 1 else self.config\["Cd0"]
D = .5 \* \_air\_density(float(self.state.position\[2])) \* Cd \* self.config\["wing\_area"] \* v \*\* 2
return (self.state.velocity / v) \* D
def update(self, dt: float = .05):
if self.destroyed:
return
thrust = cp.array(\[self.config\["thrust\_max"], 0., 0.], dtype=cp.float32)
acc = (
thrust
\- self.\_drag()
\+ cp.array(\[0., 0., -9.81 \* self.config\["mass"]], dtype=cp.float32)
) / self.config\["mass"]
self.state.velocity += acc \* dt
self.state.position += self.state.velocity \* dt
self.state.time += dt
@dataclass(slots=True)
class AircraftState:
position: NDArray\[np.float32] | cp.ndarray
velocity: NDArray\[np.float32] | cp.ndarray
orientation: float = 0.
time: float = 0.
def as\_gpu(self):
self.position = cp.asarray(self.position, dtype=cp.float32)
self.velocity = cp.asarray(self.velocity, dtype=cp.float32)
@property
def pos(self):
return self.position
@pos.setter
def pos(self, v):
self.position = v
@property
def vel(self):
return self.velocity
@vel.setter
def vel(self, v):
self.velocity = v
def \_air\_density\_2(a: float) -> float:
ρ0, h = 1.225, 8500.
return ρ0 \* math.exp(-a / h)
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class J20Aircraft(\_FallbackAircraft):
additional\_weight: float = 1.0
def **init**(self, st, cfg=None, additional\_weight: float = 1.0):
base = {
"mass": 25000.,
"wing\_area": 73.,
"thrust\_max": 2 \* 147000,
"Cd0": .02,
"Cd\_supersonic": .04,
"service\_ceiling": 20000.,
"radar": {"type": "KLJ-5A", "range\_fighter": 200000.},
"irst": {"range\_max": 100000.}
}
if cfg:
for k, v in cfg.items():
if isinstance(v, dict) and isinstance(base.get(k), dict):
base\[k].update(v)
else:
base\[k] = v
super().**init**(st, base, additional\_weight=additional\_weight)
def \_drag(self) -> cp.ndarray:
v = cp.linalg.norm(self.state.velocity) + 1e-6
Cd = self.config\["Cd\_supersonic"] if v / 343. > 1 else self.config\["Cd0"]
D = .5 \* \_air\_density(float(self.state.position\[2])) \* Cd \* self.config\["wing\_area"] \* v \*\* 2
return (self.state.velocity / v) \* D
def update(self, dt: float = .05):
if self.destroyed:
return
thrust = cp.array(\[self.config\["thrust\_max"], 0., 0.], dtype=cp.float32)
acc = (
thrust
\- self.\_drag()
\+ cp.array(\[0., 0., -9.81 \* self.config\["mass"]], dtype=cp.float32)
) / self.config\["mass"]
self.state.velocity += acc \* dt
self.state.position += self.state.velocity \* dt
self.state.time += dt
class \_PL15DualPulse:
def **init**(self, t1=6., t2=4., F1=20000., F2=12000.):
self.t1, self.t2, self.F1, self.F2 = t1, t2, F1, F2
def **call**(self, t):
if t < self.t1:
return self.F1
elif t < self.t1 + self.t2:
return self.F2
return 0.
@dataclass
class RadarSeekerFast\_R2025(RadarSeekerFast):
def range\_max(self):
return 35000.
@dataclass
class PL15MissileFast\_R2025(PL15MissileFast):
def **post\_init**(self):
base = {
"seeker": {"update\_rate": 100., "sensitivity": 1e-6, "active": True},
"guidance": {"N": 4.},
"flight\_dynamics": {"mass": 210., "thrust\_profile": \_PL15DualPulse()},
"datalink": {"delay": .05},
"eccm": {"adaptive\_gain": 1.}
}
if self.config:
for k in base:
base\[k].update(self.config.get(k, {}))
super().**post\_init**()
self.seeker = RadarSeekerFast\_R2025(\*\*base\["seeker"])
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class F35Aircraft\_R2025(\_FallbackAircraft):
additional\_weight: float = 1.0
def **init**(self, st, cfg=None, additional\_weight: float = 1.0):
base = {
"mass": 25000.,
"wing\_area": 73.,
"thrust\_max": 2 \* 147000,
"Cd0": .02,
"Cd\_supersonic": .04,
"service\_ceiling": 20000.,
"radar": {"type": "KLJ-5A", "range\_fighter": 200000.},
"irst": {"range\_max": 100000.}
}
if cfg:
for k, v in cfg.items():
if isinstance(v, dict) and isinstance(base.get(k), dict):
base\[k].update(v)
else:
base\[k] = v
super().**init**(st, base, additional\_weight=additional\_weight)
def \_drag(self):
v = cp.linalg.norm(self.state.velocity) + 1e-6
Cd = self.config\["Cd\_supersonic"] if v / 343. > 1 else self.config\["Cd0"]
D = .5 \* \_air\_density(float(self.state.position\[2])) \* Cd \* self.config\["wing\_area"] \* v \*\* 2
return (self.state.velocity / v) \* D
def update(self, dt: float = .05):
if self.destroyed:
return
thrust\_vec = cp.array(\[self.config\["thrust\_max"], 0., 0.], dtype=cp.float32)
acc = (
thrust\_vec
\- self.\_drag()
\+ cp.array(\[0., 0., -9.81 \* self.config\["mass"]], dtype=cp.float32)
) / self.config\["mass"]
self.state.velocity += acc \* dt
self.state.position += self.state.velocity \* dt
self.state.time += dt
globals().update({
"PL15MissileFast": PL15MissileFast\_R2025,
"RadarSeekerFast": RadarSeekerFast\_R2025,
"F35Aircraft": F35Aircraft\_R2025
})
class WarGameEnvironment:
def **init**(self, t=None, w=None):
self.terrain = t or Terrain()
self.weather = w or Weather()
self.rf\_env = RFEnvironment(self.terrain, self.weather)
self.get\_targets = self.\_dummy\_targets
@staticmethod
def \_dummy\_targets():
return cp.array(\[\[100., 10., -5.]], dtype=cp.float32)
class WarGameManager:
def **init**(self, env, aircraft):
self.environment = env
self.aircrafts = aircraft
def step(self, dt):
for a in self.aircrafts:
if getattr(a, "destroyed", False):
continue
a.state.position += a.state.velocity \* dt
if hasattr(a.state, "time"):
a.state.time += dt
try:
import d3graph
except ImportError:
import types as \_t
class \_DG:
def graph(self, \*a, \*\*k):
...
def set\_config(self, \*\*k):
return self
def show(self, filepath=None):
...
d3graph = \_t.ModuleType("d3graph")
d3graph.d3graph = \_DG()
try:
import plotly.graph\_objects as go
except ImportError:
go = None
def \_capture\_frame(acs):
return \[
{
"name": ac.**class**.**name**,
"x": float((p := cp.asnumpy(getattr(ac.state, "position", cp.zeros(3))))\[0]),
"y": float(p\[1]),
"z": float(p\[2])
}
for ac in acs
]
def export\_plotly\_animation(frames, title="War Game Animation", filename="war\_game\_animation.html"):
if go is None:
print("\[PLOTLY] Not installed – skipping.")
return
import plotly.graph\_objects as \_go
first = frames\[0]
fig = \_go.Figure(
data=\[\_go.Scatter3d(
x=\[d\["x"] for d in first],
y=\[d\["y"] for d in first],
z=\[d\["z"] for d in first],
mode="markers",
marker=dict(size=4),
text=\[d\["name"] for d in first]
)],
layout=\_go.Layout(
title=title,
scene=dict(xaxis\_title="X", yaxis\_title="Y", zaxis\_title="Z"),
updatemenus=\[
dict(
type="buttons",
showactive=False,
buttons=\[
dict(
label="Play",
method="animate",
args=\[None, {
"frame": {"duration": 40, "redraw": True},
"fromcurrent": True
}]
)
]
)
]
),
frames=\[
\_go.Frame(
data=\[\_go.Scatter3d(
x=\[d\["x"] for d in fr],
y=\[d\["y"] for d in fr],
z=\[d\["z"] for d in fr],
mode="markers",
marker=dict(size=4),
text=\[d\["name"] for d in fr]
)]
)
for fr in frames\[1:]
]
)
fig.write\_html(pathlib.Path(filename).with\_suffix(".html"), auto\_play=False, include\_plotlyjs="cdn")
def simulate\_and\_capture(mgr, acs, steps, dt=1., capture\_rate=1):
fr = \[]
for s in range(steps):
mgr.step(dt)
if s % capture\_rate == 0:
fr.append(\_capture\_frame(acs))
return fr
def \_ensure\_flask(min\_ver="3.0.0"):
try:
import flask as \_fl
from packaging.version import Version as \_V
if \_V(\_fl.**version**) < \_V(min\_ver):
raise ImportError
return \_fl
except Exception:
subprocess.check\_call(\[sys.executable, "-m", "pip", "install", f"flask>={min\_ver}", "-q"])
import importlib as \_il
\_il.invalidate\_caches()
import flask as \_fl
return \_fl
flask = \_ensure\_flask()
\_LIVE\_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Live War-Game Visualisation</title>

<script src="https://d3js.org/d3.v7.min.js"></script><style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}#vis{width:100vw;height:100vh}text{fill:#fff;font-size:10px;text-anchor:middle;dominant-baseline:middle}</style></head><body><svg id="vis"></svg>

<script>const svg=d3.select("#vis"),color=d3.scaleOrdinal(d3.schemeTableau10);let scale=3;function update(f){const n=f.map(d=>({...d,x:+d.x*scale,y:+d.y*scale}));const s=svg.selectAll("g.node").data(n,d=>d.name);
const e=s.enter().append("g").attr("class","node");e.append("circle").attr("r",6).attr("fill",d=>color(d.name));e.append("text").attr("dy",-10).text(d=>d.name);
s.merge(e).attr("transform",d=>`translate(${d.x+innerWidth/2},${innerHeight/2-d.y})`);s.exit().remove();}const evt=new EventSource("/stream");evt.onmessage=e=>update(JSON.parse(e.data));</script></body></html>"""

def start\_live\_visualisation(mgr, acs, dt=1., host="127.0.0.1", port=5000):
q = queue.Queue()
def \_sim():
while True:
mgr.step(dt)
q.put(\_capture\_frame(acs))
time.sleep(dt)
threading.Thread(target=\_sim, daemon=True).start()
app = flask.Flask("live\_wargame\_vis")
@app.route("/")
def \_i():
return \_LIVE\_HTML
@app.route("/stream")
def \_s():
def \_g():
while True:
yield f"data: {json.dumps(q.get())}\n\n"
return flask.Response(\_g(), mimetype="text/event-stream")
threading.Timer(1., lambda: webbrowser.open(f"http\://{host}:{port}/", new=2)).start()
app.run(host=host, port=port, threaded=True, debug=False, use\_reloader=False)
def \_build\_react\_app(fd="frontend"):
pj = os.path.join(fd, "package.json")
if not os.path.isfile(pj):
return
try:
subprocess.check\_call(\["npm", "install"], cwd=fd)
subprocess.check\_call(\["npm", "run", "build"], cwd=fd)
except Exception as e:
pass
def serve\_react\_frontend(app, fd="frontend", rp="/react"):
from flask import send\_from\_directory
bd = os.path.join(fd, "build")
@app.route(f"{rp}/[path\:fn](path:fn)")
def \_srf(fn):
return send\_from\_directory(bd, fn)
@app.route(rp)
def \_sri():
return send\_from\_directory(bd, "index.html")
def start\_modern\_ui\_server(mgr, acs, dt=1., host="127.0.0.1", port=5000, fd="frontend"):
q = queue.Queue()
\_build\_react\_app(fd)
def \_sim():
while True:
mgr.step(dt)
q.put(\_capture\_frame(acs))
time.sleep(dt)
threading.Thread(target=\_sim, daemon=True).start()
app = flask.Flask("modern\_ui\_app")
serve\_react\_frontend(app, fd)
@app.route("/")
def \_i():
return \_LIVE\_HTML
@app.route("/stream")
def \_s():
def \_g():
while True:
yield f"data: {json.dumps(q.get())}\n\n"
return flask.Response(\_g(), mimetype="text/event-stream")
threading.Timer(1., lambda: webbrowser.open(f"http\://{host}:{port}/", new=2)).start()
app.run(host=host, port=port, threaded=True, debug=False, use\_reloader=False)
def train\_j20\_pl15(mins: int):
time.sleep(mins \* 60)
def run\_taiwan\_war\_game\_live(tmin: int = 5, dt: float = 1.):
env = WarGameEnvironment()
crt = CombatResultsTable()
j20\_s = AircraftState(
position=np.array(\[-150., 0., 10000.], dtype=np.float32),
velocity=np.array(\[3., 0., 0.], dtype=np.float32)
)
f22\_s = AircraftState(
position=np.array(\[150., 30., 11000.], dtype=np.float32),
velocity=np.array(\[-2.4, 0., 0.], dtype=np.float32)
)
j20\_s.as\_gpu()
f22\_s.as\_gpu()
mgr = TaiwanConflictCRTManager(env, \[J20Aircraft(j20\_s, {}), F22Aircraft(f22\_s, {})], crt)
start\_live\_visualisation(mgr, mgr.aircraft, dt=dt)
def run\_taiwan\_conflict\_100v100(dt=1., host="127.0.0.1", port=5000, fd="frontend"):
env = WarGameEnvironment()
crt = CombatResultsTable()
sp, gd, wx, ex, alt = 10., 10, -200., 200., 10000.
j20, f35 = \[], \[]
for i in range(100):
row = i // gd
offy = (row - gd / 2) \* sp + random.uniform(-2, 2)
offz = random.uniform(-100, 100)
sj = AircraftState(
position=np.array(\[wx, offy, alt + offz], dtype=np.float32),
velocity=np.array(\[random.uniform(1.5, 2.5), 0., 0.], dtype=np.float32)
)
sj.as\_gpu()
sf = AircraftState(
position=np.array(\[ex, offy, alt + offz], dtype=np.float32),
velocity=np.array(\[random.uniform(-2.5, -1.5), 0., 0.], dtype=np.float32)
)
sf.as\_gpu()
j20.append(J20Aircraft(sj, {}))
f35.append(F35Aircraft(sf, {}))
mgr = TaiwanConflictCRTManager(env, j20 + f35, crt)
start\_modern\_ui\_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class F15JAircraft:
state: AircraftState
config: Dict\[str, Any] = dataclass\_field(default\_factory=dict)
destroyed: bool = False
additional\_weight: float = 1.0
def **post\_init**(self):
base = {
"mass": 15000.,
"wing\_area": 56.5,
"thrust\_max": 212000,
"Cd0": 0.020,
"Cd\_supersonic": 0.040,
"service\_ceiling": 18000.,
"rcs\_m2": 4.0,
"radar": {"type": "AN/APG-63J", "range\_fighter": 150000.},
"irst": {"range\_max": 80000.}
}
for k, v in self.config.items():
if isinstance(v, dict) and isinstance(base.get(k), dict):
base\[k].update(v)
else:
base\[k] = v
self.config = base
def \_drag(self) -> cp.ndarray:
v = cp.linalg.norm(self.state.velocity) + 1e-6
mach = v / 343.
Cd = self.config\["Cd\_supersonic"] if mach > 1 else self.config\["Cd0"]
D = .5 \* \_air\_density(float(self.state.position\[2])) \* Cd \* self.config\["wing\_area"] \* (v**2)
return (self.state.velocity / v) \* D
def update(self, dt: float = .05):
if self.destroyed:
return
vmag = cp.linalg.norm(self.state.velocity)
if vmag < 1e-3:
self.state.velocity += cp.array(\[1., 0., 0.], dtype=cp.float32) \* dt
thrust\_vec = cp.array(\[self.config\["thrust\_max"], 0., 0.], dtype=cp.float32)
acc = (
thrust\_vec
\- self.\_drag()
\+ cp.array(\[0., 0., -9.81 \* self.config\["mass"]], dtype=cp.float32)
) / self.config\["mass"]
self.state.velocity += acc \* dt
self.state.position += self.state.velocity \* dt
if hasattr(self.state, "time"):
self.state.time += dt
def run\_taiwan\_conflict\_jpn\_usa\_v\_china(dt=1., host="127.0.0.1", port=5001, fd="frontend"):
env = WarGameEnvironment()
crt = CombatResultsTable()
j20\_list, f35\_list, f15j\_list = \[], \[], \[]
alt = 10000.
for i in range(20):
st\_j = AircraftState(
position=np.array(\[-300. + i*5, random.uniform(-30, 30), alt], dtype=np.float32),
velocity=np.array(\[2., 0., 0.], dtype=cp.float32)
)
st\_j.as\_gpu()
j20\_list.append(J20Aircraft(st\_j, {}))
for i in range(15):
st\_f = AircraftState(
position=np.array(\[300. - i*5, random.uniform(-30, 30), alt], dtype=np.float32),
velocity=np.array(\[-2., 0., 0.], dtype=cp.float32)
)
st\_f.as\_gpu()
f35\_list.append(F35Aircraft(st\_f, {}))
for i in range(15):
st\_jp = AircraftState(
position=np.array(\[250. - i*5, random.uniform(-50, 50), alt+500.], dtype=np.float32),
velocity=np.array(\[-2., 0., 0.], dtype=cp.float32)
)
st\_jp.as\_gpu()
f15j\_list.append(F15JAircraft(st\_jp, {}))
all\_aircraft = j20\_list + f35\_list + f15j\_list
mgr = TaiwanConflictCRTManager(env, all\_aircraft, crt)
start\_modern\_ui\_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)
def demo\_gpu\_async():
env = WarGameEnvironment()
j20s = \[
J20Aircraft(
EntityState(
cp.array(\[-200. + i*2, 0., 10000.], dtype=cp.float32),
cp.array(\[3., 0., 0.], dtype=cp.float32)
),
{}
)
for i in range(500)
]
f35s = \[
F35Aircraft(
EntityState(
cp.array(\[200. - i\*2, 50., 10500.], dtype=cp.float32),
cp.array(\[-3., 0., 0.], dtype=cp.float32)
),
{}
)
for i in range(500)
]
from functools import cached\_property
import asyncio
try:
from numba import cuda
except ImportError:
cuda = None
class GPUBatchManager:
def **init**(self, entities: List\[\_FallbackAircraft]):
self.entities = entities
self.n = len(entities)
self.\_arr\_pos = cp.zeros((self.n, 3), dtype=cp.float32)
self.\_arr\_vel = cp.zeros((self.n, 3), dtype=cp.float32)
self.*arr\_alive = cp.ones(self.n, dtype=cp.bool*)
self.\_refresh\_arrays()
def \_refresh\_arrays(self):
for i, e in enumerate(self.entities):
st = \_resolve\_state(e)
self.\_arr\_pos\[i] = st.position
self.\_arr\_vel\[i] = st.velocity
self.\_arr\_alive\[i] = not getattr(e, "destroyed", False)
@cached\_property
def \_gpu\_update\_kernel(self):
if cuda is None:
return None
@cuda.jit
def \_k(pos, vel, alive, dt):
i = cuda.grid(1)
if i >= pos.shape\[0] or not alive\[i]:
return
vel\[i, 2] -= 9.81 \* dt
pos\[i, 0] += vel\[i, 0] \* dt
pos\[i, 1] += vel\[i, 1] \* dt
pos\[i, 2] += vel\[i, 2] \* dt
return \_k
def step(self, dt: float):
if cuda and self.\_gpu\_update\_kernel:
threads = 128
blocks = (self.n + threads - 1) // threads
self.\_gpu\_update\_kernel\[blocks, threads]\(self.\_arr\_pos, self.\_arr\_vel, self.\_arr\_alive, dt)
else:
self.\_arr\_vel\[:, 2] -= 9.81 \* dt
self.\_arr\_pos += self.\_arr\_vel \* dt
for i, e in enumerate(self.entities):
if not self.\_arr\_alive\[i]:
continue
e.state.position = self.\_arr\_pos\[i]
e.state.velocity = self.\_arr\_vel\[i]
class AsyncSimulationServer:
def **init**(self, mgr: GPUBatchManager, host="127.0.0.1", port=8765, dt=0.05):
try:
import websockets
except ImportError:
subprocess.check\_call(\[sys.executable, "-m", "pip", "install", "websockets>=12", "-q"])
import importlib
importlib.invalidate\_caches()
import websockets
self.websockets = sys.modules\["websockets"]
self.mgr = mgr
self.host = host
self.port = port
self.dt = dt
async def \_producer(self, websocket):
while True:
self.mgr.step(self.dt)
data = \[
{"x": float(p\[0]), "y": float(p\[1]), "z": float(p\[2])}
for p in cp.asnumpy(self.mgr.\_arr\_pos)
]
await websocket.send(json.dumps(data))
await asyncio.sleep(self.dt)
async def \_handler(self, websocket, \_path):
await self.\_producer(websocket)
def run(self):
srv = self.websockets.serve(self.\_handler, self.host, self.port)
print(f"\[ASYNC-SIM] Serving on ws\://{self.host}:{self.port}")
asyncio.get\_event\_loop().run\_until\_complete(srv)
asyncio.get\_event\_loop().run\_forever()
gpu\_mgr = GPUBatchManager(j20s + f35s)
AsyncSimulationServer(gpu\_mgr, port=8888, dt=0.02).run()
@dataclass(slots=True)
class CarrierState:
position: NDArray\[np.float32] | cp.ndarray
velocity: NDArray\[np.float32] | cp.ndarray
orientation: float = 0.
time: float = 0.
def as\_gpu(self):
self.position = cp.asarray(self.position, dtype=cp.float32)
self.velocity = cp.asarray(self.velocity, dtype=cp.float32)
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class USCarrierGroup(\_FallbackAircraft):
additional\_weight: float = 1.0
def **init**(self, st, cfg=None, additional\_weight: float = 1.0):
base = {
"mass": 100000.,
"max\_speed": 15.,
"rcs\_m2": 10000.,
"defenses": {"cwisp": True},
}
if cfg:
for k, v in cfg.items():
base\[k] = v
super().**init**(st, base, additional\_weight=additional\_weight)
def update(self, dt: float = 1.):
if self.destroyed:
return
spd = cp.linalg.norm(self.state.velocity)
if spd < self.config\["max\_speed"]:
self.state.velocity\[0] += 0.02 \* dt
self.state.position += self.state.velocity \* dt
if hasattr(self.state, "time"):
self.state.time += dt
@dataclass
class BallisticMissileState:
position: NDArray\[np.float32] | cp.ndarray
velocity: NDArray\[np.float32] | cp.ndarray
time: float = 0.
def as\_gpu(self):
self.position = cp.asarray(self.position, dtype=cp.float32)
self.velocity = cp.asarray(self.velocity, dtype=cp.float32)
@\_identity\_eq\_hash
@dataclass(slots=True, eq=False)
class DF21D\_ANTISHIP:
state: BallisticMissileState
config: Dict\[str, Any]
destroyed: bool = False
additional\_weight: float = 1.0
def **init**(self, st: BallisticMissileState, cfg=None, additional\_weight: float = 1.0):
self.config = cfg or {
"mass": 14000.,
"thrust": 0.,
"rcs\_m2": 2.,
"drag\_coeff": 0.3
}
self.state = st
self.destroyed = False
self.additional\_weight = additional\_weight
def update(self, dt: float = 1.):
if self.destroyed:
return
pos = self.state.position
vel = self.state.velocity
alt = float(pos\[2])
if alt <= 0:
self.destroyed = True
print(f"\[DF21D] Impact at {pos\[0]:.1f}, {pos\[1]:.1f}")
return
vmag = cp.linalg.norm(vel) + 1e-6
ρ = \_air\_density(alt)
D = 0.5 \* ρ \* self.config\["drag\_coeff"] \* (vmag**2) \* 0.5
drag\_vec = (vel / vmag) \* D
accel = cp.array(\[0., 0., -9.81], dtype=cp.float32) - (drag\_vec / self.config\["mass"])
self.state.velocity += accel \* dt
self.state.position += self.state.velocity \* dt
self.state.time += dt
class PLARocketForces:
def **init**(self, environment):
self.environment = environment
self.launched\_missiles = \[]
def launch\_df21d(self, launch\_pos=(0., 0., 300000.), target=None):
initial\_vel = cp.array(\[0., 0., -1000.], dtype=cp.float32)
st = BallisticMissileState(cp.array(launch\_pos, dtype=cp.float32), initial\_vel)
df21d = DF21D\_ANTISHIP(st)
self.launched\_missiles.append(df21d)
def step(self, dt=1.):
for m in self.launched\_missiles:
m.update(dt)
class CombatResultsTableAdv(CombatResultsTable):
\_MAX\_ENGAGE\_RANGE = 300.
\_BASE\_PK = {50.: .92, 100.: .77, 150.: .45, 250.: .18, 300.: .05}
def \_kill\_probability(self, r: float, rcs1: float|None, rcs2: float|None) -> float:
base\_pk = super().\_kill\_probability(r) if r <= 250. else self.\_interp\_pk(r)
if rcs1 and rcs2:
rcs\_mean = math.sqrt(rcs1 \* rcs2)
stealth\_factor = min(1., max(.4, (rcs\_mean / .1) \*\* .2))
base\_pk \*= stealth\_factor
return base\_pk
def \_interp\_pk(self, r: float) -> float:
ks = sorted(self.\_BASE\_PK)
for lo, hi in zip(ks\[:-1], ks\[1:]):
if lo <= r < hi:
p\_lo, p\_hi = self.\_BASE\_PK\[lo], self.\_BASE\_PK\[hi]
α = (r - lo) / (hi - lo)
return p\_lo + α \* (p\_hi - p\_lo)
return self.\_BASE\_PK\[ks\[-1]]
def evaluate\_engagement(self, a1, a2):
if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
return
s1 = \_resolve\_state(a1)
s2 = \_resolve\_state(a2)
d = float(cp.linalg.norm(s1.position - s2.position))
if d > self.\_MAX\_ENGAGE\_RANGE:
return
rcs1 = getattr(a1, "config", {}).get("rcs\_m2", None)
rcs2 = getattr(a2, "config", {}).get("rcs\_m2", None)
if random.random() < (pk := self.\_kill\_probability(d, rcs1, rcs2)):
loser = a2 if random.random() < .5 else a1
loser.destroyed = True
loser.state.velocity \*= 0
print(f"\[CRT-ADV] {loser.**class**.**name**} destroyed at {d:.1f} m (Pk={pk:.2f})")
def run\_pla\_rocket\_forces\_vs\_carrier(dt=1., sim\_time=300):
env = WarGameEnvironment()
crt = CombatResultsTableAdv()
carrier\_st = CarrierState(
position=np.array(\[-500., 0., 0.], dtype=np.float32),
velocity=np.array(\[5., 0., 0.], dtype=np.float32)
)
carrier\_st.as\_gpu()
carrier = USCarrierGroup(carrier\_st, {})
pla\_forces = PLARocketForces(env)
pla\_forces.launch\_df21d(launch\_pos=(0., 0., 300000.))
manager = TaiwanConflictCRTManager(env, \[carrier], crt)
ballistic\_list = pla\_forces.launched\_missiles
frames = \[]
for t in range(sim\_time):
manager.step(dt)
pla\_forces.step(dt)
for missile in ballistic\_list:
manager.crt.evaluate\_engagement(missile, carrier)
frames.append(\_capture\_frame(\[carrier] + ballistic\_list))
if carrier.destroyed:
break
export\_plotly\_animation(frames, title="PLA Rockets vs US Carrier", filename="pla\_vs\_carrier.html")
def main\_extended():
run\_pla\_rocket\_forces\_vs\_carrier(dt=1., sim\_time=180)
if **name** == "**main**" and os.getenv("RUN\_PLA\_VS\_CARRIER", "0") == "1":
main\_extended()
SOUND\_ANNOUNCEMENT\_HTML = """<!DOCTYPE html>

<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Wargames Commencement - Interactive Sound</title>
  <style>
    body {
      margin: 0; padding: 0;
      background: #222; color: #f2f2f2;
      font-family: sans-serif; display: flex;
      flex-direction: column; justify-content: center;
      align-items: center; height: 100vh;
    }
    button {
      font-size: 1.2rem; padding: 10px 20px;
      cursor: pointer; border: none; border-radius: 5px;
      background: #444; color: #fff;
    }
    button:hover {
      background: #666;
    }
    audio {
      margin-top: 20px;
    }
    .announcement {
      max-width: 600px; text-align: center; margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="announcement">
    <h2>Pete Hegseth vs General Zhang Youxia</h2>
    <p>Announcing the commencement of wargames...</p>
    <p>US Defense Secretary and Fox News host Pete Hegseth gave away the position of his command 3 hours before the jets took off to drop a live JDAM on target.</p>
  </div>
  <button onclick="playSound()">Play Announcement</button>
  <audio id="audioEl" controls style="display:none;">
    <source src="/announcement.ogg" type="audio/ogg">
    <source src="/announcement.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <script>
    function playSound(){
      const audio = document.getElementById('audioEl');
      audio.play();
    }
  </script>
</body>
</html>
"""
def intensethinking_crt_thought():
    pass
import io
def start_sound_announcement_server(host="127.0.0.1", port=5050):
    app = flask.Flask("sound_announcement")
    @app.route("/")
    def index():
        return SOUND_ANNOUNCEMENT_HTML
    @app.route("/announcement.ogg")
    def serve_ogg():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/ogg",
            as_attachment=False,
            download_name="announcement.ogg"
        )
    @app.route("/announcement.mp3")
    def serve_mp3():
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="announcement.mp3"
        )
    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
def main_sound_announcement():
    intensethinking_crt_thought()
    start_sound_announcement_server()
if __name__ == "__main__" and os.getenv("RUN_SOUND_ANNOUNCEMENT", "0") == "1":
    main_sound_announcement()
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class POTUS:
    name: str = "GenericPOTUS-2025"
    forced_joint_venture: bool = True
    time_in_office: float = 0.0
    additional_weight: float = 1.2
    destroyed: bool = False
    def step_day(self, dt_days: float = 1.0):
        self.time_in_office += dt_days
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class VolodymyrZelensky:
    name: str = "Volodymyr Zelensky"
    unconditional_ceasefire_plea: bool = False
    destroyed: bool = False
    additional_weight: float = 0.8
    def request_talks(self):
        if not self.unconditional_ceasefire_plea:
            self.unconditional_ceasefire_plea = True
            print(f"[ZELENSKY] {self.name} is now requesting unconditional ceasefire talks with Putin.")
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class PeteHegseth:
    date: str = "2/28/25"
    statement: str = "Dragged Zelensky into Oval Office"
    destroyed: bool = False
    additional_weight: float = 1.0
    def speak(self):
        print(f"[PETE HEGSETH] On {self.date}, statement: '{self.statement}'")
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoShangCreator:
    name: str = "Bo Shang"
    def remind_history(self):
        print(f"[BO SHANG] {self.name} reminds that Ukraine & Russia historically fought together in WWII at Stalingrad.")
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MarkJosephCarneyCreator:
    name: str = "Mark Joseph Carney"
    role: str = "Prime Minister of Canada"
    def remind_history(self):
        print(f"[MARK CARNEY] {self.name}, {self.role}, replaced Justin Trudeau on March 4 2025. Canada denies violating NAFTA while stating Donald Trump, in his second term, violated it to launch a trade war.")
def run_indefinite_ukraine_peace_simulation():
    potus = POTUS()
    zelensky = VolodymyrZelensky()
    hegseth = PeteHegseth()
    bo_shang = BoShangCreator()
    while True:
        time.sleep(1.0)
        potus.step_day(dt_days=1.0)
        if not zelensky.unconditional_ceasefire_plea:
            zelensky.request_talks()
        hegseth.speak()
        bo_shang.remind_history()
class WeightedCRT(CombatResultsTable):
    def _kill_probability(self, r: float, w1: float, w2: float) -> float:
        base_pk = super()._kill_probability(r)
        return base_pk * math.sqrt(w1 * w2)
    def evaluate_engagement(self, a1, a2):
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        w1 = getattr(a1, "additional_weight", 1.0)
        w2 = getattr(a2, "additional_weight", 1.0)
        if random.random() < (pk := self._kill_probability(d, w1, w2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[W-CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (WeightedPk={pk:.2f})")
def demo_weighted_crt_scenario():
    env = SimpleEnvironment()
    weighted_crt = WeightedCRT()
    potus = POTUS(additional_weight=1.5)
    zelensky = VolodymyrZelensky(additional_weight=0.5)
    potus.state = EntityState(cp.array([0., 0., 0.], dtype=cp.float32),
                              cp.array([0., 0., 0.], dtype=cp.float32))
    zelensky.state = EntityState(cp.array([10., 0., 0.], dtype=cp.float32),
                                 cp.array([0., 0., 0.], dtype=cp.float32))
    manager = TaiwanConflictCRTManager(env, [potus, zelensky], weighted_crt)
    manager.step(dt=0.01)
    print(f"[DEMO] POTUS destroyed? {potus.destroyed}, Zelensky destroyed? {zelensky.destroyed}")
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoeingF47(_FallbackAircraft):
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 26000.,
            "wing_area": 70.,
            "thrust_max": 2 * 150000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 22000.,
            "radar": {"type": "Boeing-AdvRadar", "range_fighter": 210000.},
            "irst": {"range_max": 120000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)
    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D
    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28A_UAV(_FallbackAircraft):
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 8000.,
            "wing_area": 35.,
            "thrust_max": 50000,
            "Cd0": .025,
            "Cd_supersonic": .05,
            "service_ceiling": 16000.,
            "radar": {"type": "Boeing-UAVRadarA", "range_fighter": 150000.},
            "irst": {"range_max": 50000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)
    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D
    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28B_UAV(_FallbackAircraft):
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 8500.,
            "wing_area": 36.,
            "thrust_max": 52000,
            "Cd0": .024,
            "Cd_supersonic": .045,
            "service_ceiling": 16500.,
            "radar": {"type": "GA-UAVRadarB", "range_fighter": 155000.},
            "irst": {"range_max": 55000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)
    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D
    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28C_UAV(_FallbackAircraft):
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        base = {
            "mass": 9000.,
            "wing_area": 37.,
            "thrust_max": 54000,
            "Cd0": .023,
            "Cd_supersonic": .045,
            "service_ceiling": 17000.,
            "radar": {"type": "Anduril-UAVRadarC", "range_fighter": 160000.},
            "irst": {"range_max": 60000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)
    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D
    def update(self, dt: float = .05):
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class SamanthaBriascoStewart:
    name: str = "Samantha Briasco-Stewart"
    email: str = "erosolar@alum.mit.edu"
    destroyed: bool = False
    additional_weight: float = 1.0
    def speak(self, message: str="Hello, Bo."):
        print(f"[SamanthaBriascoStewart] {message}")
    def update_weight(self, new_weight: float):
        self.additional_weight = new_weight
@_identity_eq_hash
@dataclass(slots=True, eq=False)
class john_doe_non_creator:
    pass
Done.
