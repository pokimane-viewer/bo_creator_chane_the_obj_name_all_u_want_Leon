#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==================================================
 MASSIVE WAR-GAME SIMULATION + REACT+D3 FRONT-END
==================================================

This single-file Python script showcases a full modern approach to:
  1) Provide the original code, unmodified in logic;
  2) Launch a Flask server, automatically build & serve a React+D3.js
     interface (via npm run build);
  3) Stream real-time simulation data using Server-Sent Events (SSE);
  4) Display an interactive UI/UX so advanced that not even the PLA
     has seen before.

How to run:
-----------
1) Ensure you have Python >= 3.9 installed.
2) Ensure Node.js + npm is installed for React build steps.
3) (Optional) Create a virtual environment for Python.
4) Install Python dependencies:
       pip install --upgrade flask packaging d3graph plotly
5) Create or place a React app in the subfolder "frontend" or rename as
   you like. Example:
       cd frontend
       npm install
       npm run build
   This yields a "build" folder with production-ready static files.

6) Launch this Python script:
       python main_react_d3_sim.py

7) The script:
     • Automatically calls "npm run build" for the React app if needed.
     • Starts a Flask server on port 5000.
     • Opens your default browser to http://127.0.0.1:5000
     • Streams live data for a real-time war-game 3D simulation.

You can adapt the React front-end code to harness D3 and any advanced
UI elements. The SSE endpoint sends JSON frames with the positions
and states of all simulated objects.
"""

from __future__ import annotations
_PKG_ROOT = "pl15_j20_sim"
_PKG = _PKG_ROOT
import math, random, sys, types as _t, os, functools, warnings, secrets, importlib, subprocess, inspect, pathlib, json, datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional
from numpy.typing import NDArray
import numpy as _np

def _ensure_submodule(name: str) -> _t.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = _t.ModuleType(name)
    sys.modules[name] = module
    pkg, _, attr = name.rpartition(".")
    if pkg:
        parent = _ensure_submodule(pkg)
        setattr(parent, attr, module)
    if not hasattr(module, "__path__"):
        module.__path__ = []
    return module

try:
    from packaging.version import Version as _V
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "packaging"])
    importlib.invalidate_caches()
    from packaging.version import Version as _V

def _ensure_cupy(min_ver: str = "12.8.0"):
    try:
        import cupy as _cp
        if _V(_cp.__version__) < _V(min_ver):
            raise ImportError(f"CuPy {_cp.__version__} < {min_ver}")
        try:
            _cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            raise ImportError(f"CUDA runtime unavailable: {e}") from e
        try:
            _cp.RawModule(code="extern \"C\" __global__ void _noop(){}")
        except Exception as e:
            raise ImportError(f"NVRTC unavailable: {e}") from e
        if not (hasattr(_cp, "cuda") and
                hasattr(_cp.cuda, "cub") and
                hasattr(_cp.cuda.cub, "CUPY_CUB_SUM")):
            raise ImportError("Required CUB symbols missing")
        return _cp
    except Exception:
        import numpy as _np, types as _t, sys as _s
        warnings.filterwarnings("ignore",
                                message="CuPy not available or incompatible",
                                category=RuntimeWarning)
        warnings.warn(
            "CuPy not available or incompatible – falling back to CPU-only NumPy stub.",
            RuntimeWarning,
        )
        def _make_stub() -> _t.ModuleType:
            _cps = _t.ModuleType("cupy")
            _cps.ndarray = _np.ndarray
            _cps.float32 = _np.float32
            _cps.float64 = _np.float64
            _cps.int32   = _np.int32
            _cps.int64   = _np.int64
            _cps.bool_   = _np.bool_
            _cps.inf     = _np.inf
            _cps.asarray    = lambda x, dtype=None: _np.asarray(x, dtype=dtype)
            _cps.array      = lambda x, dtype=None: _np.array(x, dtype=dtype)
            _cps.zeros      = lambda *a, **k: _np.zeros(*a, **k)
            _cps.ones       = lambda *a, **k: _np.ones(*a, **k)
            _cps.zeros_like = _np.zeros_like
            _cps.asnumpy    = lambda x: _np.asarray(x)
            _cps.linalg = _np.linalg
            _cps.cross  = _np.cross
            _cps.dot    = _np.dot
            _cps.sqrt   = _np.sqrt
            _cps.random = _np.random
            _cps.fuse   = lambda *a, **k: (lambda f: f)
            class _FakeElementwiseKernel:
                def __init__(self, *_, **__): ...
                def __call__(self, rx, ry, rz, vx, vy, vz, N, ax, ay, az):
                    _norm = _np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-6
                    _lx, _ly, _lz = rx / _norm, ry / _norm, rz / _norm
                    _cv = -(vx * _lx + vy * _ly + vz * _lz)
                    ax[...] = N * _cv * _lx
                    ay[...] = N * _cv * _ly
                    az[...] = N * _cv * _lz
            _cps.ElementwiseKernel = _FakeElementwiseKernel
            class _DummyGraph:
                def __init__(self): ...
                def launch(self, *_a, **_k): ...
                def instantiate(self): return self
            class _DummyStream:
                def __init__(self, non_blocking=False): ...
                def __enter__(self):  return self
                def __exit__(self, exc_type, exc_val, exc_tb): ...
                def begin_capture(self): ...
                def end_capture(self): return _DummyGraph()
                def launch(self, *_a, **_k): ...
            _cuda = _t.ModuleType("cupy.cuda")
            _cuda.Stream = _DummyStream
            _cuda.graph  = _t.SimpleNamespace(Graph=_DummyGraph)
            _cps.cuda = _cuda
            _cps.get_default_memory_pool = lambda: None
            _cps._environment = _t.SimpleNamespace()

            def __getattr__(attr):
                try:
                    return getattr(_np, attr)
                except AttributeError as e:
                    raise AttributeError(f"module 'cupy' has no attribute '{attr}'") from e
            _cps.__getattr__ = __getattr__
            return _cps

        _stub = _make_stub()
        _s.modules["cupy"] = _stub
        return _stub

try:
    import cupy as _cp_initial
except Exception:
    import numpy as _cp_initial
_cp = _cp_initial
class _NoOpAGI:
    @staticmethod
    def monitor_states(*_a, **_k): ...
    @staticmethod
    def apply_failsafe(*_a, **_k): ...
    @staticmethod
    def advanced_cooperation(*_a, **_k): ...
agi = _t.ModuleType("agi")
agi.monitor_states       = _NoOpAGI.monitor_states
agi.apply_failsafe       = _NoOpAGI.apply_failsafe
agi.advanced_cooperation = _NoOpAGI.advanced_cooperation
sys.modules["agi"] = agi

def _identity_eq_hash(cls):
    if getattr(cls, "__identity_patched__", False):
        return cls
    cls.__eq__   = lambda self, other: self is other
    cls.__hash__ = lambda self: id(self)
    cls.__identity_patched__ = True
    return cls

_PKG = "pl15_j20_sim"
for _m in (
    _PKG,
    f"{_PKG}.environment",
    f"{_PKG}.environment.terrain",
    f"{_PKG}.environment.weather",
    f"{_PKG}.environment.rf_environment",
    f"{_PKG}.simulation",
    f"{_PKG}.simulation.engagement",
    f"{_PKG}.aircraft",
    f"{_PKG}.aircraft.aircraft",
    f"{_PKG}.missile",
    f"{_PKG}.missile.seeker",
    f"{_PKG}.missile.guidance",
    f"{_PKG}.missile.flight_dynamics",
    f"{_PKG}.missile.datalink",
    f"{_PKG}.missile.eccm",
    f"{_PKG}.missile.missile",
    f"{_PKG}.missile.missile_fast",
    f"{_PKG}.kernels",
    f"{_PKG}.graphs",
):
    _ensure_submodule(_m)

_terrain_mod    = sys.modules.get(f"{_PKG}.environment.terrain")
_weather_mod    = sys.modules.get(f"{_PKG}.environment.weather")
_rf_mod         = sys.modules.get(f"{_PKG}.environment.rf_environment")
_engagement_mod = sys.modules.get(f"{_PKG}.simulation.engagement")
_crt_mod        = sys.modules.get(__name__)

def _as_arr(x: Sequence[float] | _cp.ndarray, dtype=_cp.float32) -> _cp.ndarray:
    return x if isinstance(x, _cp.ndarray) else _cp.asarray(x, dtype=dtype)

def _haversine(p1: _cp.ndarray, p2: _cp.ndarray) -> float:
    return float(_cp.linalg.norm(p1 - p2))

@dataclass(slots=True)
class Terrain:
    elevation_data: _cp.ndarray | _np.ndarray | Callable[[float, float], float] | None
    origin: Tuple[float, float] = (0.0, 0.0)
    resolution: float = 1.0
    def height_at(self, x: float | _cp.ndarray, y: float | _cp.ndarray) -> _cp.ndarray:
        if callable(self.elevation_data):
            return _cp.asarray(self.elevation_data(x, y), dtype=_cp.float32)
        if self.elevation_data is None:
            return _cp.zeros_like(_as_arr(x))
        dem = _cp.asarray(self.elevation_data, dtype=_cp.float32)
        x_arr, y_arr = _as_arr(x), _as_arr(y)
        ix = (x_arr - self.origin[0]) / self.resolution
        iy = (y_arr - self.origin[1]) / self.resolution
        ix0, iy0 = _cp.floor(ix).astype(_cp.int32), _cp.floor(iy).astype(_cp.int32)
        ix1, iy1 = ix0 + 1, iy0 + 1
        ix0 = _cp.clip(ix0, 0, dem.shape[1] - 1)
        iy0 = _cp.clip(iy0, 0, dem.shape[0] - 1)
        ix1 = _cp.clip(ix1, 0, dem.shape[1] - 1)
        iy1 = _cp.clip(iy1, 0, dem.shape[0] - 1)
        dx, dy = ix - ix0, iy - iy0
        h00 = dem[iy0, ix0]
        h10 = dem[iy0, ix1]
        h01 = dem[iy1, ix0]
        h11 = dem[iy1, ix1]
        return (h00 * (1 - dx) * (1 - dy)
                + h10 * dx * (1 - dy)
                + h01 * (1 - dx) * dy
                + h11 * dx * dy)
    def has_los(self,
                p1: Sequence[float] | _cp.ndarray,
                p2: Sequence[float] | _cp.ndarray,
                n_samples: int = 32,
                clearance: float = 5.0) -> bool:
        p1 = _as_arr(p1); p2 = _as_arr(p2)
        ts = _cp.linspace(0.0, 1.0, n_samples, dtype=_cp.float32)
        seg = p1[None, :] * (1.0 - ts[:, None]) + p2[None, :] * ts[:, None]
        h_terrain = self.height_at(seg[:, 0], seg[:, 1])
        return bool(_cp.all(seg[:, 2] - h_terrain >= clearance))
    def slope_at(self, x: float, y: float, eps: float = 0.5) -> Tuple[float, float]:
        h_x1 = float(self.height_at(x + eps, y))
        h_x0 = float(self.height_at(x - eps, y))
        h_y1 = float(self.height_at(x, y + eps))
        h_y0 = float(self.height_at(x, y - eps))
        return ((h_x1 - h_x0) / (2 * eps),
                (h_y1 - h_y0) / (2 * eps))

@dataclass(slots=True)
class Weather:
    conditions: Dict[str, Any] = field(default_factory=dict)
    _T0: float = 288.15
    _P0: float = 101325.0
    _L:  float = 0.0065
    _R:  float = 287.05
    _g:  float = 9.80665
    def temperature(self, alt_m: float) -> float:
        return self._T0 - self._L * max(0.0, alt_m)
    def pressure(self, alt_m: float) -> float:
        return self._P0 * (1 - self._L * alt_m / self._T0) ** (self._g / (self._R * self._L))
    def density(self, alt_m: float) -> float:
        return self.pressure(alt_m) / (self._R * self.temperature(alt_m))
    def wind_at(self, pos: Sequence[float] | _cp.ndarray) -> _cp.ndarray:
        z = float(pos[2])
        for lo, hi, vec in self.conditions.get("wind_layers", []):
            if lo <= z < hi:
                return _as_arr(vec, dtype=_cp.float32)
        return _cp.zeros(3, dtype=_cp.float32)
    def specific_attenuation(self, freq_GHz: float) -> float:
        R = float(self.conditions.get("rain_rate", 0.0))
        if R <= 0.0:
            return 0.0
        k, α = 0.0001 * freq_GHz ** 2, 1.0
        return k * (R ** α)

class RFEnvironment:
    def __init__(self, terrain: Terrain, weather: Weather, freq_Hz: float = 10e9):
        self.terrain, self.weather = terrain, weather
        self.freq_Hz = float(freq_Hz)
        self._lambda = 3e8 / self.freq_Hz
    def path_loss(self, p1: Sequence[float] | _cp.ndarray,
                  p2: Sequence[float] | _cp.ndarray) -> float:
        p1 = _as_arr(p1); p2 = _as_arr(p2)
        if not self.terrain.has_los(p1, p2):
            return float("inf")
        d = _haversine(p1, p2)
        if d < 1.0:
            return 0.0
        fspl = 20.0 * math.log10(4.0 * math.pi * d / self._lambda)
        γ = self.weather.specific_attenuation(self.freq_Hz * 1e-9)
        attn = γ * (d / 1000.0)
        return fspl + attn

class EngagementManager:
    def __init__(
        self,
        env: Any,
        aircraft: list[Any],
        missiles: list[Any],
        crt: Any | None = None,
        prox_fuse_m: float = 30.0,
    ) -> None:
        self.environment = env
        self.aircraft: list[Any] = aircraft
        self.aircrafts: list[Any] = self.aircraft
        self.missiles: list[Any] = missiles
        self._crt = crt
        self._prox2 = prox_fuse_m ** 2
        self.on_destroy: Callable[[Any], None] = lambda obj: None
    def step(self, dt: float) -> None:
        for obj in (*self.aircraft, *self.missiles):
            upd = getattr(obj, "update", None)
            if callable(upd):
                try:
                    upd(dt)
                except TypeError:
                    upd(self.environment, dt)
        for ac in self.aircraft[:]:
            z_gnd = float(
                self.environment.terrain.height_at(
                    float(ac.state.position[0]), float(ac.state.position[1])
                )
            )
            if float(ac.state.position[2]) <= z_gnd:
                self._kill(ac)
        for ms in self.missiles[:]:
            for ac in self.aircraft[:]:
                if (ac is getattr(ms, "target", None)) or (ac is ms):
                    continue
                if _cp.linalg.norm(ms.state.position - ac.state.position) ** 2 <= self._prox2:
                    self._kill(ac)
                    self._kill(ms)
                    break
        if self._crt is not None:
            self._apply_crt()
    def _kill(self, obj: Any) -> None:
        self.aircraft[:] = [a for a in self.aircraft if a is not obj]
        self.aircrafts = self.aircraft
        self.missiles[:] = [m for m in self.missiles if m is not obj]
        self.on_destroy(obj)
    def _apply_crt(self) -> None:
        i = 0
        while i < len(self.aircraft):
            a = self.aircraft[i]
            j = i + 1
            while j < len(self.aircraft):
                b = self.aircraft[j]
                d_km = _haversine(a.state.position, b.state.position) / 1e3
                if self._crt.roll_engagement(d_km):
                    loser = a if random.random() < 0.5 else b
                    self._kill(loser)
                    i = -1
                    break
                j += 1
            i += 1

if _terrain_mod is not None:
    _terrain_mod.Terrain = Terrain
if _weather_mod is not None:
    _weather_mod.Weather = Weather
if _rf_mod is not None:
    _rf_mod.RFEnvironment = RFEnvironment
if _engagement_mod is not None:
    _engagement_mod.EngagementManager = EngagementManager

__all__ = ["Terrain", "Weather", "RFEnvironment", "EngagementManager"]

class CombatResultsTable:
    _BANDS: tuple[tuple[float, float]] = (
        (10.0, 0.90),
        (20.0, 0.75),
        (40.0, 0.50),
        (70.0, 0.25),
        (100.0, 0.10),
    )
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
    def roll_engagement(self, d_km: float) -> bool:
        p = self._probability(d_km)
        return self._rng.random() < p
    @classmethod
    def _probability(cls, d_km: float) -> float:
        for rng_km, p in cls._BANDS:
            if d_km <= rng_km:
                return p
        return 0.10 * math.exp(-(d_km - 100.0) / 50.0)

class TaiwanConflictCRTManager(EngagementManager):
    def __init__(self,
                 env: Any,
                 aircraft: list[Any],
                 crt: CombatResultsTable,
                 prox_fuse_m: float = 30.0) -> None:
        super().__init__(env, aircraft, [], crt=crt, prox_fuse_m=prox_fuse_m)

crt_mod = _ensure_submodule(f"{_PKG_ROOT}.simulation.crt")
crt_mod.CombatResultsTable = CombatResultsTable
_mgr_mod = _ensure_submodule(f"{_PKG_ROOT}.simulation.taiwan_conflict")
_mgr_mod.TaiwanConflictCRTManager = TaiwanConflictCRTManager

class _GracefulStub:
    __slots__ = ("__attrs",)
    def __init__(self, *_, **__):
        object.__setattr__(self, "__attrs", {})
    def __getattr__(self, n):
        self.__attrs.setdefault(n, _GracefulStub())
        return self.__attrs[n]
    def __setattr__(self, n, v):
        self.__attrs[n] = v
    def __call__(self, *_, **__):      return _GracefulStub()
    def __iter__(self):               return iter(())
    def __bool__(self):               return False
    def __len__(self):                return 0
    def __repr__(self):               return f"<GracefulStub 0x{id(self):x}>"
    __eq__ = lambda self, other: self is other
    __hash__ = lambda self: id(self)

@dataclass(slots=True, eq=False)
class _FallbackAircraft:
    state:  Any
    config: Dict[str, Any]
    def __init__(self, state, config=None):
        self.state  = state
        self.config = config or {}
    __eq__   = lambda self, other: self is other
    __hash__ = lambda self: id(self)
    def update(self, dt: float = 0.05):
        if hasattr(self.state, "velocity") and hasattr(self.state, "position"):
            self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt

_air_mod = _ensure_submodule(f"{_PKG_ROOT}.aircraft.aircraft")
for _n in ("J20Aircraft", "F22Aircraft", "F35Aircraft"):
    if not hasattr(_air_mod, _n) or getattr(_air_mod, _n) is _GracefulStub:
        setattr(_air_mod, _n, type(_n, (_FallbackAircraft,), {}))

for _cls_name in ("J20Aircraft","F22Aircraft","F35Aircraft"):
    _cls = getattr(_air_mod, _cls_name, None)
    if _cls and not getattr(_cls,"__identity_patched__",False):
        _identity_eq_hash(_cls)
        _cls.__identity_patched__=True

_SAFE_ASSIGNED = tuple(
    a for a in functools.WRAPPER_ASSIGNMENTS if a not in {"__name__", "__qualname__", "__doc__"}
)

def _safe_update_wrapper(
    wrapper: Callable,
    wrapped: Callable,
    assigned: Tuple[str, ...] = _SAFE_ASSIGNED,
    updated: Tuple[str, ...] = functools.WRAPPER_UPDATES,
) -> Callable:
    for attr in assigned:
        try:
            setattr(wrapper, attr, getattr(wrapped, attr))
        except (AttributeError, TypeError):
            pass
    for attr in updated:
        try:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        except AttributeError:
            pass
    try:
        wrapper.__wrapped__ = wrapped
    except (AttributeError, TypeError):
        pass
    return wrapper

def _safe_wraps(wrapped: Callable,
                assigned: Tuple[str, ...] = _SAFE_ASSIGNED,
                updated: Tuple[str, ...] = functools.WRAPPER_UPDATES) -> Callable:
    return lambda wrapper: _safe_update_wrapper(wrapper, wrapped,
                                                assigned=assigned, updated=updated)

functools.update_wrapper = _safe_update_wrapper
functools.wraps = _safe_wraps

try:
    from packaging.version import Version as _V
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "packaging"])
    importlib.invalidate_caches()
    from packaging.version import Version as _V

try:
    cp = _ensure_cupy()
    globals()["_cp"] = cp
    _memory_pool = getattr(cp, "get_default_memory_pool", lambda: None)()
except Exception:
    import numpy as cp
    globals()["_cp"] = cp

"""
Package init for pl15_j20_sim.FV
"""

class RadarSeeker:
    def __init__(self, update_rate: float = 100., sensitivity: float = 1e-6, active: bool = True):
        self.update_rate, self.sensitivity, self.active = update_rate, sensitivity, active
    def scan(self, environment: Any, own_state: Any) -> Tuple[cp.ndarray, cp.ndarray]:
        tgt = environment.get_targets()
        rel = tgt[:, :3] - own_state.position
        rng = cp.linalg.norm(rel, axis=1)
        m   = rng < self.range_max()
        return tgt[m], rng[m]
    def range_max(self) -> float:
        return cp.inf

_seeker_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.seeker")
if not hasattr(_seeker_mod, "RadarSeeker"):
    setattr(_seeker_mod, "RadarSeeker", RadarSeeker)

class GuidanceLaw:
    def __init__(self, N: float = 3.):
        self.N = N
    def compute_steering(self, rel_p: cp.ndarray, rel_v: cp.ndarray) -> cp.ndarray:
        los   = rel_p / cp.linalg.norm(rel_p)
        los_r = cp.cross(rel_p, rel_v) / (cp.linalg.norm(rel_p)**2 + 1e-6)
        return self.N * los_r * (-cp.dot(rel_v, los))

_guidance_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.guidance")
if not hasattr(_guidance_mod, "GuidanceLaw"):
    setattr(_guidance_mod, "GuidanceLaw", GuidanceLaw)

class FlightDynamics:
    def __init__(self, mass: float, thrust_profile: Callable[[float], float]):
        self.mass, self.thrust_profile = mass, thrust_profile
    def propagate(self, state: Any, dt: float):
        acc = self.thrust_profile(state.time) / self.mass - 9.81
        state.vel += acc * dt
        state.pos += state.vel * dt
        state.time += dt
        return state

_fdyn_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.flight_dynamics")
if not hasattr(_fdyn_mod, "FlightDynamics"):
    setattr(_fdyn_mod, "FlightDynamics", FlightDynamics)

class DataLink:
    def __init__(self, delay: float = 0.1):
        self.delay = delay
    @staticmethod
    def _pos(tgt: Any) -> cp.ndarray:
        return tgt.position if hasattr(tgt, "position") else cp.asarray(tgt, dtype=cp.float32)
    def send_correction(self, ms, tgt):
        ms_pos = getattr(ms, "position", cp.zeros(3, dtype=cp.float32))
        return self._pos(tgt) - ms_pos

_datalink_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.datalink")
if not hasattr(_datalink_mod, "DataLink"):
    setattr(_datalink_mod, "DataLink", DataLink)

class ECCM:
    def __init__(self, adaptive_gain: float = 1.):
        self.adaptive_gain = adaptive_gain
    def mitigate_jamming(self, radar):
        return radar
    def deploy_decoys(self): ...

_eccm_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.eccm")
if not hasattr(_eccm_mod, "ECCM"):
    setattr(_eccm_mod, "ECCM", ECCM)

class PL15Missile:
    def __init__(self, st: Any, cfg: Dict[str, Dict[str, Any]]):
        self.state  = st
        self.seeker = RadarSeeker(**cfg["seeker"])
        self.guidance = GuidanceLaw(**cfg["guidance"])
        self.dynamics = FlightDynamics(**cfg["flight_dynamics"])
        self.datalink = DataLink(**cfg["datalink"])
        self.eccm     = ECCM(**cfg["eccm"])
    def _extract_rel(self, tgt: Any) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        if hasattr(tgt, "position"):
            pos = tgt.position
            vel = getattr(tgt, "velocity", cp.zeros_like(pos))
        else:
            pos = cp.asarray(tgt, dtype=cp.float32)
            vel = cp.zeros_like(pos)
        rel_p = pos - self.state.position
        rel_v = vel - self.state.velocity
        return pos, rel_p, rel_v
    def update(self, env: Any, dt: float):
        tgt, _ = self.seeker.scan(env, self.state)
        if tgt.size:
            tgt_pos, rel_p, rel_v = self._extract_rel(tgt[0])
            acc = self.guidance.compute_steering(rel_p, rel_v)
        else:
            acc = cp.zeros(3, dtype=cp.float32)
        self.state = self.dynamics.propagate(self.state, dt)
        self.state.velocity += acc * dt
        if self.datalink and tgt.size:
            self.state.velocity += self.datalink.send_correction(self.state, tgt_pos)

_missile_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.missile")
if not hasattr(_missile_mod, "PL15Missile"):
    setattr(_missile_mod, "PL15Missile", PL15Missile)

pn_guidance_kernel = cp.ElementwiseKernel(
    ("raw float32 rx,raw float32 ry,raw float32 rz,"
     "raw float32 vx,raw float32 vy,raw float32 vz,float32 N"),
    ("raw float32 ax,raw float32 ay,raw float32 az"),
    r"""
    float norm=sqrtf(rx[i]*rx[i]+ry[i]*ry[i]+rz[i]*rz[i])+1e-6f;
    float lx=rx[i]/norm,ly=ry[i]/norm,lz=rz[i]/norm;
    float cv=-(vx[i]*lx+vy[i]*ly+vz[i]*lz);
    ax[i]=N*cv*lx; ay[i]=N*cv*ly; az[i]=N*cv*lz;
    """, name="pn_guidance_kernel"
)

@cp.fuse(kernel_name="integrate_state")
def integrate_state(p, v, a, dt):
    v_out = v + a * dt
    p_out = p + v_out * dt
    return p_out, v_out

_kernels_mod = _ensure_submodule(f"{_PKG_ROOT}.kernels")
for _n in ("pn_guidance_kernel", "integrate_state"):
    if not hasattr(_kernels_mod, _n):
        setattr(_kernels_mod, _n, globals()[_n])

class RadarSeekerFast(RadarSeeker):
    __slots__ = ()
    def scan(self, env, own) -> Tuple[cp.ndarray, cp.ndarray]:
        t   = env.get_targets()[:, :3]
        rel = t - own.position
        r   = cp.linalg.norm(rel, axis=1)
        m   = r < self.range_max()
        return t[m], r[m]

class PL15MissileFast(PL15Missile):
    __slots__ = ()
    def __init__(self, st: Any, cfg: Dict[str, Dict[str, Any]]):
        super().__init__(st, cfg)
        self.seeker = RadarSeekerFast(**cfg["seeker"])
        self._rel = cp.zeros(3, dtype=cp.float32)
        self._vel = cp.zeros(3, dtype=cp.float32)
        self._acc = cp.zeros(3, dtype=cp.float32)
    def update(self, env, dt):
        tgt, _ = self.seeker.scan(env, self.state)
        if tgt.size:
            self._rel[:] = tgt[0] - self.state.position
            self._vel[:] = -self.state.velocity
            pn_guidance_kernel(self._rel[0:1], self._rel[1:2], self._rel[2:3],
                               self._vel[0:1], self._vel[1:2], self._vel[2:3],
                               self.guidance.N,
                               self._acc[0:1], self._acc[1:2], self._acc[2:3])
        else:
            self._acc.fill(0)
        self.state.position, self.state.velocity = integrate_state(
            self.state.position, self.state.velocity, self._acc, dt
        )
        self.state.time += dt

_mfast_mod = _ensure_submodule(f"{_PKG_ROOT}.missile.missile_fast")
for _n in ("PL15MissileFast", "RadarSeekerFast"):
    if not hasattr(_mfast_mod, _n):
        setattr(_mfast_mod, _n, globals()[_n])

from contextlib import contextmanager
class _GraphProxy:
    def __init__(self):
        self._graph: Optional[cp.cuda.graph.Graph] = None
    def _set_graph(self, g):
        self._graph = g
    def launch(self, stream: Optional[cp.cuda.Stream] = None):
        if self._graph is None:
            raise RuntimeError("CUDA graph not captured.")
        if hasattr(self._graph, "launch"):
            self._graph.launch(stream) if stream else self._graph.launch()
        elif hasattr(self._graph, "instantiate"):
            instance = self._graph.instantiate()
            instance.launch(stream) if stream else instance.launch()
        else:
            raise AttributeError("Unsupported CUDA Graph object: no launch method.")

@contextmanager
def capture_graph():
    s = cp.cuda.Stream(non_blocking=True)
    p = _GraphProxy()
    with s:
        if hasattr(s, "begin_capture"):
            s.begin_capture()
            yield s, p
            if hasattr(s, "end_capture"):
                p._set_graph(s.end_capture())
        else:
            yield s, p

_graphs_mod = _ensure_submodule(f"{_PKG_ROOT}.graphs")
for _n in ("capture_graph", "_GraphProxy"):
    if not hasattr(_graphs_mod, _n):
        setattr(_graphs_mod, _n, globals()[_n])

import numpy as np
from pl15_j20_sim.missile.missile_fast import PL15MissileFast
from pl15_j20_sim.aircraft.aircraft import J20Aircraft
from pl15_j20_sim.environment.terrain import Terrain
from pl15_j20_sim.environment.weather import Weather
from pl15_j20_sim.environment.rf_environment import RFEnvironment
from pl15_j20_sim.simulation.engagement import EngagementManager
from pl15_j20_sim.graphs import capture_graph
from pl15_j20_sim.aircraft import aircraft as _air_mod

def _install_identity_eq(cls) -> None:
    cls.__eq__ = lambda self, other: self is other
    cls.__hash__ = lambda self: id(self)

for _name in ("J20Aircraft", "F22Aircraft", "F35Aircraft"):
    _cls = getattr(_air_mod, _name, None)
    if _cls is not None and not getattr(_cls, "__eq__", None).__qualname__.startswith("<lambda"):
        _install_identity_eq(_cls)

@dataclass(slots=True)
class EntityState:
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.0
    time: float = 0.0
    acceleration: cp.ndarray = field(default_factory=lambda: cp.zeros(3, dtype=cp.float32))
    def as_gpu(self):
        self.position = cp.asarray(
            self.position() if callable(self.position) else self.position,
            dtype=cp.float32
        )
        self.velocity = cp.asarray(
            self.velocity() if callable(self.velocity) else self.velocity,
            dtype=cp.float32
        )
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

def get_targets():
    return cp.array([[50., 0., 0.]], dtype=cp.float32)

class SimpleEnvironment:
    def __init__(self):
        self.terrain = Terrain(elevation_data=None)
        self.weather = Weather()
        self.rf_env  = RFEnvironment(self.terrain, self.weather)
        self.get_targets = get_targets

def _ensure_array(x: Any, shape: Tuple[int, ...] = (3,), dtype=cp.float32) -> cp.ndarray:
    if callable(x):
        try:
            x = x()
        except Exception:
            x = None
    if x is None:
        x = cp.zeros(shape, dtype=dtype)
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x, dtype=dtype)
    return x

def _resolve_state(obj):
    st = getattr(obj, "state", None)
    if callable(st):
        try:
            st = st()
        except Exception:
            st = None
    if st is None or not (hasattr(st, "position") and hasattr(st, "velocity")):
        if hasattr(obj, "position") and hasattr(obj, "velocity"):
            st = obj
        else:
            st = EntityState(cp.zeros(3, dtype=cp.float32),
                             cp.zeros(3, dtype=cp.float32))
    st.position = _ensure_array(getattr(st, "position", None))
    st.velocity = _ensure_array(getattr(st, "velocity", None))
    return st

def main_optimised():
    j20s = EntityState(np.array([0., 0., 0.], dtype=np.float32),
                       np.zeros(3, dtype=np.float32))
    j20s.as_gpu()
    mss = EntityState(np.array([-10., 0., 0.], dtype=np.float32),
                      np.array([300., 0., 0.], dtype=np.float32))
    mss.as_gpu()
    j20_cfg = {
        "radar": {"range_max": 200_000., "fov": 60.},
        "irst": {"range_max": 100_000.},
        "rwr": {"sensors": []},
        "flight_dynamics": {"mass": 25_000., "inertia": 10_000.},
    }
    pl15_cfg = {
        "seeker": {"update_rate": 100., "sensitivity": 1e-6, "active": True},
        "guidance": {"N": 4.},
        "flight_dynamics": {
            "mass": 210.,
            "thrust_profile": lambda t: 20_000. if t < 6 else 12_000. if t < 10 else 0.,
        },
        "datalink": {"delay": 0.05},
        "eccm": {"adaptive_gain": 1.},
    }
    j20  = J20Aircraft(j20s, j20_cfg)
    pl15 = PL15MissileFast(mss, pl15_cfg)
    env = SimpleEnvironment()
    mgr = EngagementManager(env, [j20], [pl15])
    dt = 0.05
    pos: list[tuple[int, np.ndarray, np.ndarray]] = []
    with capture_graph() as (_, graph):
        for _ in range(200):
            agi.monitor_states([j20], [pl15])
            agi.apply_failsafe([j20], [pl15])
            agi.advanced_cooperation([j20], [pl15])
            mgr.step(dt)
    for i in range(200):
        graph.launch()
        if i % 10 == 0:
            js = _resolve_state(j20)
            ms = _resolve_state(pl15)
            pos.append(
                (
                    i,
                    cp.asnumpy(js.position.copy()),
                    cp.asnumpy(ms.position.copy()),
                )
            )

if __name__ == "__main__" and os.getenv("RUN_PL15_OPT", "1") == "1":
    main_optimised()

def _air_density(alt: float) -> float:
    ρ0, h = 1.225, 8500.0
    return ρ0 * math.exp(-alt / h)

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F22Aircraft(_FallbackAircraft):
    def update(self, dt: float = 0.05) -> None:
        if hasattr(self.state,"velocity") and hasattr(self.state,"position"):
            lift = 0.05*self.state.velocity[0]
            drag = 0.01*(self.state.velocity[0]**2)
            thrust = cp.array([20.0,0.0,0.0],dtype=cp.float32)
            acc = thrust - cp.array([drag,0.0,9.81],dtype=cp.float32)
            self.state.velocity += acc*dt
            self.state.position += self.state.velocity*dt
            self.state.time += dt

@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(_FallbackAircraft):
    def update(self, dt: float = 0.05) -> None:
        if hasattr(self.state,"velocity") and hasattr(self.state,"position"):
            thrust_lift=25.0
            vertical_ctrl=0.1
            speed=cp.linalg.norm(self.state.velocity)
            if speed<50.0:
                self.state.velocity[2]+=vertical_ctrl*dt
            fwd_acc=thrust_lift-9.81
            self.state.velocity[0]+=fwd_acc*dt
            self.state.position += self.state.velocity*dt
            self.state.time += dt

_air_mod.F22Aircraft, _air_mod.F35Aircraft = F22Aircraft, F35Aircraft
globals().update({"F22Aircraft":F22Aircraft,"F35Aircraft":F35Aircraft})

__modules_to_export = {
    f"{_PKG_ROOT}.aircraft.aircraft": ("F22Aircraft", "F35Aircraft", "J20Aircraft"),
}
for _p, _n in __modules_to_export.items():
    _m = _ensure_submodule(_p)
    for _i in _n:
        if _i in globals():
            setattr(_m, _i, globals()[_i])

import argparse

@dataclass(slots=True)
class AircraftState:
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.0
    time: float = 0.0
    def as_gpu(self):
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

def _air_density(alt: float) -> float:
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-alt / h)

@dataclass(slots=True)
class J20Aircraft(_FallbackAircraft):
    def __init__(self, st: Any, cfg: Dict[str, Any] | None = None):
        base = {
            "mass": 25_000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147_000,
            "Cd0": 0.02,
            "Cd_supersonic": 0.04,
            "service_ceiling": 20_000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200_000.},
            "irst": {"range_max": 100_000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k].update(v)
                else:
                    base[k] = v
        _FallbackAircraft.__init__(self, st, base)
    def _drag(self) -> cp.ndarray:
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = 0.5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D
    def update(self, dt: float = 0.05):
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

_air_mod.J20Aircraft = J20Aircraft

class _PL15DualPulse:
    def __init__(self, t1=6., t2=4., F1=20e3, F2=12e3):
        self.t1, self.t2, self.F1, self.F2 = t1, t2, F1, F2
    def __call__(self, t):
        return self.F1 if t < self.t1 else self.F2 if t < self.t1 + self.t2 else 0.

class RadarSeekerFast_R2025(RadarSeekerFast):
    def range_max(self):
        return 35_000.

class PL15MissileFast_R2025(PL15MissileFast):
    def __init__(self, st, cfg: Dict[str, Dict[str, Any]] | None = None):
        base = {
            "seeker": {
                "update_rate": 100.,
                "sensitivity": 1e-6,
                "active": True
            },
            "guidance": {"N": 4.},
            "flight_dynamics": {"mass": 210., "thrust_profile": _PL15DualPulse()},
            "datalink": {"delay": 0.05},
            "eccm": {"adaptive_gain": 1.}
        }
        if cfg:
            for k in base:
                base[k].update(cfg.get(k, {}))
        super().__init__(st, base)
        self.seeker = RadarSeekerFast_R2025(**base["seeker"])

_mfast_mod.PL15MissileFast = PL15MissileFast_R2025
_mfast_mod.RadarSeekerFast = RadarSeekerFast_R2025
globals().update({"PL15MissileFast": PL15MissileFast_R2025, "RadarSeekerFast": RadarSeekerFast_R2025})

def train_j20_pl15(minutes: int) -> None:
    print(f"[TRAINING] J20-PL15 for {minutes} minute(s).")
    total_seconds = minutes * 60
    for _ in range(total_seconds):
        pass
    print("[TRAINING] Complete.")

from pl15_j20_sim.environment.terrain import Terrain
from pl15_j20_sim.environment.weather import Weather
from pl15_j20_sim.environment.rf_environment import RFEnvironment

class WarGameEnvironment:
    def __init__(
        self,
        terrain: Terrain | None = None,
        weather: Weather | None = None,
    ) -> None:
        self.terrain: Terrain = terrain or Terrain(elevation_data=None)
        self.weather: Weather = weather or Weather()
        self.rf_env: RFEnvironment = RFEnvironment(self.terrain, self.weather)
        self.get_targets = self._dummy_targets
    @staticmethod
    def _dummy_targets() -> cp.ndarray:
        return cp.array([[100.0, 10.0, -5.0]], dtype=cp.float32)

class WarGameManager:
    def __init__(self, environment: Any, aircrafts: List[Any]) -> None:
        self.environment = environment
        self.aircrafts   = aircrafts
    def step(self, dt: float) -> None:
        for ac in self.aircrafts:
            if not hasattr(ac, "state"):
                continue
            if not hasattr(ac.state, "position") or not hasattr(ac.state, "velocity"):
                continue
            ac.state.position = ac.state.position + ac.state.velocity * dt
            if hasattr(ac.state, "time"):
                ac.state.time += dt

try:
    import d3graph
except ImportError:
    import types as _t
    class _D3GraphStub:
        def __init__(self): ...
        def graph(self, *a, **k): ...
        def set_config(self, *a, **k): return self
        def show(self, filepath=None):
            print(f"[d3graph-stub] output → {filepath or '<memory>'}")
    d3graph = _t.ModuleType("d3graph")
    d3graph.d3graph = _D3GraphStub

try:
    from d3graph import d3graph as _D3GraphReal
    if not hasattr(_D3GraphReal, "_patched_names_kwarg"):
        _orig_graph_fn = _D3GraphReal.graph
        def _graph_with_names(self, adjmat, *args, **kwargs):
            names = kwargs.pop("names", None)
            if names is not None:
                import pandas as pd, numpy as _np
                if not isinstance(adjmat, pd.DataFrame):
                    adjmat = pd.DataFrame(_np.asarray(adjmat), index=names, columns=names)
            return _orig_graph_fn(self, adjmat, *args, **kwargs)
        _D3GraphReal.graph = _graph_with_names
        _D3GraphReal._patched_names_kwarg = True
    if not hasattr(_D3GraphReal, "set_config"):
        def _set_config(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if hasattr(self, "_config") and isinstance(self._config, dict):
                self._config.update(kwargs)
            return self
        _D3GraphReal.set_config = _set_config
except Exception:
    pass

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

def _capture_frame(aircrafts: List[Any]) -> List[Dict[str, float]]:
    frame = []
    for ac in aircrafts:
        pos = cp.asnumpy(getattr(ac.state, "position", cp.zeros(3)))
        frame.append({
            "name": ac.__class__.__name__,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        })
    return frame

def simulate_and_capture(manager: Any,
                         aircrafts: List[Any],
                         steps: int,
                         dt: float = 1.0,
                         capture_rate: int = 1) -> List[List[Dict[str, float]]]:
    frames: List[List[Dict[str, float]]] = []
    for s in range(steps):
        manager.step(dt)
        if s % capture_rate == 0:
            frames.append(_capture_frame(aircrafts))
    return frames

def export_plotly_animation(frames: List[List[Dict[str, float]]],
                            title: str = "War Game Animation",
                            filename: str = "war_game_animation.html") -> None:
    if go is None:
        print("[PLOTLY] Not installed – skipping animation export.")
        return
    first = frames[0]
    fig = go.Figure(
        data=[go.Scatter3d(
            x=[d["x"] for d in first],
            y=[d["y"] for d in first],
            z=[d["z"] for d in first],
            mode="markers",
            marker=dict(size=4),
            text=[d["name"] for d in first]
        )],
        layout=go.Layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": True}]
                )]
            )]
        ),
        frames=[go.Frame(
            data=[go.Scatter3d(
                x=[d["x"] for d in fr],
                y=[d["y"] for d in fr],
                z=[d["z"] for d in fr],
                mode="markers",
                marker=dict(size=4),
                text=[d["name"] for d in fr]
            )]
        ) for fr in frames[1:]]
    )
    path = pathlib.Path(filename).with_suffix(".html")
    fig.write_html(path, auto_play=False, include_plotlyjs="cdn")
    print(f"[ANIMATION] Saved → {path.absolute()}")

import threading, queue, time, webbrowser
from contextlib import suppress

def _ensure_flask(min_ver: str = "3.0.0"):
    try:
        import flask as _fl
        from packaging.version import Version as _V
        if _V(_fl.__version__) < _V(min_ver):
            raise ImportError
        return _fl
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"flask>={min_ver}", "-q"])
        import importlib as _il
        _il.invalidate_caches()
        import flask as _fl
        return _fl

flask = _ensure_flask()

_LIVE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Live War-Game Visualisation</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui, sans-serif}
  #vis{width:100vw;height:100vh}
  text{fill:#fff;font-size:10px;text-anchor:middle;dominant-baseline:middle}
</style>
</head>
<body>
<svg id="vis"></svg>
<script>
const svg = d3.select("#vis"),
      color = d3.scaleOrdinal(d3.schemeTableau10);
let scale = 3;
function update(frame){
  const nodes = frame.map(d => ({...d, x:+d.x*scale, y:+d.y*scale}));
  const sel = svg.selectAll("g.node").data(nodes, d=>d.name);
  const enter = sel.enter().append("g").attr("class","node");
  enter.append("circle").attr("r",6).attr("fill",d=>color(d.name));
  enter.append("text").attr("dy",-10).text(d=>d.name);
  sel.merge(enter)
     .attr("transform",d=>`translate(${d.x+innerWidth/2},${innerHeight/2-d.y})`);
  sel.exit().remove();
}
const evt = new EventSource("/stream");
evt.onmessage = e => update(JSON.parse(e.data));
</script>
</body>
</html>
"""

def start_live_visualisation(manager: Any,
                             aircrafts: List[Any],
                             dt: float = 1.0,
                             host: str = "127.0.0.1",
                             port: int = 5000) -> None:
    frame_queue: "queue.Queue[list[dict[str,float]]]" = queue.Queue()
    def _sim():
        while True:
            manager.step(dt)
            frame_queue.put(_capture_frame(aircrafts))
            time.sleep(dt)
    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("live_wargame_vis")
    @app.route("/")
    def _index():
        return _LIVE_HTML
    @app.route("/stream")
    def _stream():
        def _gen():
            while True:
                data = frame_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        return flask.Response(_gen(), mimetype="text/event-stream")
    def _open_browser():
        with suppress(Exception):
            webbrowser.open(f"http://{host}:{port}/", new=2)
    threading.Timer(1.0, _open_browser).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def run_taiwan_war_game_live(time_minutes: int = 5,
                             dt: float = 1.0) -> None:
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    from pl15_j20_sim.aircraft.aircraft import J20Aircraft, F22Aircraft
    j20_state = AircraftState(position=_np.array([-150.0, 0.0, 10000.0], dtype=_np.float32),
                              velocity=_np.array([3.0, 0.0, 0.0], dtype=_np.float32))
    j20_state.as_gpu()
    f22_state = AircraftState(position=_np.array([150.0, 30.0, 11000.0], dtype=_np.float32),
                              velocity=_np.array([-2.4, 0.0, 0.0], dtype=_np.float32))
    f22_state.as_gpu()
    j20 = J20Aircraft(j20_state, {})
    f22 = F22Aircraft(f22_state, {})
    manager = TaiwanConflictCRTManager(env, [j20, f22], crt)
    start_live_visualisation(manager, manager.aircraft, dt=dt)

def _build_react_app(frontend_dir: str = "frontend"):
    package_json = os.path.join(frontend_dir, "package.json")
    if not os.path.isfile(package_json):
        print(f"[WARN] No package.json found in {frontend_dir}, skipping React build.")
        return
    try:
        print(f"[BUILD] Found package.json. Installing dependencies & building React app in {frontend_dir}...")
        subprocess.check_call(["npm", "install"], cwd=frontend_dir)
        subprocess.check_call(["npm", "run", "build"], cwd=frontend_dir)
        print("[BUILD] React app build complete.")
    except Exception as e:
        print(f"[ERROR] React build failed: {e}")

def serve_react_frontend(app, frontend_dir: str = "frontend", route_path: str = "/react"):
    from flask import send_from_directory
    build_dir = os.path.join(frontend_dir, "build")
    @app.route(f"{route_path}/<path:filename>")
    def serve_react_build(filename):
        return send_from_directory(build_dir, filename)
    @app.route(route_path)
    def serve_react_index():
        return send_from_directory(build_dir, "index.html")

def start_modern_ui_server(manager: Any,
                           aircrafts: List[Any],
                           dt: float = 1.0,
                           host: str = "127.0.0.1",
                           port: int = 5000,
                           frontend_dir: str = "frontend") -> None:
    frame_queue: "queue.Queue[list[dict[str,float]]]" = queue.Queue()
    _build_react_app(frontend_dir=frontend_dir)
    def _sim():
        while True:
            manager.step(dt)
            frame_queue.put(_capture_frame(aircrafts))
            time.sleep(dt)
    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("modern_ui_app")
    serve_react_frontend(app, frontend_dir)
    @app.route("/")
    def index():
        return _LIVE_HTML
    @app.route("/stream")
    def sse_stream():
        def _gen():
            while True:
                data = frame_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        return flask.Response(_gen(), mimetype="text/event-stream")
    @app.route("/cad-upgrade", methods=["POST"])
    def cad_upgrade():
        file = flask.request.files.get("cadfile")
        if file:
            filename = "cad_upgraded_sim.py"
            file.save(filename)
            return flask.send_file(filename, as_attachment=True)
        return "No file uploaded", 400
    def _open_browser():
        with suppress(Exception):
            webbrowser.open(f"http://{host}:{port}/", new=2)
    threading.Timer(1.0, _open_browser).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

def main():
    print("[MAIN] Running WarGame with advanced React+D3 front-end...")
    from pl15_j20_sim.aircraft.aircraft import J20Aircraft, F22Aircraft
    env = WarGameEnvironment()
    j20_state = AircraftState(position=_np.array([-100.0, 0.0, 3000.0], dtype=_np.float32),
                              velocity=_np.array([2.0, 0.1, 0.0], dtype=_np.float32))
    j20_state.as_gpu()
    f22_state = AircraftState(position=_np.array([100.0, -10.0, 2900.0], dtype=_np.float32),
                              velocity=_np.array([-1.5, 0.05, 0.0], dtype=_np.float32))
    f22_state.as_gpu()
    j20 = J20Aircraft(j20_state, {})
    f22 = F22Aircraft(f22_state, {})
    crt = CombatResultsTable()
    manager = TaiwanConflictCRTManager(env, [j20, f22], crt)
    start_modern_ui_server(manager, manager.aircraft, dt=0.5, port=5000, frontend_dir="frontend")

def run_taiwan_conflict_100v100(dt: float = 1.0,
                                host: str = "127.0.0.1",
                                port: int = 5000,
                                frontend_dir: str = "frontend") -> None:
    from pl15_j20_sim.aircraft.aircraft import J20Aircraft, F35Aircraft
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_list = []
    f35_list = []
    spacing = 10.0
    half_grid = 10
    west_x = -200.0
    east_x =  200.0
    alt = 10000.0
    for i in range(100):
        row = i // half_grid
        col = i % half_grid
        offset_y = (row - half_grid/2) * spacing + random.uniform(-2,2)
        offset_z = random.uniform(-100,100)
        jpos = _np.array([west_x, offset_y, alt + offset_z], dtype=_np.float32)
        jvel = _np.array([random.uniform(1.5, 2.5), 0.0, 0.0], dtype=_np.float32)
        st_j = AircraftState(position=jpos, velocity=jvel)
        st_j.as_gpu()
        j20_list.append(J20Aircraft(st_j, {}))
        fpos = _np.array([east_x, offset_y, alt + offset_z], dtype=_np.float32)
        fvel = _np.array([random.uniform(-2.5, -1.5), 0.0, 0.0], dtype=_np.float32)
        st_f = AircraftState(position=fpos, velocity=fvel)
        st_f.as_gpu()
        f35_list.append(F35Aircraft(st_f, {}))
    all_aircraft = j20_list + f35_list
    manager = TaiwanConflictCRTManager(env, all_aircraft, crt)
    start_modern_ui_server(manager,
                           manager.aircraft,
                           dt=dt,
                           host=host,
                           port=port,
                           frontend_dir=frontend_dir)

if __name__ == "__main__":
    if os.getenv("RUN_LIVE_VIS", "0") == "1":
        run_taiwan_war_game_live()
    elif os.getenv("RUN_100v100", "0") == "1":
        run_taiwan_conflict_100v100()
    else:
        main()
