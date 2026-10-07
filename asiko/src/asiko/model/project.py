"""Модель проекта ASIKO (недеструктивная: исходники только читаются).

Все времена — секунды (float). На шкале проекта (timeline) 0 — начало проекта;
время исходника (source) — от начала файла.
"""
from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field, fields, asdict
from typing import Any

from .presets import (
    BUILTIN_PRESETS,
    NEUTRAL_VALUES,
    REVERB_OFF_DB,
    SPACE_PARAMS,
    get_preset,
)

FORMAT_NAME = "asiko.project"
FORMAT_VERSION = 1


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _filter_kwargs(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


# --------------------------------------------------------------------------- sources


@dataclass
class BeatGrid:
    """Разметка долей исходника (время исходника, секунды)."""

    bpm: float
    beats: list[float]
    downbeats: list[float]
    beats_per_bar: int = 4
    confidence: float = 0.0          # общая уверенность 0..1
    tempo_confidence: float = 0.0
    downbeat_confidence: float = 0.0
    method: str = "auto"             # auto | manual | edited
    phrase_bars: int = 8
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        return cls(**_filter_kwargs(cls, d))

    @classmethod
    def manual(cls, bpm: float, first_downbeat: float, duration: float,
               beats_per_bar: int = 4, phrase_bars: int = 8) -> "BeatGrid":
        period = 60.0 / bpm
        n0 = -int(math.floor(first_downbeat / period))
        beats = []
        k = n0
        while True:
            t = first_downbeat + k * period
            if t > duration:
                break
            if t >= 0:
                beats.append(round(t, 6))
            k += 1
        downbeats = [b for i, b in enumerate(beats)
                     if round((b - first_downbeat) / period) % beats_per_bar == 0]
        return cls(bpm=float(bpm), beats=beats, downbeats=downbeats,
                   beats_per_bar=beats_per_bar, confidence=1.0, tempo_confidence=1.0,
                   downbeat_confidence=1.0, method="manual", phrase_bars=phrase_bars,
                   notes=["Сетка задана вручную"])

    def phrase_starts(self) -> list[float]:
        if not self.downbeats:
            return []
        step = max(1, self.phrase_bars)
        return self.downbeats[::step]


@dataclass
class Source:
    id: str
    name: str
    path: str
    sha256: str
    size: int
    sample_rate: int
    channels: int
    frames: int
    format: str = ""
    subtype: str = ""
    rel_path: str | None = None
    beats: BeatGrid | None = None
    phrases: list[float] = field(default_factory=list)   # ручные границы фраз (время исходника)
    kind: str = "audio"

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        d = dict(d)
        beats = d.pop("beats", None)
        s = cls(**_filter_kwargs(cls, d))
        s.beats = BeatGrid.from_dict(beats) if beats else None
        return s


# --------------------------------------------------------------------------- tracks & clips


@dataclass
class Clip:
    id: str
    source_id: str
    start: float          # время на шкале проекта
    src_in: float         # начало в исходнике
    src_out: float        # конец в исходнике (не включительно)
    name: str = ""
    gain_db: float = 0.0
    fade_in: float = 0.005
    fade_out: float = 0.005
    fade_shape: str = "smooth"
    locked: bool = False

    @property
    def length(self) -> float:
        return self.src_out - self.src_in

    @property
    def end(self) -> float:
        return self.start + self.length

    def to_src(self, t: float) -> float:
        return self.src_in + (t - self.start)

    def to_timeline(self, s: float) -> float:
        return self.start + (s - self.src_in)

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class Track:
    id: str
    name: str
    clips: list[Clip] = field(default_factory=list)
    gain_db: float = 0.0
    mute: bool = False
    solo: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        d = dict(d)
        clips = [Clip.from_dict(c) for c in d.pop("clips", [])]
        t = cls(**_filter_kwargs(cls, d))
        t.clips = clips
        return t


# --------------------------------------------------------------------------- space transitions


@dataclass
class SpaceState:
    preset: str
    name: str
    values: dict[str, float]
    room: str = "none"

    @classmethod
    def from_preset(cls, key: str, user_presets: dict | None = None,
                    overrides: dict | None = None) -> "SpaceState":
        p = get_preset(key, user_presets)
        if p is None:
            raise KeyError(key)
        values = dict(NEUTRAL_VALUES)
        values.update(p["values"])
        if overrides:
            values.update({k: float(v) for k, v in overrides.items()})
        return cls(preset=key, name=p["name"], values=values, room=p.get("room", "none"))

    def effective_reverb(self, room: str) -> float:
        """Уровень посыла в заданную комнату (дБ), −80 = нет."""
        if self.room == room and self.room != "none":
            return float(self.values.get("reverb_db", REVERB_OFF_DB))
        return REVERB_OFF_DB

    @classmethod
    def neutral(cls) -> "SpaceState":
        return cls(preset="offscreen", name=BUILTIN_PRESETS["offscreen"]["name"],
                   values=dict(NEUTRAL_VALUES), room="none")

    @classmethod
    def from_dict(cls, d: dict) -> "SpaceState":
        d = dict(d)
        vals = dict(NEUTRAL_VALUES)
        vals.update({k: float(v) for k, v in d.get("values", {}).items()})
        return cls(preset=d.get("preset", ""), name=d.get("name", ""), values=vals,
                   room=d.get("room", "none"))


@dataclass
class ParamCurve:
    """Когда меняется параметр внутри перехода: доли длительности перехода 0..1."""

    start: float = 0.0
    end: float = 1.0
    shape: str = "smooth"

    @classmethod
    def from_dict(cls, d: dict) -> "ParamCurve":
        return cls(**_filter_kwargs(cls, d))


def default_curves() -> dict[str, ParamCurve]:
    return {p: ParamCurve() for p in SPACE_PARAMS}


@dataclass
class SpaceTransition:
    id: str
    track_id: str
    start: float
    end: float
    from_state: SpaceState
    to_state: SpaceState
    curves: dict[str, ParamCurve] = field(default_factory=default_curves)
    filter_slope: int = 12
    anchor_clip_id: str | None = None
    locked_params: list[str] = field(default_factory=list)
    approved: bool = False
    name: str = ""

    @property
    def length(self) -> float:
        return self.end - self.start

    def curve_abs(self, param: str) -> tuple[float, float]:
        c = self.curves[param]
        L = self.length
        return self.start + c.start * L, self.start + c.end * L

    @classmethod
    def from_dict(cls, d: dict) -> "SpaceTransition":
        d = dict(d)
        fs = SpaceState.from_dict(d.pop("from_state"))
        ts = SpaceState.from_dict(d.pop("to_state"))
        curves_raw = d.pop("curves", {})
        curves = default_curves()
        for k, v in curves_raw.items():
            if k in curves:
                curves[k] = ParamCurve.from_dict(v)
        t = cls(from_state=fs, to_state=ts, **_filter_kwargs(cls, d))
        t.curves = curves
        return t


# --------------------------------------------------------------------------- music transitions


@dataclass
class MusicParams:
    switch_time: float           # время шкалы, где звучит точка входа B
    b_entry_src: float           # точка входа в исходнике B
    a_fade_start: float          # время шкалы
    a_fade_len: float
    b_fade_start: float          # время шкалы
    b_fade_len: float
    curve: str = "equal_power"   # equal_power | linear
    a_tail: str = "source"       # source | cut | effect
    effect_tail_room: str = "hall"
    overlap_filter: str = "none"  # none | hp_a | lp_a

    @property
    def a_fade_end(self) -> float:
        return self.a_fade_start + self.a_fade_len

    @property
    def b_fade_end(self) -> float:
        return self.b_fade_start + self.b_fade_len

    @property
    def overlap(self) -> tuple[float, float]:
        return min(self.a_fade_start, self.b_fade_start), max(self.a_fade_end, self.b_fade_end)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicParams":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class MusicVariant:
    id: str
    kind: str                    # short | long | phrase
    name: str
    params: MusicParams
    description: str = ""
    notes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "MusicVariant":
        d = dict(d)
        p = MusicParams.from_dict(d.pop("params"))
        v = cls(params=p, **_filter_kwargs(cls, d))
        return v


@dataclass
class MusicTransition:
    id: str
    clip_a_id: str
    clip_b_id: str
    target_time: float
    params: MusicParams
    variants: list[MusicVariant] = field(default_factory=list)
    selected_variant: str | None = None
    modified: bool = False
    b_on_downbeat: bool = True
    keep_a_tail: bool = True
    exact_time: bool = False
    approved: bool = False
    name: str = ""
    notes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    # исходное положение клипов до создания перехода (для удаления перехода)
    orig_a: dict = field(default_factory=dict)
    orig_b: dict = field(default_factory=dict)

    def variant(self, vid_or_kind: str) -> MusicVariant | None:
        for v in self.variants:
            if v.id == vid_or_kind or v.kind == vid_or_kind:
                return v
        return None

    @classmethod
    def from_dict(cls, d: dict) -> "MusicTransition":
        d = dict(d)
        p = MusicParams.from_dict(d.pop("params"))
        vs = [MusicVariant.from_dict(v) for v in d.pop("variants", [])]
        m = cls(params=p, **_filter_kwargs(cls, d))
        m.variants = vs
        return m


# --------------------------------------------------------------------------- protections


@dataclass
class Protection:
    """Защищённый временной участок (защита настроек)."""

    id: str
    start: float
    end: float | None = None        # None — до конца проекта
    track_id: str | None = None     # None — все дорожки
    note: str = ""

    def covers(self, t0: float, t1: float) -> bool:
        e = math.inf if self.end is None else self.end
        return t0 < e and t1 > self.start

    @classmethod
    def from_dict(cls, d: dict) -> "Protection":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class FrozenRegion:
    """Участок, зафиксированный в отрендеренный PCM (защита сэмплов)."""

    id: str
    start: float
    end: float
    file: str                 # путь относительно папки данных проекта
    sha256: str
    sample_rate: int
    margin: float = 0.02
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenRegion":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class VideoRef:
    path: str
    offset: float = 0.0             # время шкалы, где начинается видео
    rel_path: str | None = None
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "VideoRef":
        return cls(**_filter_kwargs(cls, d))


# --------------------------------------------------------------------------- project


@dataclass
class Project:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Новый проект"
    sample_rate: int = 48000
    revision: int = 0
    sources: list[Source] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    space_transitions: list[SpaceTransition] = field(default_factory=list)
    music_transitions: list[MusicTransition] = field(default_factory=list)
    protections: list[Protection] = field(default_factory=list)
    frozen: list[FrozenRegion] = field(default_factory=list)
    constraints: dict[str, bool] = field(default_factory=lambda: {"keep_tempo": False, "keep_pitch": False})
    video: VideoRef | None = None

    # ---- поиск
    def source(self, sid: str) -> Source | None:
        return next((s for s in self.sources if s.id == sid), None)

    def track(self, tid: str) -> Track | None:
        return next((t for t in self.tracks if t.id == tid), None)

    def clip(self, cid: str) -> Clip | None:
        for t in self.tracks:
            for c in t.clips:
                if c.id == cid:
                    return c
        return None

    def clip_track(self, cid: str) -> Track | None:
        for t in self.tracks:
            for c in t.clips:
                if c.id == cid:
                    return t
        return None

    def all_clips(self) -> list[Clip]:
        return [c for t in self.tracks for c in t.clips]

    def space_transition(self, xid: str) -> SpaceTransition | None:
        return next((x for x in self.space_transitions if x.id == xid), None)

    def music_transition(self, xid: str) -> MusicTransition | None:
        return next((x for x in self.music_transitions if x.id == xid), None)

    def any_transition(self, xid: str):
        return self.space_transition(xid) or self.music_transition(xid)

    def transitions_on_track(self, tid: str) -> list[SpaceTransition]:
        return sorted((x for x in self.space_transitions if x.track_id == tid), key=lambda x: x.start)

    def music_for_clip(self, cid: str) -> list[MusicTransition]:
        return [m for m in self.music_transitions if cid in (m.clip_a_id, m.clip_b_id)]

    def content_end(self) -> float:
        end = 0.0
        for c in self.all_clips():
            end = max(end, c.end)
        for x in self.space_transitions:
            end = max(end, x.end)
        return end

    def protection(self, pid: str) -> Protection | None:
        return next((p for p in self.protections if p.id == pid), None)

    # ---- сериализация
    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "id": self.id,
            "name": self.name,
            "sample_rate": self.sample_rate,
            "revision": self.revision,
            "sources": [s.to_dict() for s in self.sources],
            "tracks": [asdict(t) for t in self.tracks],
            "space_transitions": [asdict(x) for x in self.space_transitions],
            "music_transitions": [asdict(m) for m in self.music_transitions],
            "protections": [asdict(p) for p in self.protections],
            "frozen": [asdict(f) for f in self.frozen],
            "constraints": dict(self.constraints),
            "video": asdict(self.video) if self.video else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        from .migrate import migrate

        d = migrate(copy.deepcopy(d))
        p = cls(
            id=d.get("id") or uuid.uuid4().hex,
            name=d.get("name", "Проект"),
            sample_rate=int(d.get("sample_rate", 48000)),
            revision=int(d.get("revision", 0)),
        )
        p.sources = [Source.from_dict(s) for s in d.get("sources", [])]
        p.tracks = [Track.from_dict(t) for t in d.get("tracks", [])]
        p.space_transitions = [SpaceTransition.from_dict(x) for x in d.get("space_transitions", [])]
        p.music_transitions = [MusicTransition.from_dict(x) for x in d.get("music_transitions", [])]
        p.protections = [Protection.from_dict(x) for x in d.get("protections", [])]
        p.frozen = [FrozenRegion.from_dict(x) for x in d.get("frozen", [])]
        c = {"keep_tempo": False, "keep_pitch": False}
        c.update({k: bool(v) for k, v in d.get("constraints", {}).items()})
        p.constraints = c
        p.video = VideoRef.from_dict(d["video"]) if d.get("video") else None
        return p

    def clone(self) -> "Project":
        return Project.from_dict(self.to_dict())
