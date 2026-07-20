"""Structured media probing (Phase 1).

Primary tool: PyAV (`av`), which exposes libav container/stream metadata as
structured Python objects — no parsing of human console text. ffprobe is NOT
bundled by imageio-ffmpeg (it ships only ffmpeg), so PyAV is the reliable
in-environment structured probe; if a system ffprobe exists it could be added
later as a cross-check, but Phase 1 does not depend on it.

Every field that cannot be determined is left None with an entry in
`unavailable_reasons` — values are never guessed. Stream selection rule: the
FIRST audio stream in index order (recorded on the contract so the choice is
auditable).
"""
from __future__ import annotations

from pathlib import Path

from pipeline.contracts import MediaMetadata


def probe(path: Path) -> MediaMetadata:
    meta = MediaMetadata()
    try:
        import av
        meta.probe_tool_version = getattr(av, "__version__", None)
    except ImportError as e:
        meta.warnings.append(f"pyav unavailable: {e}")
        for f in ("container_format", "audio_codec", "duration_sec",
                  "sample_rate", "channels"):
            meta.unavailable_reasons[f] = "probe tool unavailable"
        return meta

    try:
        with av.open(str(path), metadata_errors="ignore") as container:
            meta.container_format = container.format.name
            meta.tags = {str(k): str(v) for k, v in (container.metadata or {}).items()
                         if _provenance_tag(k)}
            if container.duration is not None:
                meta.duration_sec = round(container.duration / av.time_base, 6)
            else:
                meta.unavailable_reasons["duration_sec"] = "container reports no duration"
            if container.bit_rate:
                meta.bit_rate = int(container.bit_rate)
            else:
                meta.unavailable_reasons["bit_rate"] = "container reports no bit rate"

            audio_streams = [s for s in container.streams if s.type == "audio"]
            meta.n_audio_streams = len(audio_streams)
            meta.has_video = any(s.type == "video" for s in container.streams)

            if not audio_streams:
                meta.warnings.append("no audio streams found")
                meta.unavailable_reasons["selected_stream_index"] = "no audio streams"
                return meta

            s = audio_streams[0]                      # documented selection rule
            meta.selected_stream_index = s.index
            cc = s.codec_context
            meta.audio_codec = getattr(cc, "name", None)
            meta.codec_profile = getattr(cc, "profile", None) or None
            if meta.codec_profile is None:
                meta.unavailable_reasons["codec_profile"] = "codec reports no profile"
            # sample_rate/channels live on the audio-specific CodecContext
            # subclass; getattr keeps the static checker honest about the
            # generic base type PyAV declares.
            _sr = getattr(cc, "sample_rate", None)
            _ch = getattr(cc, "channels", None)
            meta.sample_rate = int(_sr) if _sr else None
            meta.channels = int(_ch) if _ch else None
            layout = getattr(cc, "layout", None)
            meta.channel_layout = getattr(layout, "name", None)
            if meta.channel_layout is None:
                meta.unavailable_reasons["channel_layout"] = "no layout reported"
            fmt = getattr(cc, "format", None)
            meta.sample_format = getattr(fmt, "name", None)
            if s.start_time is not None and s.time_base is not None:
                meta.start_time_sec = float(s.start_time * s.time_base)
                meta.time_base = str(s.time_base)
            else:
                meta.unavailable_reasons["start_time_sec"] = "stream reports no start time"
            disp = getattr(s, "disposition", None)
            meta.disposition = str(disp) if disp else None
            if meta.duration_sec is None and s.duration and s.time_base:
                meta.duration_sec = round(float(s.duration * s.time_base), 6)
                meta.unavailable_reasons.pop("duration_sec", None)
            if meta.duration_sec == 0:
                meta.warnings.append("zero-duration audio stream")
    except av.error.InvalidDataError as e:  # malformed input
        meta.warnings.append(f"malformed media: {e}")
        meta.unavailable_reasons["container_format"] = "file could not be parsed"
    except (OSError, av.AVError) as e:  # type: ignore[attr-defined]
        meta.warnings.append(f"probe failed: {e}")
        meta.unavailable_reasons["container_format"] = f"probe error: {e}"
    return meta


_PROVENANCE_KEYS = {"encoder", "creation_time", "handler_name", "title", "artist",
                    "date", "comment", "software", "device", "location"}


def _provenance_tag(key: str) -> bool:
    return str(key).lower() in _PROVENANCE_KEYS
