"""Extração de trechos (clips) de vídeo em MP4/H.264 via PyAV.

Decodifica e recodifica em H.264 (libx264) — precisão de frame na entrada
do trecho, e um MP4 que qualquer navegador reproduz (ao contrário de
simplesmente recortar o container original, que só pode cortar em keyframes
e frequentemente ficaria muito antes do ponto pedido).
"""

import io

import av

ClipError = (FileNotFoundError, ValueError, av.error.FFmpegError)


def probe_fps(video_path: str) -> float:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return float(rate) if rate else 25.0


def extract_clip_h264(video_path: str, start_sec: float, duration_sec: float) -> bytes:
    """MP4 com `duration_sec` segundos a partir de `start_sec`; b"" se não
    houver frames nesse intervalo."""
    end_sec = start_sec + duration_sec

    in_container = av.open(video_path)
    try:
        in_stream = in_container.streams.video[0]
        fps = in_stream.average_rate or in_stream.guessed_rate
        if not fps:
            raise ValueError("Não foi possível determinar o fps do vídeo")

        buf = io.BytesIO()
        out_container = av.open(buf, mode="w", format="mp4")
        out_stream = out_container.add_stream("libx264", rate=fps)
        out_stream.width = in_stream.codec_context.width
        out_stream.height = in_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"

        # Seek para o keyframe mais próximo antes de start_sec; a decodificação
        # segue até o timestamp exato, garantindo precisão de frame na saída.
        in_container.seek(int(start_sec / in_stream.time_base), stream=in_stream, any_frame=False, backward=True)

        wrote_any = False
        for frame in in_container.decode(in_stream):
            frame_time = frame.time
            if frame_time is None:
                continue
            if frame_time < start_sec:
                continue
            if frame_time > end_sec:
                break
            frame = frame.reformat(format="yuv420p")
            for packet in out_stream.encode(frame):
                out_container.mux(packet)
            wrote_any = True

        for packet in out_stream.encode():  # flush
            out_container.mux(packet)

        out_container.close()
        return buf.getvalue() if wrote_any else b""
    finally:
        in_container.close()
