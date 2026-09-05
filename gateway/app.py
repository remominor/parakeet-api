"""Low-latency OpenAI transcription gateway for the embedded Parakeet engine."""
from __future__ import annotations
import asyncio, hmac, io, logging, os, struct, time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from .vad import StreamingVad

logging.basicConfig(level=os.getenv("PARAKEET_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("parakeet-api")
TARGET_RATE, MODEL_ID = 16000, "parakeet-tdt-0.6b-v2"
FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
def env_list(name: str) -> list[str]: return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]
@dataclass(frozen=True)
class Settings:
    keys: list[str]; aliases: set[str]; limit: int; timeout: float; cors: list[str]; log_text: bool
    ws_vad_threshold: float; ws_min_silence_ms: int; ws_speech_pad_ms: int; ws_max_utterance_ms: int; ws_max_frame_bytes: int
    @classmethod
    def load(cls):
        return cls(
            env_list("PARAKEET_API_KEYS"), {MODEL_ID, "parakeet", "parakeet-en", "whisper-1", *env_list("PARAKEET_MODEL_ALIASES")},
            int(os.getenv("PARAKEET_MAX_UPLOAD_MB", "64"))*1048576, float(os.getenv("PARAKEET_UPSTREAM_TIMEOUT", "300")),
            env_list("PARAKEET_CORS_ORIGINS"), os.getenv("PARAKEET_LOG_TRANSCRIPTS", "false").lower() in {"1","true","yes","on"},
            float(os.getenv("PARAKEET_WS_VAD_THRESHOLD", "0.5")), int(os.getenv("PARAKEET_WS_MIN_SILENCE_MS", "350")),
            int(os.getenv("PARAKEET_WS_SPEECH_PAD_MS", "120")), int(os.getenv("PARAKEET_WS_MAX_UTTERANCE_MS", "8000")),
            int(os.getenv("PARAKEET_WS_MAX_FRAME_BYTES", "65536")),
        )
SETTINGS = Settings.load()
@dataclass
class Stats:
    requests_total: int = 0; requests_failed: int = 0; transcoded_total: int = 0; passthrough_total: int = 0; audio_seconds_total: float = 0.; latency_ms: list[float] = field(default_factory=list)
    def observe(self, value: float): self.latency_ms.append(value); del self.latency_ms[:-1000]
STATS, INFERENCE = Stats(), asyncio.Semaphore(1)
def wav_bytes(pcm: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 36+len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16,1,1,TARGET_RATE,TARGET_RATE*2,2,16) + b"data" + struct.pack("<I",len(pcm)) + pcm
def is_wav(data: bytes) -> bool: return len(data)>=12 and data[:4]==b"RIFF" and data[8:12]==b"WAVE"
def decode_audio(data: bytes) -> bytes:
    import av
    from av.audio.resampler import AudioResampler
    chunks=[]
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio: raise ValueError("file contains no audio stream")
        stream=container.streams.audio[0]; stream.thread_type="AUTO"; resampler=AudioResampler(format="s16",layout="mono",rate=TARGET_RATE)
        for frame in container.decode(stream):
            for output in resampler.resample(frame): chunks.append(bytes(memoryview(output.planes[0])[:output.samples*2]))
        for output in resampler.resample(None): chunks.append(bytes(memoryview(output.planes[0])[:output.samples*2]))
    pcm=b"".join(chunks)
    if not pcm: raise ValueError("audio stream decoded to zero samples")
    return wav_bytes(pcm)
def credentials_valid(authorization: str|None, api_key: str|None) -> bool:
    if not SETTINGS.keys: return True
    presented=authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else api_key
    return bool(presented) and any(hmac.compare_digest(presented,key) for key in SETTINGS.keys)
def auth(authorization: str|None, api_key: str|None):
    if not SETTINGS.keys: return
    if not credentials_valid(authorization, api_key):
        raise HTTPException(401, "missing bearer token" if not authorization and not api_key else "invalid API key")
def valid_model(model: str|None):
    if model and model.strip() not in SETTINGS.aliases: raise HTTPException(404,f"unknown model {model!r}; available: {MODEL_ID}")
def stamp(seconds: float, comma: bool) -> str:
    ms=int(round(seconds*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000); return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"
def subtitle(result: dict[str,Any], kind: str) -> str:
    words=result.get("words") or [{"word":result.get("text",""),"start":0,"end":result.get("duration",0)}]; out=["WEBVTT\n"] if kind=="vtt" else []
    for n,start in enumerate(range(0,len(words),8),1):
        group=words[start:start+8]
        if kind=="srt": out.append(str(n))
        out += [f"{stamp(float(group[0].get('start',0)),kind=='srt')} --> {stamp(float(group[-1].get('end',0)),kind=='srt')}"," ".join(str(w.get("word","")).strip() for w in group).strip(),""]
    return "\n".join(out)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client=httpx.AsyncClient(timeout=httpx.Timeout(SETTINGS.timeout,connect=2),limits=httpx.Limits(max_keepalive_connections=0,max_connections=1),headers={"Connection":"close"}); app.state.ready=False; asyncio.create_task(warm(app))
    try: yield
    finally: await app.state.client.aclose()
async def warm(app: FastAPI):
    silence=wav_bytes(b"\0\0"*(TARGET_RATE//2))
    for _ in range(60):
        try:
            response=await app.state.client.post("http://127.0.0.1:8081/v1/audio/transcriptions",files={"file":("warmup.wav",silence,"audio/wav")},data={"response_format":"json"},timeout=120)
            if response.status_code==200: app.state.ready=True; LOG.info("model warm and ready"); return
        except httpx.HTTPError: pass
        await asyncio.sleep(2)
    LOG.error("model did not become ready")
async def engine_transcribe(client: httpx.AsyncClient, payload: bytes, form: dict[str, Any]) -> tuple[dict[str, Any], float]:
    try:
        async with INFERENCE:
            started=time.perf_counter()
            response=await client.post("http://127.0.0.1:8081/v1/audio/transcriptions",files={"file":("audio.wav",payload,"audio/wav")},data=form)
            engine_ms=(time.perf_counter()-started)*1000
    except httpx.HTTPError as exc:
        raise HTTPException(502,f"speech engine unavailable: {exc.__class__.__name__}") from exc
    if response.status_code!=200:
        raise HTTPException(response.status_code,response.text[:400] or "speech engine failed")
    return response.json(), engine_ms
app=FastAPI(title="Parakeet OpenAI API",version="1.0.0",lifespan=lifespan)
if SETTINGS.cors: app.add_middleware(CORSMiddleware,allow_origins=SETTINGS.cors,allow_credentials=False,allow_methods=["GET","POST","OPTIONS"],allow_headers=["Authorization","Content-Type","X-API-Key"])
@app.exception_handler(HTTPException)
async def errors(_:Request, exc:HTTPException): return JSONResponse(status_code=exc.status_code,content={"error":{"message":str(exc.detail),"type":"invalid_request_error" if exc.status_code<500 else "server_error","code":exc.status_code}})
@app.get("/health")
async def health(): return {"status":"ok"}
@app.get("/readyz")
async def readyz(request:Request): return JSONResponse(status_code=200 if request.app.state.ready else 503,content={"ready":bool(request.app.state.ready),"model":MODEL_ID})
@app.get("/v1/models")
async def models(authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
    auth(authorization,x_api_key); return {"object":"list","data":[{"id":MODEL_ID,"object":"model","created":0,"owned_by":"parakeet.cpp"}]}
@app.get("/stats")
async def stats(authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
    auth(authorization,x_api_key); values=sorted(STATS.latency_ms)
    def pct(q): return values[min(len(values)-1,int(q*(len(values)-1)))] if values else 0.
    return {"requests_total":STATS.requests_total,"requests_failed":STATS.requests_failed,"passthrough_total":STATS.passthrough_total,"transcoded_total":STATS.transcoded_total,"audio_seconds_total":round(STATS.audio_seconds_total,2),"latency_ms":{"samples":len(values),"p50":round(pct(.5),1),"p95":round(pct(.95),1),"p99":round(pct(.99),1)}}
@app.post("/v1/audio/translations")
async def translations(): raise HTTPException(501,"translation is not supported; use /v1/audio/transcriptions")
@app.websocket("/v1/audio/transcriptions/ws")
async def stream_transcriptions(websocket: WebSocket):
    if not credentials_valid(websocket.headers.get("authorization"), websocket.query_params.get("api_key")):
        await websocket.close(code=1008, reason="invalid API key")
        return
    await websocket.accept()
    try:
        detector=await asyncio.to_thread(
            StreamingVad, Path(__file__).with_name("silero_vad.onnx"), SETTINGS.ws_vad_threshold,
            SETTINGS.ws_min_silence_ms, SETTINGS.ws_speech_pad_ms, SETTINGS.ws_max_utterance_ms,
        )
    except Exception as exc:
        LOG.exception("could not initialize WebSocket VAD")
        await websocket.send_json({"type":"error","code":"vad_unavailable","message":str(exc)})
        await websocket.close(code=1011)
        return
    await websocket.send_json({"type":"ready","sample_rate":TARGET_RATE,"model":MODEL_ID})
    turn_id=0
    try:
        while True:
            message=await websocket.receive()
            if message["type"]=="websocket.disconnect": return
            data=message.get("bytes")
            if not data or message.get("text") is not None or len(data)%2 or len(data)>SETTINGS.ws_max_frame_bytes:
                await websocket.send_json({"type":"error","code":"invalid_audio_frame","message":"send non-empty PCM16 mono binary frames no larger than the configured limit"})
                await websocket.close(code=1003)
                return
            events=await asyncio.to_thread(detector.feed, data)
            for event in events:
                if event.kind=="speech_started":
                    await websocket.send_json({"type":"speech_started","turn_id":str(turn_id+1)})
                    continue
                if event.audio is None: continue
                turn_id+=1
                started=time.perf_counter()
                STATS.requests_total+=1
                try:
                    result,engine_ms=await engine_transcribe(websocket.app.state.client, wav_bytes(event.audio), {"response_format":"json"})
                except HTTPException as exc:
                    STATS.requests_failed+=1
                    await websocket.send_json({"type":"error","code":"transcription_failed","message":str(exc.detail),"turn_id":str(turn_id)})
                    continue
                total_ms=(time.perf_counter()-started)*1000
                STATS.observe(total_ms); STATS.audio_seconds_total+=float(result.get("duration") or 0)
                await websocket.send_json({"type":"final","turn_id":str(turn_id),"text":result.get("text", ""),"duration_ms":round(len(event.audio)/32,1),"engine_ms":round(engine_ms,1),"total_ms":round(total_ms,1)})
    except WebSocketDisconnect:
        return
@app.post("/v1/audio/transcriptions")
async def transcribe(request:Request,file:UploadFile=File(...),model:str|None=Form(None),response_format:str=Form("json"),language:str|None=Form(None),prompt:str|None=Form(None),temperature:float|None=Form(None),timestamp_granularities:list[str]|None=Form(None,alias="timestamp_granularities[]"),authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
    auth(authorization,x_api_key); valid_model(model); output=(response_format or "json").lower()
    if output not in FORMATS: raise HTTPException(400,f"response_format must be one of {', '.join(sorted(FORMATS))}")
    data=await file.read()
    if not data: raise HTTPException(400,"empty upload")
    if len(data)>SETTINGS.limit: raise HTTPException(413,f"file exceeds {SETTINGS.limit//1048576} MB limit")
    STATS.requests_total+=1; started=time.perf_counter()
    try: payload,converted=(data,False) if is_wav(data) else (await asyncio.to_thread(decode_audio,data),True)
    except Exception as exc: STATS.requests_failed+=1; raise HTTPException(400,f"could not decode audio: {exc}") from exc
    STATS.transcoded_total+=int(converted); STATS.passthrough_total+=int(not converted); engine_format="verbose_json" if output in {"srt","vtt"} else output; form={"response_format":engine_format}; granularity=list(timestamp_granularities or [])
    if output in {"srt","vtt"} and "word" not in granularity: granularity.append("word")
    if granularity: form["timestamp_granularities[]"]=granularity
    for key,value in (("language",language),("prompt",prompt),("temperature",temperature)):
        if value is not None: form[key]=str(value)
    try: result,engine_ms=await engine_transcribe(request.app.state.client,payload,form)
    except HTTPException: STATS.requests_failed+=1; raise
    total_ms=(time.perf_counter()-started)*1000; STATS.observe(total_ms); headers={"X-Parakeet-Model":MODEL_ID,"X-Parakeet-Engine-Ms":f"{engine_ms:.1f}","X-Parakeet-Total-Ms":f"{total_ms:.1f}","X-Parakeet-Transcoded":"1" if converted else "0"}
    if output=="text": return PlainTextResponse(result.get("text", ""),headers=headers)
    STATS.audio_seconds_total+=float(result.get("duration") or 0); LOG.info("total=%.1fms engine=%.1fms %s",total_ms,engine_ms,repr(result.get("text","")) if SETTINGS.log_text else f"chars={len(result.get('text',''))}")
    if output in {"srt","vtt"}: return PlainTextResponse(subtitle(result,output),media_type="application/x-subrip" if output=="srt" else "text/vtt",headers=headers)
    return JSONResponse(result,headers=headers)
