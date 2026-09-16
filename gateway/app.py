"""Low-latency OpenAI transcription gateway for the embedded Parakeet engine."""
from __future__ import annotations
import asyncio, base64, binascii, hmac, io, ipaddress, json, logging, os, re, socket, struct, time, uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from .vad import SAMPLE_RATE, StreamingVad, VadEvent

logging.basicConfig(level=os.getenv("PARAKEET_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG=logging.getLogger("parakeet-api"); TARGET_RATE=16000; MODEL_ID="parakeet-tdt-0.6b-v2"; VERSION="1.2.0"; FORMATS={"json","text","verbose_json","srt","vtt"}; RID_RE=re.compile(r"^[A-Za-z0-9._-]{1,128}$"); SENTENCE_RE=re.compile(r"[.?!][\"')\]]*$")
def env_list(name:str)->list[str]: return [x.strip() for x in os.getenv(name,"").split(",") if x.strip()]
def enabled(name:str, default="false")->bool: return os.getenv(name,default).lower() in {"1","true","yes","on"}
@dataclass(frozen=True)
class Settings:
 keys:list[str]; aliases:set[str]; limit:int; timeout:float; cors:list[str]; log_text:bool; ws_vad_threshold:float; ws_min_silence_ms:int; ws_speech_pad_ms:int; ws_max_utterance_ms:int; ws_max_frame_bytes:int; low_confidence:float; segment_max_duration_ms:int; segment_max_chars:int; segment_pause_ms:int; url_connect_timeout:float; url_total_timeout:float; url_allowed_hosts:set[str]; url_allow_private:bool; metrics_enabled:bool; webui_enabled:bool
 @classmethod
 def load(cls): return cls(env_list("PARAKEET_API_KEYS"),{MODEL_ID,"parakeet","parakeet-en","whisper-1",*env_list("PARAKEET_MODEL_ALIASES")},int(os.getenv("PARAKEET_MAX_UPLOAD_MB","64"))*1048576,float(os.getenv("PARAKEET_UPSTREAM_TIMEOUT","300")),env_list("PARAKEET_CORS_ORIGINS"),enabled("PARAKEET_LOG_TRANSCRIPTS"),float(os.getenv("PARAKEET_WS_VAD_THRESHOLD","0.5")),int(os.getenv("PARAKEET_WS_MIN_SILENCE_MS","350")),int(os.getenv("PARAKEET_WS_SPEECH_PAD_MS","120")),int(os.getenv("PARAKEET_WS_MAX_UTTERANCE_MS","8000")),int(os.getenv("PARAKEET_WS_MAX_FRAME_BYTES","65536")),float(os.getenv("PARAKEET_LOW_CONFIDENCE_THRESHOLD","0.70")),int(os.getenv("PARAKEET_SEGMENT_MAX_DURATION_MS","6000")),int(os.getenv("PARAKEET_SEGMENT_MAX_CHARS","100")),int(os.getenv("PARAKEET_SEGMENT_PAUSE_MS","700")),float(os.getenv("PARAKEET_URL_CONNECT_TIMEOUT","5")),float(os.getenv("PARAKEET_URL_TOTAL_TIMEOUT","30")),{x.lower() for x in env_list("PARAKEET_URL_ALLOWED_HOSTS")},enabled("PARAKEET_URL_ALLOW_PRIVATE"),enabled("PARAKEET_METRICS_ENABLED"),enabled("PARAKEET_WEBUI_ENABLED"))
SETTINGS=Settings.load()
@dataclass
class Stats:
 requests_total:int=0; requests_failed:int=0; requests_active:int=0; requests_queued:int=0; transcoded_total:int=0; passthrough_total:int=0; audio_seconds_total:float=0.; total_ms_sum:float=0.; engine_ms_sum:float=0.; websocket_connections_total:int=0; websocket_connections_active:int=0; websocket_turns_total:int=0
 request_ms:deque[float]=field(default_factory=lambda:deque(maxlen=1000)); ws_ms:deque[float]=field(default_factory=lambda:deque(maxlen=1000)); engine_ms:deque[float]=field(default_factory=lambda:deque(maxlen=1000)); decode_ms:deque[float]=field(default_factory=lambda:deque(maxlen=1000))
STATS=Stats(); INFERENCE=asyncio.Semaphore(1)
def wav_bytes(pcm:bytes)->bytes: return b"RIFF"+struct.pack("<I",36+len(pcm))+b"WAVEfmt "+struct.pack("<IHHIIHH",16,1,1,TARGET_RATE,TARGET_RATE*2,2,16)+b"data"+struct.pack("<I",len(pcm))+pcm
def is_wav(data:bytes)->bool: return len(data)>=12 and data[:4]==b"RIFF" and data[8:12]==b"WAVE"
def decode_audio(data:bytes)->bytes:
 import av
 from av.audio.resampler import AudioResampler
 chunks=[]
 with av.open(io.BytesIO(data)) as container:
  if not container.streams.audio: raise ValueError("file contains no audio stream")
  stream=container.streams.audio[0]; stream.thread_type="AUTO"; resampler=AudioResampler(format="s16",layout="mono",rate=TARGET_RATE)
  for frame in container.decode(stream):
   for out in resampler.resample(frame): chunks.append(bytes(memoryview(out.planes[0])[:out.samples*2]))
  for out in resampler.resample(None): chunks.append(bytes(memoryview(out.planes[0])[:out.samples*2]))
 pcm=b"".join(chunks)
 if not pcm: raise ValueError("audio stream decoded to zero samples")
 return wav_bytes(pcm)
def credentials_valid(authz:str|None,key:str|None)->bool:
 if not SETTINGS.keys:return True
 value=authz[7:].strip() if authz and authz.lower().startswith("bearer ") else key
 return bool(value) and any(hmac.compare_digest(value,x) for x in SETTINGS.keys)
def auth(authz:str|None,key:str|None):
 if not credentials_valid(authz,key):raise HTTPException(401,"missing bearer token" if not authz and not key else "invalid API key")
def valid_model(model:str|None):
 if model and model.strip() not in SETTINGS.aliases:raise HTTPException(404,f"unknown model {model!r}; available: {MODEL_ID}")
def validate_options(language:str|None,prompt:str|None,temperature:float|None):
 if language and language.strip().lower() not in {"en","eng","en-us"}:raise HTTPException(422,"Parakeet TDT 0.6B v2 supports English transcription only.")
 if prompt and prompt.strip():raise HTTPException(422,"prompt is not supported by Parakeet TDT 0.6B v2.")
 if temperature is not None and temperature!=0:raise HTTPException(422,"non-zero temperature is not supported; Parakeet uses deterministic greedy decoding.")
def request_id(value:str|None)->str:return value if value and RID_RE.fullmatch(value) else str(uuid.uuid4())
def confidence(result:dict[str,Any])->dict[str,float|int]|None:
 values=[float(w.get("conf",w.get("confidence"))) for w in result.get("words",[]) if w.get("conf",w.get("confidence")) is not None]
 return {"mean":round(sum(values)/len(values),3),"min":round(min(values),3),"low_word_count":sum(v<SETTINGS.low_confidence for v in values)} if values else None
def make_segment(i:int,words:list[dict[str,Any]])->dict[str,Any]:return {"id":i,"start":float(words[0].get("start",0)),"end":float(words[-1].get("end",0)),"text":" ".join(str(x.get("word","")).strip() for x in words).strip()}
def segments(result:dict[str,Any])->list[dict[str,Any]]:
 words=result.get("words") or []
 if not words:return result.get("segments") or []
 out=[]; current=[]
 for word in words:
  if current:
   prev=current[-1]; text=" ".join(str(x.get("word","")).strip() for x in current+[word]).strip(); duration=(float(word.get("end",0))-float(current[0].get("start",0)))*1000; pause=(float(word.get("start",0))-float(prev.get("end",0)))*1000
   if pause>=SETTINGS.segment_pause_ms or duration>SETTINGS.segment_max_duration_ms or len(text)>SETTINGS.segment_max_chars:out.append(make_segment(len(out),current));current=[]
  current.append(word)
  if SENTENCE_RE.search(str(word.get("word","")).strip()):out.append(make_segment(len(out),current));current=[]
 if current:out.append(make_segment(len(out),current))
 return out
def enrich(result:dict[str,Any])->dict[str,Any]:
 if result.get("words"):result["segments"]=segments(result)
 return result
def stamp(seconds:float,comma:bool)->str:
 ms=int(round(seconds*1000));h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000);return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"
def subtitle(result:dict[str,Any],kind:str)->str:
 cues=segments(result) or [{"start":0,"end":result.get("duration",0),"text":result.get("text","")}];out=["WEBVTT\n"] if kind=="vtt" else []
 for n,cue in enumerate(cues,1):
  if kind=="srt":out.append(str(n))
  out += [f"{stamp(float(cue['start']),kind=='srt')} --> {stamp(float(cue['end']),kind=='srt')}",str(cue["text"]),""]
 return "\n".join(out)
async def read_upload(file:UploadFile)->bytes:
 chunks=[];size=0
 while chunk:=await file.read(1024*1024):
  size+=len(chunk)
  if size>SETTINGS.limit:raise HTTPException(413,f"file exceeds {SETTINGS.limit//1048576} MB limit")
  chunks.append(chunk)
 return b"".join(chunks)
def validate_url(value:str)->None:
 try:parsed=urlparse(value);host=(parsed.hostname or "").lower()
 except ValueError as exc:raise HTTPException(422,"malformed audio_url") from exc
 if parsed.scheme not in {"http","https"} or not host or parsed.username or parsed.password:raise HTTPException(422,"audio_url must be a valid http or https URL without credentials")
 try:addresses={x[4][0] for x in socket.getaddrinfo(host,parsed.port or (443 if parsed.scheme=="https" else 80),type=socket.SOCK_STREAM)}
 except socket.gaierror as exc:raise HTTPException(422,"audio_url host could not be resolved") from exc
 for address in addresses:
  ip=ipaddress.ip_address(address)
  if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified) and not (SETTINGS.url_allow_private or host in SETTINGS.url_allowed_hosts):raise HTTPException(422,"audio_url resolves to a private or local address")
async def download_audio(url:str)->bytes:
 validate_url(url);timeout=httpx.Timeout(SETTINGS.url_total_timeout,connect=SETTINGS.url_connect_timeout)
 try:
  async with httpx.AsyncClient(timeout=timeout,follow_redirects=False) as client:
   async with client.stream("GET",url,headers={"Accept":"audio/*,application/octet-stream"}) as response:
    if 300<=response.status_code<400:raise HTTPException(422,"audio_url redirects are not allowed")
    if response.status_code!=200:raise HTTPException(422,f"audio_url returned HTTP {response.status_code}")
    length=response.headers.get("content-length")
    if length and int(length)>SETTINGS.limit:raise HTTPException(413,f"audio_url exceeds {SETTINGS.limit//1048576} MB limit")
    chunks=[];size=0
    async for chunk in response.aiter_bytes(1024*1024):
     size+=len(chunk)
     if size>SETTINGS.limit:raise HTTPException(413,f"audio_url exceeds {SETTINGS.limit//1048576} MB limit")
     chunks.append(chunk)
    return b"".join(chunks)
 except HTTPException:raise
 except (httpx.HTTPError,ValueError) as exc:raise HTTPException(422,f"could not download audio_url: {exc.__class__.__name__}") from exc
@asynccontextmanager
async def lifespan(app:FastAPI):
 app.state.started=time.monotonic();app.state.client=httpx.AsyncClient(timeout=httpx.Timeout(SETTINGS.timeout,connect=2),limits=httpx.Limits(max_keepalive_connections=0,max_connections=1),headers={"Connection":"close"});app.state.ready=False;asyncio.create_task(warm(app))
 try:yield
 finally:await app.state.client.aclose()
async def warm(app:FastAPI):
 silence=wav_bytes(b"\0\0"*(TARGET_RATE//2))
 for _ in range(60):
  try:
   response=await app.state.client.post("http://127.0.0.1:8081/v1/audio/transcriptions",files={"file":("warmup.wav",silence,"audio/wav")},data={"response_format":"json"},timeout=120)
   if response.status_code==200:app.state.ready=True;LOG.info("model warm and ready");return
  except httpx.HTTPError:pass
  await asyncio.sleep(2)
 LOG.error("model did not become ready")
async def engine_transcribe(client:httpx.AsyncClient,payload:bytes,form:dict[str,Any],*,parse_json=True)->tuple[dict[str,Any]|str,float]:
 STATS.requests_queued+=1
 try:
  async with INFERENCE:
   STATS.requests_queued-=1;started=time.perf_counter();response=await client.post("http://127.0.0.1:8081/v1/audio/transcriptions",files={"file":("audio.wav",payload,"audio/wav")},data=form);engine_ms=(time.perf_counter()-started)*1000
 except httpx.HTTPError as exc:raise HTTPException(502,f"speech engine unavailable: {exc.__class__.__name__}") from exc
 finally:STATS.requests_queued=max(0,STATS.requests_queued)
 if response.status_code!=200:raise HTTPException(response.status_code,response.text[:400] or "speech engine failed")
 if not parse_json:return response.text,engine_ms
 try:return response.json(),engine_ms
 except ValueError as exc:raise HTTPException(502,"speech engine returned an invalid JSON response") from exc
app=FastAPI(title="Parakeet OpenAI API",version=VERSION,lifespan=lifespan)
if SETTINGS.cors:app.add_middleware(CORSMiddleware,allow_origins=SETTINGS.cors,allow_credentials=False,allow_methods=["GET","POST","OPTIONS"],allow_headers=["Authorization","Content-Type","X-API-Key","X-Request-ID"])
@app.middleware("http")
async def correlation_id(request:Request,call_next):
 request.state.request_id=request_id(request.headers.get("x-request-id"));response=await call_next(request)
 if request.url.path=="/v1/audio/transcriptions":response.headers.setdefault("X-Request-ID",request.state.request_id)
 return response
@app.exception_handler(HTTPException)
async def errors(_:Request,exc:HTTPException):return JSONResponse(status_code=exc.status_code,content={"error":{"message":str(exc.detail),"type":"invalid_request_error" if exc.status_code<500 else "server_error","code":exc.status_code}})
@app.get("/health")
async def health():return {"status":"ok"}
@app.get("/readyz")
async def readyz(request:Request):return JSONResponse(status_code=200 if request.app.state.ready else 503,content={"ready":bool(request.app.state.ready),"model":MODEL_ID})
@app.get("/v1/models")
async def models(authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
 auth(authorization,x_api_key);return {"object":"list","data":[{"id":MODEL_ID,"object":"model","created":0,"owned_by":"parakeet.cpp"}]}
def percentiles(values:deque[float])->dict[str,float|int]:
 data=sorted(values)
 def pick(q):return data[min(len(data)-1,int(q*(len(data)-1)))] if data else 0.
 return {"samples":len(data),"p50":round(pick(.5),1),"p95":round(pick(.95),1),"p99":round(pick(.99),1)}
@app.get("/stats")
async def stats(authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
 auth(authorization,x_api_key);return {"requests_total":STATS.requests_total,"requests_failed":STATS.requests_failed,"requests_active":STATS.requests_active,"requests_queued":STATS.requests_queued,"passthrough_total":STATS.passthrough_total,"transcoded_total":STATS.transcoded_total,"audio_seconds_total":round(STATS.audio_seconds_total,2),"websocket_connections_total":STATS.websocket_connections_total,"websocket_connections_active":STATS.websocket_connections_active,"websocket_turns_total":STATS.websocket_turns_total,"http_latency_ms":percentiles(STATS.request_ms),"websocket_latency_ms":percentiles(STATS.ws_ms),"engine_ms":percentiles(STATS.engine_ms),"decode_ms":percentiles(STATS.decode_ms)}
@app.get("/info")
async def info(request:Request,authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key")):
 auth(authorization,x_api_key);return {"service":"parakeet-api","version":VERSION,"model":MODEL_ID,"engine":"parakeet.cpp","language":["en"],"uptime_seconds":round(time.monotonic()-request.app.state.started,1),"capabilities":{"word_timestamps":True,"word_confidence":True,"segments":True,"srt":True,"vtt":True,"websocket_turn_endpointing":True,"realtime_transcription":True,"partial_transcription":False,"translation":False,"prompt":False,"temperature_sampling":False},"limits":{"max_upload_mb":SETTINGS.limit//1048576,"websocket_max_frame_bytes":SETTINGS.ws_max_frame_bytes,"websocket_max_utterance_ms":SETTINGS.ws_max_utterance_ms,"websocket_completed_turn_queue":2}}
@app.get("/metrics")
async def metrics():
 if not SETTINGS.metrics_enabled:raise HTTPException(404,"metrics endpoint is disabled")
 def summary(name,values,total):
  p=percentiles(values);return [f"# TYPE {name} summary",f'{name}{{quantile="0.5"}} {p["p50"]/1000}',f'{name}{{quantile="0.95"}} {p["p95"]/1000}',f'{name}{{quantile="0.99"}} {p["p99"]/1000}',f"{name}_count {p['samples']}",f"{name}_sum {total/1000}"]
 lines=["# TYPE parakeet_requests_total counter",f"parakeet_requests_total {STATS.requests_total}",f"parakeet_requests_failed_total {STATS.requests_failed}",f"parakeet_audio_seconds_total {STATS.audio_seconds_total}",f"parakeet_transcoded_total {STATS.transcoded_total}",f"parakeet_passthrough_total {STATS.passthrough_total}",f"parakeet_ws_connections_active {STATS.websocket_connections_active}",f"parakeet_ws_turns_total {STATS.websocket_turns_total}"]+summary("parakeet_engine_duration_seconds",STATS.engine_ms,STATS.engine_ms_sum)+summary("parakeet_request_duration_seconds",STATS.request_ms,STATS.total_ms_sum)
 return Response("\n".join(lines)+"\n",media_type="text/plain; version=0.0.4")
@app.post("/v1/audio/translations")
async def translations():raise HTTPException(501,"translation is not supported; use /v1/audio/transcriptions")
def vad_values(value:dict[str,Any])->dict[str,Any]:
 allowed={"threshold","min_silence_ms","speech_pad_ms","max_utterance_ms"}
 if not isinstance(value,dict) or set(value)-allowed:raise ValueError("unsupported VAD configuration field")
 out={}
 for key,item in value.items():
  if key=="threshold" and isinstance(item,(int,float)) and .05<=float(item)<=.95:out[key]=float(item)
  elif key=="min_silence_ms" and isinstance(item,int) and 50<=item<=5000:out[key]=item
  elif key=="speech_pad_ms" and isinstance(item,int) and 0<=item<=2000:out[key]=item
  elif key=="max_utterance_ms" and isinstance(item,int) and 250<=item<=60000:out[key]=item
  else:raise ValueError(f"invalid VAD {key}")
 return out
def realtime_error(code:str,message:str)->dict[str,Any]:return {"type":"error","error":{"type":"invalid_request_error","code":code,"message":message}}
def pcm24_to_16(data:bytes)->bytes:
 """Linear 24 kHz PCM16 -> 16 kHz PCM16; realtime adapter only."""
 if len(data)%2:raise ValueError("audio must be PCM16")
 import numpy as np
 source=np.frombuffer(data,dtype="<i2")
 if not len(source):return b""
 target=np.interp(np.arange((len(source)*2)//3)*1.5,np.arange(len(source)),source).round().astype("<i2")
 return target.tobytes()
@dataclass
class CompletedTurn:
 turn_id:str; audio:bytes; start_ms:float; end_ms:float; item_id:str|None=None
async def run_turn_worker(app:FastAPI,queue:asyncio.Queue[CompletedTurn],emit,session_id:str):
 while True:
  turn=await queue.get();STATS.requests_total+=1;STATS.requests_active+=1;STATS.websocket_turns_total+=1;started=time.perf_counter()
  try:result,engine_ms=await engine_transcribe(app.state.client,wav_bytes(turn.audio),{"response_format":"verbose_json","timestamp_granularities[]":["word"]})
  except HTTPException as exc:STATS.requests_failed+=1;await emit(realtime_error("transcription_failed",str(exc.detail))|{"turn_id":turn.turn_id});queue.task_done();continue
  finally:STATS.requests_active-=1
  assert isinstance(result,dict);total_ms=(time.perf_counter()-started)*1000;STATS.ws_ms.append(total_ms);STATS.engine_ms.append(engine_ms);STATS.engine_ms_sum+=engine_ms;STATS.total_ms_sum+=total_ms;STATS.audio_seconds_total+=len(turn.audio)/32000
  await emit((turn,result,engine_ms,total_ms));LOG.info("request_id=%s turn_id=%s total=%.1fms engine=%.1fms chars=%s",session_id,turn.turn_id,total_ms,engine_ms,len(result.get("text","")));queue.task_done()
async def run_vad_events(detector:StreamingVad,events:list[VadEvent],turn_state:dict[str,Any],queue:asyncio.Queue[CompletedTurn],emit,lifecycle):
 for event in events:
  if event.kind=="speech_started":
   turn_state["id"]=str(int(turn_state.get("next",0))+1);turn_state["next"]=int(turn_state["id"]);await lifecycle("speech_started",turn_state["id"],event)
  elif event.kind=="speech_stopped" and turn_state.get("id"):await lifecycle("speech_stopped",turn_state["id"],event)
  elif event.kind=="turn" and event.audio and turn_state.get("id"):
   turn=CompletedTurn(turn_state["id"],event.audio,event.audio_start_ms or 0,event.audio_end_ms or 0,"item_"+uuid.uuid4().hex);event.item_id=turn.item_id;turn_state["id"]=None
   if queue.full():await emit(realtime_error("turn_queue_full","completed-turn queue is full; turn was discarded")|{"turn_id":turn.turn_id})
   else:await queue.put(turn);await lifecycle("committed",turn.turn_id,event)
@app.websocket("/v1/audio/transcriptions/ws")
async def stream_transcriptions(websocket:WebSocket):
 if not credentials_valid(websocket.headers.get("authorization"),websocket.query_params.get("api_key")):await websocket.close(code=1008,reason="invalid API key");return
 await websocket.accept();STATS.websocket_connections_total+=1;STATS.websocket_connections_active+=1;verbose=enabled_value(websocket.query_params.get("verbose"));session_id=request_id(websocket.headers.get("x-request-id"));send_lock=asyncio.Lock()
 async def emit(value):
  async with send_lock:
   if isinstance(value,tuple):
    turn,result,engine_ms,total_ms=value;out={"type":"final","turn_id":turn.turn_id,"text":result.get("text",""),"duration_ms":round(len(turn.audio)/32,1),"engine_ms":round(engine_ms,1),"total_ms":round(total_ms,1)}
    if val:=confidence(result):out["confidence"]=val
    if verbose:out["words"]=result.get("words",[])
    await websocket.send_json(out)
   else:await websocket.send_json(value)
 async def lifecycle(kind,turn_id,event):
  if kind=="speech_started":await emit({"type":"speech_started","turn_id":turn_id,"audio_start_ms":round(event.audio_start_ms or 0,1)})
  elif kind=="speech_stopped":await emit({"type":"speech_stopped","turn_id":turn_id,"audio_end_ms":round(event.audio_end_ms or 0,1)})
 try:
  detector=await asyncio.to_thread(StreamingVad,Path(__file__).with_name("silero_vad.onnx"),SETTINGS.ws_vad_threshold,SETTINGS.ws_min_silence_ms,SETTINGS.ws_speech_pad_ms,SETTINGS.ws_max_utterance_ms);queue=asyncio.Queue(maxsize=2);state={"next":0,"id":None};worker=asyncio.create_task(run_turn_worker(websocket.app,queue,emit,session_id));await emit({"type":"ready","sample_rate":TARGET_RATE,"model":MODEL_ID,"request_id":session_id})
  while True:
   message=await websocket.receive()
   if message["type"]=="websocket.disconnect":return
   if message.get("text") is not None:
    try:control=json.loads(message["text"])
    except ValueError:await emit(realtime_error("invalid_control","control message must be JSON"));continue
    if control.get("type")=="commit":
     event=await asyncio.to_thread(detector.flush)
     if event:await run_vad_events(detector,[VadEvent("speech_stopped",audio_end_ms=event.audio_end_ms),event],state,queue,emit,lifecycle)
    elif control.get("type")=="clear":await asyncio.to_thread(detector.clear);state["id"]=None
    elif control.get("type")=="config":
     try:values=vad_values(control.get("vad"));await asyncio.to_thread(detector.configure,**values);await emit({"type":"config","vad":values})
     except ValueError as exc:await emit(realtime_error("invalid_vad_config",str(exc)))
    else:await emit(realtime_error("unsupported_control","unsupported native WebSocket control event"))
    continue
   data=message.get("bytes")
   if not data or len(data)%2 or len(data)>SETTINGS.ws_max_frame_bytes:await emit(realtime_error("invalid_audio_frame","send non-empty PCM16 mono binary frames no larger than the configured limit"));await websocket.close(code=1003);return
   await run_vad_events(detector,await asyncio.to_thread(detector.feed,data),state,queue,emit,lifecycle)
 except WebSocketDisconnect:return
 except Exception as exc:LOG.exception("WebSocket VAD failure");await emit(realtime_error("vad_unavailable",str(exc)))
 finally:
  if 'worker' in locals():worker.cancel()
  STATS.websocket_connections_active-=1
@app.websocket("/v1/realtime")
async def realtime_transcriptions(websocket:WebSocket):
 if not credentials_valid(websocket.headers.get("authorization"),websocket.query_params.get("api_key")):await websocket.close(code=1008,reason="invalid API key");return
 if websocket.query_params.get("intent","transcription")!="transcription" or (websocket.query_params.get("model") and websocket.query_params.get("model") not in SETTINGS.aliases):await websocket.close(code=1008,reason="transcription-only realtime endpoint");return
 await websocket.accept();STATS.websocket_connections_total+=1;STATS.websocket_connections_active+=1;session_id="sess_"+uuid.uuid4().hex;send_lock=asyncio.Lock()
 session={"id":session_id,"object":"realtime.transcription_session","model":MODEL_ID,"input_audio_format":"pcm16","input_audio_sample_rate_hz":24000,"turn_detection":{"threshold":SETTINGS.ws_vad_threshold,"silence_duration_ms":SETTINGS.ws_min_silence_ms,"prefix_padding_ms":SETTINGS.ws_speech_pad_ms}}
 async def emit(value):
  async with send_lock:
   if isinstance(value,tuple):
    turn,result,_,_=value;item_id=turn.item_id or "item_"+uuid.uuid4().hex
    await websocket.send_json({"type":"conversation.item.created","event_id":"event_"+uuid.uuid4().hex,"item":{"id":item_id,"type":"message","role":"user","status":"completed"}})
    await websocket.send_json({"type":"conversation.item.input_audio_transcription.completed","event_id":"event_"+uuid.uuid4().hex,"item_id":item_id,"content_index":0,"transcript":result.get("text",""),"usage":{"type":"duration","seconds":round(len(turn.audio)/32000,3)}})
   else:await websocket.send_json(value)
 async def lifecycle(kind,turn_id,event):
  if kind=="speech_started":await emit({"type":"input_audio_buffer.speech_started","event_id":"event_"+uuid.uuid4().hex,"audio_start_ms":round(event.audio_start_ms or 0,1)})
  elif kind=="speech_stopped":await emit({"type":"input_audio_buffer.speech_stopped","event_id":"event_"+uuid.uuid4().hex,"audio_end_ms":round(event.audio_end_ms or 0,1)})
  elif kind=="committed":await emit({"type":"input_audio_buffer.committed","event_id":"event_"+uuid.uuid4().hex,"item_id":getattr(event,"item_id",None),"audio_start_ms":round(event.audio_start_ms or 0,1),"audio_end_ms":round(event.audio_end_ms or 0,1)})
 try:
  detector=await asyncio.to_thread(StreamingVad,Path(__file__).with_name("silero_vad.onnx"),SETTINGS.ws_vad_threshold,SETTINGS.ws_min_silence_ms,SETTINGS.ws_speech_pad_ms,SETTINGS.ws_max_utterance_ms);queue=asyncio.Queue(maxsize=2);state={"next":0,"id":None};worker=asyncio.create_task(run_turn_worker(websocket.app,queue,emit,session_id));await emit({"type":"session.created","event_id":"event_"+uuid.uuid4().hex,"session":session})
  while True:
   message=await websocket.receive()
   if message["type"]=="websocket.disconnect":return
   if message.get("bytes") is not None:await emit(realtime_error("unsupported_event","realtime endpoint accepts JSON events only"));continue
   try:event=json.loads(message.get("text") or "")
   except ValueError:await emit(realtime_error("invalid_event","event must be JSON"));continue
   kind=event.get("type")
   if kind=="input_audio_buffer.append":
    if set(event)-{"type","audio"}:await emit(realtime_error("unsupported_event","unsupported append fields"));continue
    try:pcm=base64.b64decode(event.get("audio",""),validate=True);pcm=pcm24_to_16(pcm)
    except (ValueError,binascii.Error):await emit(realtime_error("invalid_audio","audio must be base64 PCM16 at 24 kHz"));continue
    await run_vad_events(detector,await asyncio.to_thread(detector.feed,pcm),state,queue,emit,lifecycle)
   elif kind=="input_audio_buffer.commit":
    event_out=await asyncio.to_thread(detector.flush)
    if event_out:await run_vad_events(detector,[VadEvent("speech_stopped",audio_end_ms=event_out.audio_end_ms),event_out],state,queue,emit,lifecycle)
   elif kind=="input_audio_buffer.clear":await asyncio.to_thread(detector.clear);state["id"]=None
   elif kind=="session.update":
    update=event.get("session")
    try:
     if not isinstance(update,dict) or set(update)!={"turn_detection"} or not isinstance(update["turn_detection"],dict):raise ValueError("only session.turn_detection is supported")
     turn=update["turn_detection"];mapping={"threshold":"threshold","silence_duration_ms":"min_silence_ms","prefix_padding_ms":"speech_pad_ms"}
     if set(turn)-set(mapping):raise ValueError("unsupported turn_detection field")
     values=vad_values({mapping[k]:v for k,v in turn.items()});await asyncio.to_thread(detector.configure,**values)
     for key,value in turn.items():session["turn_detection"][key]=value
     await emit({"type":"session.updated","event_id":"event_"+uuid.uuid4().hex,"session":session})
    except ValueError as exc:await emit(realtime_error("invalid_session_update",str(exc)))
   else:await emit(realtime_error("unsupported_event","unsupported realtime event"))
 except WebSocketDisconnect:return
 except Exception as exc:LOG.exception("Realtime VAD failure");await emit(realtime_error("realtime_unavailable",str(exc)))
 finally:
  if 'worker' in locals():worker.cancel()
  STATS.websocket_connections_active-=1
def enabled_value(value:str|None)->bool:return bool(value and value.lower() in {"1","true","yes","on"})
@app.post("/v1/audio/transcriptions")
async def transcribe(request:Request,file:UploadFile|None=File(None),audio_url:str|None=Form(None),model:str|None=Form(None),response_format:str=Form("json"),language:str|None=Form(None),prompt:str|None=Form(None),temperature:float|None=Form(None),timestamp_granularities:list[str]|None=Form(None,alias="timestamp_granularities[]"),authorization:str|None=Header(None),x_api_key:str|None=Header(None,alias="X-API-Key"),x_request_id:str|None=Header(None,alias="X-Request-ID")):
 auth(authorization,x_api_key);valid_model(model);validate_options(language,prompt,temperature);rid=request.state.request_id
 if bool(file and file.filename)==bool(audio_url and audio_url.strip()):raise HTTPException(400,"provide exactly one of file or audio_url")
 output=(response_format or "json").lower()
 if output not in FORMATS:raise HTTPException(400,f"response_format must be one of {', '.join(sorted(FORMATS))}")
 data=await (read_upload(file) if file and file.filename else download_audio(audio_url.strip()))
 if not data:raise HTTPException(400,"empty upload")
 STATS.requests_total+=1;STATS.requests_active+=1;started=time.perf_counter();decode_started=time.perf_counter()
 try:payload,converted=(data,False) if is_wav(data) else (await asyncio.to_thread(decode_audio,data),True)
 except Exception as exc:STATS.requests_failed+=1;STATS.requests_active-=1;raise HTTPException(400,f"could not decode audio: {exc}") from exc
 finally:STATS.decode_ms.append((time.perf_counter()-decode_started)*1000)
 STATS.transcoded_total+=int(converted);STATS.passthrough_total+=int(not converted);granularity=list(timestamp_granularities or []);needs_words=output in {"verbose_json","srt","vtt"} or "word" in granularity
 if needs_words and "word" not in granularity:granularity.append("word")
 form={"response_format":"verbose_json" if needs_words else output}
 if granularity:form["timestamp_granularities[]"]=granularity
 try:result,engine_ms=await engine_transcribe(request.app.state.client,payload,form,parse_json=output!="text")
 except HTTPException:STATS.requests_failed+=1;raise
 finally:STATS.requests_active-=1
 total_ms=(time.perf_counter()-started)*1000;STATS.request_ms.append(total_ms);STATS.engine_ms.append(engine_ms);STATS.total_ms_sum+=total_ms;STATS.engine_ms_sum+=engine_ms;headers={"X-Request-ID":rid,"X-Parakeet-Model":MODEL_ID,"X-Parakeet-Engine-Ms":f"{engine_ms:.1f}","X-Parakeet-Total-Ms":f"{total_ms:.1f}","X-Parakeet-Transcoded":"1" if converted else "0"}
 if output=="text":return PlainTextResponse(result,headers=headers)
 assert isinstance(result,dict);STATS.audio_seconds_total+=float(result.get("duration") or 0);LOG.info("request_id=%s total=%.1fms engine=%.1fms %s",rid,total_ms,engine_ms,repr(result.get("text","")) if SETTINGS.log_text else f"chars={len(result.get('text',''))}")
 if output in {"verbose_json","srt","vtt"}:enrich(result)
 if output in {"srt","vtt"}:return PlainTextResponse(subtitle(result,output),media_type="application/x-subrip" if output=="srt" else "text/vtt",headers=headers)
 return JSONResponse(result,headers=headers)
@app.get("/",response_class=HTMLResponse)
async def webui():
 if not SETTINGS.webui_enabled:raise HTTPException(404,"web UI is disabled")
 return HTMLResponse(Path(__file__).with_name("webui.html").read_text())
