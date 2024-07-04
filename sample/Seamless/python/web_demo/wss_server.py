import asyncio
import json
import websockets
import argparse
import ssl
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="0.0.0.0",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    required=False,
                    help="grpc server port")
parser.add_argument("--use_offline",
					action='store_true',
					default=False,
					help="whether to load offline model"
					)
parser.add_argument('--offline_encoder_frontend_bmodel', 
					type=str, 
					default='../../models/BM1684X/m4t_encoder_frontend_fp16_s2t.bmodel', 
					help='path of offline Wav2Vec2Frontend bmodel')
parser.add_argument('--offline_encoder_bmodel', 
					type=str, 
					default='../../models/BM1684X/m4t_encoder_fp16_s2t.bmodel', 
					help='path of offline UnitYEncoderAdaptor bmodel')
parser.add_argument('--offline_decoder_frontend_bmodel', 
					type=str, 
					default='../../models/BM1684X/m4t_decoder_frontend_beam_size_fp16_s2t.bmodel', 
					help='path of offline text decoder frontend bmodel')
parser.add_argument('--offline_decoder_bmodel', 
					type=str, 
					default='../../models/BM1684X/m4t_decoder_beam_size_fp16_s2t.bmodel', 
					help='path of offline text decoder bmodel')
parser.add_argument('--offline_decoder_final_proj_bmodel', 
					type=str, 
					default='../../models/BM1684X/m4t_decoder_final_proj_beam_size_fp16_s2t.bmodel', 
					help='path of offline final proj bmodel')
parser.add_argument('--max_output_seq_len',
					type=int,
					default=50,
					help='max offline output len')
parser.add_argument('--beam_size',
					type=int,
					default=1,
					help='beam size')
parser.add_argument("--use_online",
					action='store_true',
					default=False,
					help="whether to load online model"
					)
parser.add_argument('--online_encoder_frontend_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_encoder_frontend_fp16_s2t.bmodel', 
					help='path of online Wav2Vec2Frontend bmodel')
parser.add_argument('--online_encoder_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_encoder_fp16_s2t.bmodel', 
					help='path of online UnitYEncoderAdaptor bmodel')
parser.add_argument('--tokenizer_model', 
					type=str, 
					default='../../models/tokenizer.model', 
					help='path of tokenizer model')
parser.add_argument('--online_decoder_frontend_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_decoder_frontend_fp16_s2t.bmodel', 
					help='path of monotonic text decoder frontend bmodel')
parser.add_argument('--online_decoder_step_bigger_1_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel', 
					help='path of monotonic text decoder bmodel')
parser.add_argument('--online_decoder_step_equal_1_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_decoder_step_equal_1_fp16_s2t.bmodel', 
					help='path of monotonic text decoder step=0 bmodel')
parser.add_argument('--online_decoder_final_proj_bmodel', 
					type=str, 
					default='../../models/BM1684X/seamless_streaming_decoder_final_proj_fp16_s2t.bmodel', 
					help='path of monotonic final proj bmodel')
parser.add_argument('--chunk_duration_ms', 
					type=int, 
					default=1600, 
					help='segment length in online rec')
parser.add_argument('--consecutive_segments_num',
					type=int,
					default=2,
					help='the processed number of segments once during streaming')
parser.add_argument('--fbank_min_input_length',
					type=int,
					default=80,
					help='the min length of fbank input to encoder')
parser.add_argument('--fbank_min_starting_wait',
					type=int,
					default=48,
					help='the waitting min length of fbank input to encoder, valid when it > fbank_min_input_length')
parser.add_argument('--tgt_lang', 
					type=str, 
					default='cmn', 
					help='output langauge')
parser.add_argument('--dev_id', 
					type=int, 
					default=0, 
					help='dev id')
parser.add_argument("--punc_model",
                    type=str,
                    default="../../models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
                    help="model from modelscope")
parser.add_argument("--punc_model_revision",
                    type=str,
                    default="v2.0.4",
                    help="")
parser.add_argument("--device",
                    type=str,
                    default="cpu",
                    help="only support cpu")
parser.add_argument("--ncpu",
                    type=int,
                    default=4,
                    help="cpu cores for punc model")
parser.add_argument("--certfile",
                    type=str,
                    default="",
                    required=False,
                    help="certfile for ssl, option: server.crt")

parser.add_argument("--keyfile",
                    type=str,
                    default="",
                    required=False,
                    help="keyfile for ssl. option: server.key")
args = parser.parse_args()


websocket_users = set()

print("model loading")
from auto_model import AutoModel

# asr
if args.use_online:
	model_asr_streaming = AutoModel(rec_model = "online",
									encoder_frontend_path=args.online_encoder_frontend_bmodel,
									encoder_path=args.online_encoder_bmodel,
									decoder_frontend_path=args.online_decoder_frontend_bmodel,
									decoder_step_bigger_1_path=args.online_decoder_step_bigger_1_bmodel,
									decoder_step_equal_1_path=args.online_decoder_step_equal_1_bmodel,
									decoder_final_proj_path=args.online_decoder_final_proj_bmodel,
									tokenizer_path=args.tokenizer_model,
									chunk_duration_ms=args.chunk_duration_ms,
									consecutive_segments_num=args.consecutive_segments_num,
									fbank_min_input_length=args.fbank_min_input_length,
									fbank_min_starting_wait=args.fbank_min_starting_wait,
									dev_id=args.dev_id,
									tgt_lang=args.tgt_lang,
									)
else:
	model_asr_streaming = None

# asr
if args.use_offline:
	model_asr = AutoModel(rec_model = "offline",
						encoder_frontend_path=args.offline_encoder_frontend_bmodel,
						encoder_path=args.offline_encoder_bmodel,
						decoder_frontend_path=args.offline_decoder_frontend_bmodel,
						decoder_path=args.offline_decoder_bmodel,
						decoder_final_proj_path=args.offline_decoder_final_proj_bmodel,
						tokenizer_path=args.tokenizer_model,
						max_output_seq_len=args.max_output_seq_len,
						beam_size=args.beam_size,
						dev_id=args.dev_id,
						tgt_lang=args.tgt_lang,
						)
else:
	model_asr = None

if args.punc_model != "":
	model_punc = AutoModel(model=args.punc_model,
	                       model_revision=args.punc_model_revision,
	                       ngpu=0,
	                       ncpu=args.ncpu,
	                       device=args.device,
	                       disable_pbar=True,
                           disable_log=True,
	                       )
else:
	model_punc = None
model_vad = None



print("model loaded! only support one client at the same time now!!!!")

async def ws_reset(websocket):
	print("ws reset now, total num is ",len(websocket_users))

	websocket.status_dict_asr_online["cache"] = {}
	websocket.status_dict_asr_online["is_final"] = True
	websocket.status_dict_vad["cache"] = {}
	websocket.status_dict_vad["is_final"] = True
	websocket.status_dict_punc["cache"] = {}
	
	await websocket.close()


async def clear_websocket():
	for websocket in websocket_users:
		await ws_reset(websocket)
	websocket_users.clear()



async def ws_serve(websocket, path):
	frames = []
	frames_asr_bytes = b''
	frames_asr_online = []
	last_segment_id = 0
	global websocket_users
	# await clear_websocket()
	websocket_users.add(websocket)
	websocket.status_dict_asr = {}
	websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
	websocket.status_dict_vad = {'cache': {}, "is_final": False}
	websocket.status_dict_punc = {'cache': {}}
	websocket.chunk_interval = 10
	websocket.vad_pre_idx = 0
	speech_start = False
	speech_end_i = -1
	websocket.wav_name = "microphone"
	websocket.mode = "online"
	print("new user connected", flush=True)
	
	try:
		async for message in websocket:
			if isinstance(message, str):
				messagejson = json.loads(message)
				
				if "is_speaking" in messagejson:
					websocket.is_speaking = messagejson["is_speaking"]
					websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
				if "chunk_interval" in messagejson:
					websocket.chunk_interval = messagejson["chunk_interval"]
				if "wav_name" in messagejson:
					websocket.wav_name = messagejson.get("wav_name")
				if "chunk_size" in messagejson:
					chunk_size = messagejson["chunk_size"]
					websocket.status_dict_asr_online["chunk_size"] = chunk_size
				if "encoder_chunk_look_back" in messagejson:
					websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson["encoder_chunk_look_back"]
				if "decoder_chunk_look_back" in messagejson:
					websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson["decoder_chunk_look_back"]
				if "hotword" in messagejson:
					websocket.status_dict_asr["hotword"] = messagejson["hotword"]
				if "mode" in messagejson:
					websocket.mode = messagejson["mode"]
				if "audio_fs" in messagejson:
					websocket.status_dict_asr_online["audio_fs"] = messagejson["audio_fs"]
				if "chunk_duration_ms" in messagejson:
					websocket.status_dict_asr["chunk_duration_ms"] = messagejson["chunk_duration_ms"]
			
			websocket.status_dict_vad["chunk_size"] = int(websocket.status_dict_asr_online["chunk_size"]*60/websocket.chunk_interval)
			if len(frames_asr_online) > 0 or len(frames_asr_bytes) > 0 or not isinstance(message, str):
				if not isinstance(message, str):
					last_segment_id = int.from_bytes(message[-2:], byteorder='little', signed=False)
					message = message[:-2]
					frames.append(message)
					duration_ms = len(message)//32
					websocket.vad_pre_idx += duration_ms
					
					# asr online
					if websocket.mode == "online":
						frames_asr_online.append(message)
					websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
					if len(frames_asr_online) % websocket.chunk_interval == 0 or websocket.status_dict_asr_online["is_final"]:
						if websocket.mode == "online":
							audio_in = b"".join(frames_asr_online)
							# print(audio_in)
							try:
								await async_asr_online(websocket, audio_in, last_segment_id)
							except Exception as e:
								print(f"error in asr streaming, {websocket.status_dict_asr_online}")
								print("Exception:", e)
						frames_asr_online = []
					# NOTE: ori
					"""
					if speech_start:
						frames_asr.append(message)
						frames_asr_byte += message
					"""
					if websocket.mode == "offline":
						frames_asr_bytes += message
					# vad online
					try:
						if model_vad is not None:
							speech_start_i, speech_end_i = await async_vad(websocket, message)
						else:
							speech_end_i = -1
					except Exception as e:
						print("error in vad")
						print("Exception:", e)
					# NOTE: ori
					"""
					if speech_start_i != -1:
						speech_start = True
						beg_bias = (websocket.vad_pre_idx-speech_start_i)//duration_ms
						frames_pre = frames[-beg_bias:]
						frames_asr = []
						frames_asr.extend(frames_pre)
						frames_asr_byte = b''.join(frames_asr)
					"""
				# asr punc offline
				# print(speech_end_i)
				# print(audio_in)
				if speech_end_i != -1 or not websocket.is_speaking or len(frames_asr_bytes) >= 1:
					# print("vad end point")
					if websocket.mode == "offline":
						# NOTE: ori
						# audio_in = b"".join(frames_asr)
						audio_in = frames_asr_bytes
						try:
							await async_asr(websocket, audio_in, last_segment_id)
						except Exception as e:
							print("error in asr offline")
							print("Exception:", e)
					frames_asr_bytes = b''
					speech_start = False
					frames_asr_online = []
					websocket.status_dict_asr_online["cache"] = {}
					if not websocket.is_speaking:
						websocket.vad_pre_idx = 0
						frames = []
						websocket.status_dict_vad["cache"] = {}
					else:
						frames = frames[-20:]
			elif not websocket.is_speaking:
				mode = websocket.mode
				res_message = json.dumps({"mode": mode, "text": '', "wav_name": websocket.wav_name,"is_final":not websocket.is_speaking,"last_segment_id":last_segment_id})
				await websocket.send(res_message)
	
	
	except websockets.ConnectionClosed:
		print("ConnectionClosed...", websocket_users,flush=True)
		await ws_reset(websocket)
		websocket_users.remove(websocket)
	except websockets.InvalidState:
		print("InvalidState...")
	except Exception as e:
		print("Exception:", e)


async def async_vad(websocket, audio_in):
	
	logging.info('async_vad')
	segments_result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)[0]["value"]
	# print(segments_result)
	
	speech_start = -1
	speech_end = -1
	
	if len(segments_result) == 0 or len(segments_result) > 1:
		return speech_start, speech_end
	if segments_result[0][0] != -1:
		speech_start = segments_result[0][0]
	if segments_result[0][1] != -1:
		speech_end = segments_result[0][1]
	return speech_start, speech_end


async def async_asr(websocket, audio_in, last_segment_id):
	logging.info('asyn_asr')
	# model_asr_streaming.reset()
	if len(audio_in) > 0:
		# print(len(audio_in))
		es = websocket.status_dict_asr["chunk_duration_ms"] * (last_segment_id + 1) / 1000.
		m, s = divmod(es, 60)
		h, m = divmod(m, 60)
		ess = "%d:%02d:%02d" % (h, m, s)
		logging.info('audio segment in ' + ess)
		rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
		# print("offline_asr, ", rec_result)
		if model_punc is not None and len(rec_result["text"])>0:
			# print("offline, before punc", rec_result, "cache", websocket.status_dict_punc)
			rec_result = model_punc.generate(input=rec_result['text'], **websocket.status_dict_punc)[0]
			# print("offline, after punc", rec_result)
		if len(rec_result["text"])>0:
			# print("offline", rec_result)
			mode = websocket.mode
			# NOTE: ori
			# message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,"is_final":websocket.is_speaking})
			message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,"is_final":not websocket.is_speaking, "last_segment_id": last_segment_id})
			await websocket.send(message)


async def async_asr_online(websocket, audio_in, last_segment_id):
	logging.info('async_asr_online')
	# model_asr_streaming.reset()
	if len(audio_in) > 0:
		# print(websocket.status_dict_asr_online.get("is_final", False))
		# print('online input: ', audio_in)
		es = websocket.status_dict_asr["chunk_duration_ms"] * (last_segment_id + 1) / 1000.
		m, s = divmod(es, 60)
		h, m = divmod(m, 60)
		ess = "%d:%02d:%02d" % (h, m, s)
		logging.info('audio segment in ' + ess)
		rec_result = model_asr_streaming.generate(input=audio_in, **websocket.status_dict_asr_online)[0]
		# print("online, ", rec_result)
		if len(rec_result["text"]):
			mode = websocket.mode
			# NOTE: ori
			# message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,"is_final":websocket.is_speaking})
			message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,"is_final":not websocket.is_speaking, "last_segment_id": last_segment_id})
			await websocket.send(message)

if len(args.certfile)>0:
	ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
	
	# Generate with Lets Encrypt, copied to this location, chown to current user and 400 permissions
	ssl_cert = args.certfile
	ssl_key = args.keyfile
	
	ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
	start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None,ssl=ssl_context)
else:
	start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
