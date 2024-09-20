import json
import socket
import argparse
import subprocess
import logging
import multiprocessing
from rich.console import Console
from pipeline import Pipeline, MinicpmModel, Llama3Model, VITS, Timer
import sys
sys.path.append("../whisper-TPU_py")
import whisperWrapper
import os
import time
import threading
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def set_argparser(name:str):
    def whisper_parser():
        return whisperWrapper.set_argparser()
    
    def llama3_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm_model_path', type=str, default = '../../BM1684X/llama3/llama3-8b_int4_1dev_256.bmodel', help='path to the LLM bmodel file')
        parser.add_argument('--tokenizer_path', type=str, default="../../BM1684X/llama3/token_config", help='path to the tokenizer file')
        parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
        parser.add_argument('--temperature_llm', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')    
        return parser
    
    def minicpm_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--minicpm_model_path', type=str, default = '../../BM1688/minicpm/minicpm-2b_int4_1core.bmodel', help='path to the LLM bmodel file') 
        parser.add_argument('--minicpm_tokenizer_model_path', type=str, default = '../../BM1688/minicpm/tokenizer.model', help='path to the tokenizer file') 
        parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
        return parser

    def tts_parser():
        parser = argparse.ArgumentParser(
            description='Inference code for bert vits models')
        parser.add_argument('--vits_model', type=str, default='../../BM1688/vits/vits_chinese_128_bm1688_f16_1core.bmodel', help='path of bmodel')
        parser.add_argument('--bert_model', type=str, default='../../BM1688/vits/bert_1688_f32_1core.bmodel', help='path of bert config')
        parser.add_argument('-d', '--devid', type=int, default=0, help='device id')
        return parser
    
    parser_dict = {
        'llama3': llama3_parser,
        'minicpm': minicpm_parser,
        'whisper': whisper_parser,
        'tts': tts_parser
    }

    parser_factory = parser_dict.get(name.lower())

    if parser_factory is None:
        raise ValueError(f"Invalid model type '{name}'")
    
    return parser_factory()


parser_global = argparse.ArgumentParser()
parser_global.add_argument("--host",
                    type=str,
                    default="0.0.0.0",
                    help="host ip, localhost, 0.0.0.0")
parser_global.add_argument("--port",
                    type=int,
                    default=10095,
                    help="grpc server port")
parser_global.add_argument("--profile", action='store_true', help="print profiling result")
parser_global.add_argument("--streaming_output", action='store_true', default=False, help="whether to streaming output for file or audio")
parser_global.add_argument("--llm_type", type=str, default="minicpm-2b", help="llm model type, support minicpm-2b, llama3-8b")
parser_global.add_argument("--min_tts_input_len", type=int, default=6, help="minimum TTS input text length")
args_global, _ = parser_global.parse_known_args()

# model params loading
parser_whisper, parser_llama3, parser_minicpm, parser_tts = set_argparser('whisper'), set_argparser('llama3'), set_argparser('minicpm'), set_argparser('tts')
args_whisper, _ = parser_whisper.parse_known_args()
args_llama3, _ = parser_llama3.parse_known_args()
args_minicpm, _ = parser_minicpm.parse_known_args()
args_tts, _ = parser_tts.parse_known_args()

console, timer = Console(), Timer()

# ------ init 
timer = Timer()
print("model loading")
with timer:
	if args_global.llm_type == "minicpm-2b":
		assert os.path.exists(args_minicpm.minicpm_model_path)
		assert os.path.exists(args_minicpm.minicpm_tokenizer_model_path)
		llm_model = MinicpmModel(args_global=args_global, **args_minicpm.__dict__)
		# hot start~
		output = llm_model.gen_response_line("hello!")
	elif args_global.llm_type == "llama3-8b":
		llm_model = Llama3Model(args_global=args_global, **args_llama3.__dict__)
		# llm_model = None
	print("start LLM ...")
	# NOTE only support zh
	args_whisper.language = 'zh'
	whisper_model = whisperWrapper.WhisperWrapper(args_whisper)
	print("start S2T model ...")
	# whisper_model = None
	vits = VITS(args_tts)
	vits.init()
	print("start T2S model ...")
	pipeline = Pipeline(asr_model=whisper_model, tts_model=vits, llm_model=llm_model, args_global=args_global)
console.print(f"[blue]Initialization took {timer.run_time:.3f} seconds!")
subprocess.run('clear', shell=True)
print("model loaded")

websocket_users = set()
llm_out_queue = multiprocessing.Queue()
audio_in_queue = multiprocessing.Queue()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((args_global.host, args_global.port))
server_socket.listen(5)
client_socket = None
client_address = None

print("model loaded! only support one client at the same time now!!!!")


def infer_tts():
	global client_socket, pipeline, llm_out_queue
	try:
		pipeline.run_tts(llm_out_queue, client_socket)
		res_message = bytes("</end>".encode())
		client_socket.send(res_message)
	except Exception as e:
		print(e)


def serve():
	global client_socket, client_address, pipeline
	while True:
		try:
			print('waiting for connection ...')
			client_socket, client_address = server_socket.accept()
		except:
			print('connect fail!')
			sys.exit(-1)
		print(client_address, "new user connected", flush=True)
		while True:
			conn_close = False
			try:
				full_data = b''
				print("waiting new question ...")
				while True:
					data = client_socket.recv(1024)
					if not data:
						print(client_address[0] + ' connection close!')
						conn_close = True
						break
					try:
						if data[-6:].decode() == "</end>":
							full_data += data[:-6]
							break
						else:
							full_data += data
					except:
						full_data += data
				if len(full_data) != 0:
					audio_in_queue.put(full_data)
					p1 = multiprocessing.Process(target=infer_tts)  
					p1.start() 
					pipeline.run_asr_llm(audio_in_queue, llm_out_queue)
					p1.join()
			except ConnectionResetError:
				logger.error(client_address[0] + 'error link, try to recv data...')
				continue
			except KeyboardInterrupt:
				print('server closed!')
				break
			if conn_close:
				break
	server_socket.close()

serve()