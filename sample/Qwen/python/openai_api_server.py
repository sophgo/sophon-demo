#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from sse_starlette.sse import EventSourceResponse
from qwen import Qwen
import uvicorn
import time
import asyncio
import argparse
import yaml

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]
    index: int

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

async def predict_stream(client, model_id:str, params: dict):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    for new_response in client.chat_stream_for_api(params):
        delta_text = new_response["text"]
        finish_reason = new_response["finish_reason"]
        
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )

        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        await asyncio.sleep(0.01)


def predict(client, model_id:str, params: dict):
    response = client.chat_for_api(params)
    message = ChatMessage(
        role="assistant",
        content=response["text"]
    )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=response["finish_reason"]
    )
    return ChatCompletionResponse(
        model=model_id,
        id="",  # for open_source model, id is empty
        choices=[choice_data],
        object="chat.completion",
    )

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global clients
    data = []
    for name in clients.keys():
        model_card = ModelCard(
            id=name
        )
        data.append(model_card)
    return ModelList(
        data=data
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global clients
    client = clients.get(request.model, None)
    if client == None:
        raise HTTPException(status_code=404, detail=f"model {request.model} not Found")
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    if request.stream:
        generate = predict_stream(client, request.model, request.messages)
        return EventSourceResponse(generate, media_type="text/event-stream")
    else:
        return predict(client, request.model, request.messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--config', type=str, default='./config/api.yaml', help='path of config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    models_config = config["models"]
    port = int(config["port"])
    clients = {}
    for model in models_config:
        name = model["name"]
        clients[name] = Qwen(model["bmodel_path"], model["dev_id"], model["token_path"])
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)