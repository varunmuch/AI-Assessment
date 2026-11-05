"""Task 1: MCP server over stdio with strict Pydantic validation.

Run: python -m task1_mcp_server.server
"""

import json
import logging
import sys
import uuid
from typing import Annotated

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# stdout is reserved for JSON-RPC; every log line goes to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("task1")

CustomerId = Annotated[str, Field(pattern=r"^CUST-[0-9]{5}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class GetCustomerRecordArgs(StrictModel):
    customer_id: CustomerId


class TriggerRefundArgs(StrictModel):
    customer_id: CustomerId
    amount: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    reason: Annotated[str, Field(min_length=10)]


CUSTOMERS = {
    "CUST-00001": {"customer_id": "CUST-00001", "name": "Ada Lovelace", "tier": "gold"},
    "CUST-00042": {"customer_id": "CUST-00042", "name": "Alan Turing", "tier": "silver"},
}

TOOL_MODELS: dict[str, type[StrictModel]] = {
    "get_customer_record": GetCustomerRecordArgs,
    "trigger_refund": TriggerRefundArgs,
}


async def list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, input_schema=model.model_json_schema())
            for name, model in TOOL_MODELS.items()
        ]
    )


def run_tool(name: str, args: StrictModel) -> str:
    if name == "get_customer_record":
        record = CUSTOMERS.get(args.customer_id)
        return json.dumps(record) if record else f"No customer {args.customer_id}"
    return (
        f"Refund {uuid.uuid4()} of {args.amount:.2f} queued for {args.customer_id}: {args.reason}"
    )


async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    model = TOOL_MODELS.get(params.name)
    if model is None:
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as exc:
        logger.info("Rejected %s call: %s", params.name, exc.errors(include_url=False))
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        raise MCPError(code=types.INVALID_PARAMS, message="Invalid params", data=details)
    logger.info("Executing %s", params.name)
    return types.CallToolResult(content=[types.TextContent(text=run_tool(params.name, args))])


server = Server("task1-customer-server", on_list_tools=list_tools, on_call_tool=call_tool)


def framing_error(exc: Exception) -> types.JSONRPCError:
    """The SDK drops lines it cannot decode; answer them as JSON-RPC requires (id: null)."""
    bad_json = isinstance(exc, ValidationError) and exc.errors()[0]["type"] == "json_invalid"
    code, message = (types.PARSE_ERROR, "Parse error") if bad_json else (types.INVALID_REQUEST, "Invalid Request")
    return types.JSONRPCError(jsonrpc="2.0", id=None, error=types.ErrorData(code=code, message=message))


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        relay_send, relay_recv = anyio.create_memory_object_stream(0)

        async def relay() -> None:
            async with relay_send:
                async for item in read_stream:
                    if isinstance(item, Exception):
                        error = framing_error(item)
                        logger.info("Rejected undecodable message with %s", error.error.code)
                        await write_stream.send(SessionMessage(error))
                    else:
                        await relay_send.send(item)

        async with anyio.create_task_group() as tg:
            tg.start_soon(relay)
            await server.run(relay_recv, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
