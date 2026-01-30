import asyncio
import base64
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from agentscope_runtime.engine.schemas.agent_schemas import (
    AgentRequest,
    FileContent,
    TextContent,
)
from config import mcp_servers_config, scrapy_agent_sys_prompt
from agentscope.tool import Toolkit
from agentscope.formatter import OpenAIChatFormatter
from agentscope.mcp import StdIOStatefulClient
from agentscope_runtime.engine.app import AgentApp
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.plan import PlanNotebook
from agentscope.pipeline import stream_printing_messages
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "../uploads")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file"

    # Remove leading/trailing whitespace
    filename = filename.strip()

    # Block path traversal patterns first (before any other processing)
    # Replace ".." with underscore anywhere in the filename
    filename = filename.replace("..", "_")

    # Preserve extension if filename starts with dot (e.g., ".pdf")
    if filename.startswith("."):
        ext = filename
        base_name = "file"
    else:
        # Split into base name and extension
        parts = filename.rsplit(".", 1)
        if len(parts) == 2:
            base_name, ext = parts[0], f".{parts[1]}"
        else:
            base_name, ext = filename, ""

    # Sanitize base name: replace special chars with underscore
    base_name = re.sub(r"[^\w\-]", "_", base_name)

    # Ensure base name is not empty after sanitization
    if not base_name or base_name.strip("_") == "":
        base_name = "file"

    return base_name + ext


def validate_file_size(file_data: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
    """Check if file data size is within limit (default 10MB).

    Args:
        file_data: File binary data
        max_size: Maximum allowed size in bytes (default: 10MB)

    Returns:
        True if file size is within limit, False otherwise
    """
    return len(file_data) <= max_size


def validate_mime_type(file_data: bytes) -> bool:
    """Validate MIME type using magic bytes.

    Since user requested "全部类型" (all types), this accepts all file types.
    For production, you could add more restrictive validation here.

    Args:
        file_data: File binary data

    Returns:
        Always True (accepting all types per user requirement)
    """
    # User requirement: accept all file types
    # If you need to restrict types, use mimetypes module here
    return True


def save_file_from_base64(file_data: str, filename: str) -> dict:
    """Save base64-encoded file to uploads directory.

    Args:
        file_data: Base64 string or data URL (data:mime/type;base64,...)
        filename: Original filename

    Returns:
        dict with keys: file_id, file_url, file_path, filename

    Raises:
        ValueError: If file size exceeds limit or base64 is invalid
    """
    # Ensure uploads directory exists
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Detect and extract base64 from data URL if needed
    if file_data.startswith("data:"):
        # Extract base64 portion from data URL
        # Format: data:mime/type;base64,xxx
        if ";base64," in file_data:
            mime_type, b64_data = file_data.split(";base64,", 1)
        else:
            # Data URL without base64 encoding
            raise ValueError("Invalid data URL format")
    else:
        # Standard base64 string
        b64_data = file_data

    # Decode base64 to bytes
    try:
        file_bytes = base64.b64decode(b64_data)
    except (base64.binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 data: {e}") from e

    # Validate file size (10MB limit)
    if not validate_file_size(file_bytes):
        raise ValueError(f"File size exceeds 10MB limit")

    # Generate UUID4 file_id
    file_id = str(uuid.uuid4())

    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Create file_id subdirectory
    file_dir = os.path.join(UPLOADS_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(file_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # Generate file URL
    file_url = f"http://localhost:8080/files/{file_id}"

    return {
        "file_id": file_id,
        "file_url": file_url,
        "file_path": file_path,
        "filename": safe_filename,
        "size": len(file_bytes),
    }


agent_app = AgentApp(
    app_name="scrapy_agent",
    app_description="Scrapy agent for web scraping",
    app_version="1.0.0",
)


@agent_app.init
async def init_func(self):
    logging.info("初始化 scrapy_agent 应用...")
    self.state_service = InMemoryStateService()
    await self.state_service.start()

    self.mcp_clients: dict[str, StdIOStatefulClient] = {}

    toolkit = Toolkit()
    for name, config in mcp_servers_config.items():
        logging.info(f"初始化MCP服务器: {name}")
        # 复用已连接的客户端（如果存在）
        if name in self.mcp_clients and self.mcp_clients[name].is_connected:
            mcp_client = self.mcp_clients[name]
            logging.info(f"使用已连接的 MCP 客户端: {name}")
        else:
            mcp_client = StdIOStatefulClient(name, **config)
            await mcp_client.connect()
            self.mcp_clients[name] = mcp_client
        await toolkit.register_mcp_client(mcp_client)
        logging.info(f"mcp 工具 {name} 注册成功 !")

    for skill_name in ["web_scraping", "data_extraction"]:
        skill_path = os.path.join(SKILLS_DIR, skill_name)
        if os.path.exists(skill_path):
            toolkit.register_agent_skill(skill_path)
            logging.info(f"Skill {skill_name} 注册成功 !")

    notebook = PlanNotebook()
    model_name = os.getenv("model_name")
    logging.info(f"创建 ReActAgent - Model: {model_name}")
    self.agent = ReActAgent(
        name="scrapy_agent",
        sys_prompt=scrapy_agent_sys_prompt,
        model=OpenAIChatModel(
            model_name=model_name,
            api_key=os.getenv("api_key"),
            client_kwargs={"base_url": os.getenv("base_url")},
        ),
        max_iters=20,
        toolkit=toolkit,
        plan_notebook=notebook,
        formatter=OpenAIChatFormatter(),
    )

    logging.info("初始化完成")


@agent_app.shutdown
async def shutdown_func(self):
    logging.info("关闭 scrapy_agent 应用...")
    await self.state_service.stop()
    for name, client in self.mcp_clients.items():
        if client.is_connected:
            logging.info(f"关闭 MCP 客户端: {name}")
            await client.close()
    self.mcp_clients.clear()
    logging.info("应用已关闭")


async def _load_agent_state(self, session_id: str, user_id: str) -> bool:
    """Load agent state from state service.

    Args:
        session_id: Session identifier
        user_id: User identifier

    Returns:
        True if state was loaded, False if no state existed
    """
    state = await self.state_service.export_state(
        session_id=session_id,
        user_id=user_id,
    )
    if state:
        logging.info(f"加载历史状态成功 - SessionID: {session_id}")
        self.agent.load_state_dict(state)
        logging.info(f"恢复 agent 状态 - SessionID: {session_id}")
        return True
    else:
        logging.info(f"无历史状态，创建新会话 - SessionID: {session_id}")
        return False


async def _save_agent_state(self, session_id: str, user_id: str) -> None:
    """Save current agent state to state service.

    Args:
        session_id: Session identifier
        user_id: User identifier
    """
    state = self.agent.state_dict()
    await self.state_service.save_state(
        user_id=user_id,
        session_id=session_id,
        state=state,
    )
    logging.info(f"保存状态成功 - SessionID: {session_id}")


def _process_file_content(item: FileContent, session_id: str) -> Optional[FileContent]:
    """Process a single FileContent item (save new uploads or validate existing).

    Args:
        item: FileContent to process
        session_id: Session identifier for logging

    Returns:
        Processed FileContent or None if processing failed
    """
    if item.file_data:
        try:
            filename = item.filename or "uploaded_file"
            result = save_file_from_base64(item.file_data, filename)
            item.file_id = result["file_id"]
            item.file_url = result["file_url"]
            logging.info(
                f"文件保存成功 - SessionID: {session_id}, "
                f"FileID: {result['file_id']}, "
                f"Filename: {result['filename']}, "
                f"Size: {result['size']} bytes"
            )
            return item
        except ValueError as e:
            logging.warning(f"文件处理失败 - SessionID: {session_id}, Error: {e}")
            return None
    elif item.file_url:
        logging.info(
            f"使用已有文件 - SessionID: {session_id}, FileURL: {item.file_url}"
        )
        return item
    else:
        logging.warning(
            f"文件内容无效 - SessionID: {session_id}, 缺少 file_data 和 file_url"
        )
        return None


def _process_message_content(msg: Msg, session_id: str) -> None:
    """Process message content, handling files and text.

    Args:
        msg: Message to process
        session_id: Session identifier for logging
    """
    if not msg or not hasattr(msg, "content") or not msg.content:
        return

    processed_content = []
    for item in msg.content:
        if isinstance(item, FileContent):
            processed_item = _process_file_content(item, session_id)
            if processed_item is not None:
                processed_content.append(processed_item)
        elif isinstance(item, TextContent):
            processed_content.append(item)
        else:
            processed_content.append(item)

    msg.content = processed_content


def _process_messages(msgs, session_id: str):
    """Process all messages, handling file uploads and content transformation.

    Args:
        msgs: Messages to process
        session_id: Session identifier for logging

    Returns:
        Processed messages
    """
    if msgs and hasattr(msgs, "__iter__"):
        processed_msgs = []
        for msg in msgs:
            _process_message_content(msg, session_id)
            processed_msgs.append(msg)
        return processed_msgs
    return msgs


def sync_handler(request: AgentRequest):
    yield {"status": "ok", "payload": request}


def save_file_from_binary(file_data: bytes, filename: str) -> dict:
    """Save binary file to uploads directory.

    Args:
        file_data: File binary data
        filename: Original filename

    Returns:
        dict with keys: file_id, file_url, file_path, filename

    Raises:
        ValueError: If file size exceeds limit
    """
    # Ensure uploads directory exists
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Validate file size (10MB limit)
    if not validate_file_size(file_data):
        raise ValueError(f"File size exceeds 10MB limit")

    # Generate UUID4 file_id
    file_id = str(uuid.uuid4())

    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Create file_id subdirectory
    file_dir = os.path.join(UPLOADS_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(file_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(file_data)

    # Generate file URL
    file_url = f"http://localhost:8080/files/{file_id}"

    return {
        "file_id": file_id,
        "file_url": file_url,
        "file_path": file_path,
        "filename": safe_filename,
        "size": len(file_data),
    }


@agent_app.endpoint("/files/{file_id}")
async def file_handler(file_id: str):
    """Serve uploaded files by file_id.

    Args:
        file_id: UUID of the uploaded file

    Returns:
        File content with appropriate Content-Type header
        or 404 error if file not found
    """
    file_dir = os.path.join(UPLOADS_DIR, file_id)

    # Check if directory exists
    if not os.path.exists(file_dir):
        yield {"error": "File not found", "status": 404}
        return

    # Find the file in the directory (there's only one file per directory)
    files = list(Path(file_dir).glob("*"))
    if not files or not files[0].is_file():
        yield {"error": "File not found", "status": 404}
        return

    file_path = files[0]

    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"

    # Read and yield file content
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Return file info and content
        yield {
            "file_id": file_id,
            "filename": file_path.name,
            "content_type": content_type,
            "size": len(content),
            "content": base64.b64encode(content).decode("utf-8"),
        }
    except Exception as e:
        logging.error(f"Error serving file {file_id}: {e}")
        yield {"error": str(e), "status": 500}


class UploadRequest(BaseModel):
    filename: str = "uploaded_file"
    file_data: str


@agent_app.endpoint("/upload")
async def upload_handler(body: UploadRequest):
    """Handle file upload from base64-encoded data.

    Args:
        body: Request body containing 'filename' and 'file_data' (base64 or data URL)

    Yields:
        dict with file_id, file_url, filename, size on success
        dict with error and status on failure
    """
    try:
        if not body.file_data:
            yield {"error": "Missing file_data", "status": 400}
            return

        # Use existing save_file_from_base64 function
        result = save_file_from_base64(body.file_data, body.filename)

        logging.info(
            f"文件上传成功 - FileID: {result['file_id']}, "
            f"Filename: {result['filename']}, Size: {result['size']} bytes"
        )

        yield result
    except ValueError as e:
        logging.error(f"文件上传失败: {e}")
        yield {"error": str(e), "status": 400}
    except Exception as e:
        logging.error(f"文件上传失败（未知错误）: {e}", exc_info=True)
        yield {"error": "Internal server error", "status": 500}


@agent_app.endpoint("/async")
async def async_handler(request: AgentRequest):
    yield {"status": "ok", "payload": request}


@agent_app.endpoint("/stream_async")
async def stream_async_handler(request: AgentRequest):
    for i in range(5):
        yield f"async chunk {i}, with request payload {request}\n"


@agent_app.endpoint("/stream_sync")
def stream_sync_handler(request: AgentRequest):
    for i in range(5):
        yield f"sync chunk {i}, with request payload {request}\n"


@agent_app.task("/task", queue="celery1")
def task_handler(request: AgentRequest):
    time.sleep(30)
    yield {"status": "ok", "payload": request}


@agent_app.task("/atask")
async def atask_handler(request: AgentRequest):
    await asyncio.sleep(15)
    yield {"status": "ok", "payload": request}


@agent_app.query(framework="agentscope")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    """Handle query requests for the agent.

    Args:
        msgs: Messages to process
        request: Agent request containing session and user info
        **kwargs: Additional keyword arguments

    Yields:
        Tuple of (message, is_last) from agent execution
    """
    assert kwargs is not None, "kwargs is Required for query_func"
    session_id = request.session_id
    user_id = request.user_id

    logging.info(f"收到查询请求 - SessionID: {session_id}, UserID: {user_id}")

    await _load_agent_state(self, session_id, user_id)

    msgs = _process_messages(msgs, session_id)

    logging.info(f"开始执行 agent 任务 - SessionID: {session_id}")
    try:
        async for msg, last in stream_printing_messages(
            agents=[self.agent],
            coroutine_task=self.agent(msgs),
        ):
            yield msg, last
        logging.info(f"agent 任务执行完成 - SessionID: {session_id}")
    except Exception as e:
        logging.error(
            f"agent 任务执行失败 - SessionID: {session_id}, Error: {e}", exc_info=True
        )
        raise

    await _save_agent_state(self, session_id, user_id)

    logging.info(f"查询请求处理完成 - SessionID: {session_id}")
