import asyncio
import logging
import os
import time
from typing import Dict, Any
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from agentscope_runtime.engine.schemas.agent_schemas import (
    AgentRequest,
    Message,
    TextContent,
)

from dotenv import load_dotenv

load_dotenv()

from agent.scrapy_agent import agent_app


from agentscope_runtime.engine.deployers.local_deployer import (
    LocalDeployManager,
    DeploymentMode,
)




async def main():
    """Deploy app in detached process mode"""
    print("🚀 Deploying AgentApp in detached process mode...")

    # Create deployment manager
    deploy_manager = LocalDeployManager(
        host="0.0.0.0",
        port=8080,
    )

    # Deploy in detached mode:q
    deployment_info = await agent_app.deploy(
        deploy_manager,
        mode=DeploymentMode.DAEMON_THREAD,
    )
    print(f"✅ Deployment successful: {deployment_info['url']}")
    print(f"📍 Deployment ID: {deployment_info['deploy_id']}")



if __name__ == "__main__":
    asyncio.run(main())
    input("Press Enter to stop the server...")
