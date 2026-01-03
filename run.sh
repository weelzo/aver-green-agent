#!/bin/bash
# AVER Benchmark - AgentBeats Runner
# This script is called by agentbeats run_ctrl
#
# AgentBeats sets these environment variables:
#   AGENT_PORT - The port to bind to
#   AGENT_URL  - The public URL for this agent

python main.py run
