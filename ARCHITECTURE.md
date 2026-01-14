## Core Freeze (v0.1)

The following modules are considered frozen core:

- core/llm/contract/*
- core/llm/engine.py
- core/conversation/session.py

They must NOT be modified for feature extensions.
All new capabilities must be implemented via plugins.
