# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-12

### Added
- Initial public release of SynNeon.
- **Gaze Gesture Detection:** Real-time directional dwell and swipe event detection (UP, DOWN, LEFT, RIGHT) from eye-tracking streams.
- **Compass Keyboard:** AAC interface operable via eye gestures or keyboard controls, featuring AI-assisted abbreviation expansion.
- **Cognitive Steering:** Context-aware abbreviation expansion incorporating conversational history, user profile, and Redis-backed session memory.
- **Multimodal Context Server:** FastAPI REST endpoints supporting Vision-Language Model (VLM) scene analysis, gaze fixations, speech-to-text (STT), and text-to-speech (TTS).
- **Dual Provider Support:** Seamless support for OpenAI and Google Gemini across LLM, STT, and TTS pipelines.
- **Pupillometry:** Streaming pupil diameter measurements and real-time visualization tools.
- **Dashboard UI:** React/Vite web application providing live monitoring, pupillometry plots, and capture inspection.
- **Docker Stack:** Complete containerized environment with Redis, Redis Insight, and preconfigured mock hardware support.
