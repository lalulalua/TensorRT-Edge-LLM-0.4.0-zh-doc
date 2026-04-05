> **Modified:** This file was changed after the NVIDIA TensorRT-Edge-LLM 0.4.0 release for this unofficial distribution. See [PROJECT_ORIGIN.md](PROJECT_ORIGIN.md). The primary repository README is [README.md](README.md) (Chinese).

# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

---

## About this repository

This tree is a **non-official** redistribution based on the upstream **0.4.0** release of [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM). It is not maintained by NVIDIA. The maintainer of **this** Git repository is responsible for local changes and any releases from it. Original copyrights and the Apache-2.0 license from the upstream project still apply; see [LICENSE](LICENSE), [NOTICE](NOTICE), and [PROJECT_ORIGIN.md](PROJECT_ORIGIN.md).

---

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](docs/source/developer_guide/01.1_Overview.md). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](docs/source/developer_guide/01.2_Quick_Start_Guide.md).

---

## Documentation

### Developer Guide

Complete documentation for installation, usage, and deployment:

- **[Overview](docs/source/developer_guide/01.1_Overview.md)** - What is TensorRT Edge-LLM and key features
- **[Quick Start Guide](docs/source/developer_guide/01.2_Quick_Start_Guide.md)** - Get started in ~15 minutes
- **[Installation](docs/source/developer_guide/01.3_Installation.md)** - Detailed installation instructions
- **[Supported Models](docs/source/developer_guide/02_Supported_Models.md)** - Complete model compatibility matrix
- **[Python Export Pipeline](docs/source/developer_guide/03.1_Python_Export_Pipeline.md)** - Model export and quantization
- **[Engine Builder](docs/source/developer_guide/03.2_Engine_Builder.md)** - Building TensorRT engines
- **[C++ Runtime Overview](docs/source/developer_guide/04.1_C++_Runtime_Overview.md)** - Runtime system architecture
  - [LLM Inference Runtime](docs/source/developer_guide/04.2_LLM_Inference_Runtime.md)
  - [LLM SpecDecode Runtime](docs/source/developer_guide/04.3_LLM_Inference_SpecDecode_Runtime.md)
  - [Advanced Runtime Features](docs/source/developer_guide/04.4_Advanced_Runtime_Features.md)
- **[Examples](docs/source/developer_guide/05_Examples.md)** - Working code examples
- **[Chat Template Format](docs/source/developer_guide/06_Chat_Template_Format.md)** - Chat template configuration
- **[Customization Guide](docs/source/developer_guide/07_Customization_Guide.md)** - Extending and customizing the framework
- **[TensorRT Plugins](docs/source/developer_guide/08_TensorRT_Plugins.md)** - Introduction for TensorRT plugins.

### Additional Resources

- **[Examples Directory](examples/)** - LLM and VLM inference examples
- **[Tests](tests/)** - Comprehensive test suite for contributors

---

## Use Cases

**🚗 Automotive**
- In-vehicle AI assistants
- Voice-controlled interfaces
- Scene understanding
- Driver assistance systems

**🤖 Robotics**
- Natural language interaction
- Task planning and reasoning
- Visual question answering
- Human-robot collaboration

**🏭 Industrial IoT**
- Equipment monitoring with NLP
- Automated inspection
- Predictive maintenance
- Voice-controlled machinery

**📱 Edge Devices**
- On-device chatbots
- Offline language processing
- Privacy-preserving AI
- Low-latency inference

---

## Tech Blogs

*Coming soon*

Stay tuned for technical deep-dives, optimization guides, and deployment best practices.

---

## Latest News

*Coming soon*

For **NVIDIA's** upstream project, follow [NVIDIA/TensorRT-Edge-LLM on GitHub](https://github.com/NVIDIA/TensorRT-Edge-LLM). For updates that apply only to **this** unofficial copy, use the issue tracker or release pages of the Git hosting service where you obtained this repository.

---

## Support

- **Documentation**: [Developer Guide](docs/source/developer_guide/01.1_Overview.md)
- **Upstream NVIDIA project** (official source): [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues) and [Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **This unofficial repository**: use your Git host’s issue tracker for this repo (not NVIDIA support channels) for problems specific to this distribution.
- **NVIDIA Developer Forums** (general NVIDIA topics): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

Contributions may be proposed according to [CONTRIBUTING.md](CONTRIBUTING.md), which explains how this repository relates to the upstream NVIDIA project.

---
