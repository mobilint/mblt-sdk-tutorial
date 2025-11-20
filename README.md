# Mobilint SDK Tutorial

<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="./assets/Mobilint_Logo.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

This repository provides examples and explanations to help users easily get started with the Mobilint SDK, which includes the compiler (qubee) and the runtime software (maccel) library.

Models converted using the compiler can be executed on the Mobilint NPU through the runtime. When properly configured, this workflow enables models to achieve faster inference performance while maintaining the original model’s accuracy.

## Before you start

Before getting started, ensure that you have access to a Mobilint NPU.
If you don’t have one, please contact [us](mailto:contact@mobilint.com) to discuss evaluation options for your AI application.

The SDK is distributed through the [Mobilint Download Center](https://dl.mobilint.com/). Please sign up for an account before downloading the SDK.

## Overview

<div align="center">
<img src="./assets/Compiler.avif" width="45%", alt="Compiler Diagram">
<img src="./assets/Runtime.avif" width="45%", alt="Runtime Diagram">
</div>

Mobilint [SDK qb](https://www.mobilint.com/sdk-qb) consists of two main components: the [compiler](compilation/README.md) and the [runtime](runtime/README.md).

The Mobilint qb compiler converts models from popular deep learning frameworks into the Mobilint Model eXeCUtable (MXQ) format.
Using a pre-trained model and a calibration dataset, the compiler parses, quantizes, and optimizes the model for execution on the Mobilint NPU.

The Mobilint qb runtime enables execution of Mobilint-compiled models on the NPU.
Using the runtime library, you can integrate your compiled models into real-world applications in a simple and efficient way.

With the Mobilint qb SDK, you can compile your models and run them on the Mobilint NPU in a simple and efficient way.

For more information, please refer to the [compiler](compilation/README.md) and [runtime](runtime/README.md) tutorials.

## Support & Issues

If you encounter any issues while following this tutorial, please contact our [technical support email](mailto:tech-support@mobilint.com).
