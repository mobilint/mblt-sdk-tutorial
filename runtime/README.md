# Mobilint Runtime Tutorial

Tutorials in this section provide detailed instructions for running models using the Mobilint runtime library.

<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>

## Runtime Preparation

The Mobilint qb runtime tutorial proceeds on the inference PC that is equipped with a Mobilint NPU.

> Note: The environments for the runtime do not need to be the same as the compiler environment. The runtime only requires a system equipped with a Mobilint NPU.

After the hardware connection, start the Mobilint NPU driver to enable access to the device.
Detailed instructions can be found in the [Driver installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation).

If the driver is successfully installed, you can enable NPU access inside a Docker container using the following flag:

```bash
--device /dev/aries0:/dev/aries0
```

Next, install the runtime library.
Refer to the [Runtime Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion) for more information.

Mobilint also provides a utility tool for checking NPU status, verifying MXQ files, and running simple inference tasks.
Refer to the [Utility Tool Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation) for details.

Now, you are ready to run your models!

Try the tutorials in current directory to run your compiled models on the Mobilint NPU.
