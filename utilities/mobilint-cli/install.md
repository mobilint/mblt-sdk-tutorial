# Install mobilint-cli

English | [한국어](install.KR.md)

The CLI does not require Python or a virtual environment.

For Windows, follow the [official Windows installation instructions](https://docs.mobilint.com/runtime/en/v1.6/installing_utility/#option-4-installation-via-the-installer-windows).

## Prepare the device

- REGULUS EVK: The board comes with `mobilint-cli`. To update its packages, use the latest `Image Archive` for your board from the [Mobilint Download Center](https://dl.mobilint.com/). Updating the entire board image is recommended over updating individual packages.
- Ubuntu host: Install the driver for the connected NPU. For ARIES, follow the [driver installation guide](https://docs.mobilint.com/aries/en/latest/driver-installation/).
- Docker: Install the driver on the host. For the first ARIES device, check `ls -l /dev/aries0` on the host. Add `--device=/dev/aries0` to `docker run` when creating the container, then check the same device path inside it.

## Ubuntu: Install from the Mobilint APT repository

Run the commands in the environment where you want to use the CLI. You can use any working directory. If you are logged in as `root`, omit `sudo`.

If you already registered the Mobilint APT repository in that environment when installing the driver or runtime, start at step 3.

1. Install the packages needed for HTTPS downloads.

   ```bash
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   ```

2. Download the signing key and register the repository. Keep the entry that starts with `deb` on one line.

   ```bash
   sudo install -d -m 0755 /etc/apt/keyrings
   sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
   sudo chmod a+r /etc/apt/keyrings/mblt.asc
   echo 'deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse' | sudo tee /etc/apt/sources.list.d/mobilint.list
   ```

3. Update the package list and install the CLI.

   ```bash
   sudo apt-get update
   sudo apt-get install mobilint-cli
   ```

For other Linux installation methods, see the [utility installation guide](https://docs.mobilint.com/runtime/en/v1.6/installing_utility/#install-mobilint-cli).

## Verify the installation

```bash
mobilint-cli --version
```

If the command prints a version number, the CLI is installed.

To check whether it detects the NPU, run `devices` as shown in the [usage guide](usage.md).
